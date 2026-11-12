"""Random-stream independence (simulate.py, emulator, scenario_gen) and scenario_gen episode rates / durations."""
import copy
import json
import warnings
import numpy as np
import pytest
from dms_model_simulator import scenario_gen as SG
from dms_model_simulator.simulate import Simulation, rng_streams

CAM = "steering_column"
SEGS = [{"target": "road_ahead", "duration": 2.0}, {"target": "left_mirror", "duration": 1.0},
        {"target": "road_ahead", "duration": 3.0}]


def _scen(yawn_rate=0.0):
    segs = copy.deepcopy(SEGS)
    segs[2]["yawn"] = {"rate_per_min": yawn_rate}
    return {"name": "rng", "behavior": {}, "segments": segs}


def _pred(records):
    return [json.dumps(r["pred"], sort_keys=True, default=float) for r in records]


def test_streams_are_separate_children_of_the_seed():
    b, e = rng_streams(3)
    b2, e2 = rng_streams(3)
    assert b.random() == b2.random() and e.random() == e2.random()
    _, e3 = rng_streams(3, emulator_seed=3)                 # emulator_seed == seed is the default
    _, e4 = rng_streams(3)
    assert e3.random() == e4.random()


def test_emulator_seed_varies_model_noise_with_fixed_behaviour():
    a = Simulation(_scen(), CAM, seed=7); a.run()
    b = Simulation(_scen(), CAM, seed=7, emulator_seed=99); b.run()
    assert all(x.blink == y.blink and x.openness_left == y.openness_left for x, y in zip(a.frames, b.frames))
    assert _pred(a.records) != _pred(b.records)


def test_behaviour_draws_do_not_shift_emulator_noise():
    a = Simulation(_scen(0.0), CAM, seed=7); a.run()
    b = Simulation(_scen(30.0), CAM, seed=7); b.run()          # yawns in the last segment only
    n = int(round(3.0 * a.behavior.fps))                       # first two segments are identical
    assert _pred(a.records)[:n] == _pred(b.records)[:n]


def test_false_positive_rate_does_not_change_gaze_noise():
    a = Simulation(_scen(), CAM, seed=7); a.run()
    b = Simulation(_scen(), CAM, seed=7, emulator_overrides={"detection": {"false_positives": {"rate_per_frame": 0.0}}}); b.run()
    ga = [r["pred"]["gaze"] for r in a.records]
    gb = [r["pred"]["gaze"] for r in b.records]
    assert ga == gb


def _without_face_dir(records):
    return [json.dumps({k: v for k, v in r["pred"].items() if k != "face_dir"}, sort_keys=True, default=float)
            for r in records]


def test_face_dir_noise_has_its_own_stream():
    a = Simulation(_scen(), CAM, seed=7); a.run()
    b = Simulation(_scen(), CAM, seed=7, emulator_overrides={"face_dir": {"base_std_deg": 9.0, "temporal_rho": 0.2}}); b.run()
    assert _without_face_dir(a.records) == _without_face_dir(b.records)
    assert [r["pred"]["face_dir"] for r in a.records] != [r["pred"]["face_dir"] for r in b.records]


# pred values of _scen() at seed 7 and 20 fps (the emulator's reference_fps): the draws of the streams that existed
# before the face-direction stream was added (it is spawned after all of them, so they are what they were then)
# (frame: gaze yaw / pitch camera, head yaw, face box x1, face conf, left eye openness, mouth openness)
BEFORE_FACE_DIR = {
    10: (-1.0239830063265947, 14.33422206649113, 0.22177588412234192, 513.3803997066839, 0.957904717480573, 1.0,
         0.09883982898791255),
    50: (70.2724231368357, 14.169512451417075, 32.67290909306239, 552.5880475429418, 0.9727539400862267, 1.0,
         0.08691294511476916),
    90: (6.625410228254332, 13.303588722981747, 0.8042461087453845, 497.3470467353068, 0.9248393391041826,
         0.956205991794334, 0.08959010486316336),
}


def test_face_dir_stream_left_earlier_draws_unchanged():
    s = Simulation(_scen(), CAM, seed=7, fps=20); s.run()
    for k, exp in BEFORE_FACE_DIR.items():
        p = s.records[k]["pred"]
        face = next(d for d in p["detections"] if d["class"] == "face")
        got = (p["gaze"]["yaw_camera"], p["gaze"]["pitch_camera"], p["head_pose_camera"]["yaw"], face["bbox"][0],
               face["conf"], p["eye_openness"]["left"], p["mouth_openness"])
        assert got == pytest.approx(exp, rel=1e-9, abs=1e-9)


def _targets(cfg, seed=3, dur=600):
    return [s["target"] for s in SG.generate(cfg, dur, seed=seed)["segments"]]


def test_scenario_gen_kinds_have_own_streams():
    g = SG.load_gen_config()
    base = SG.generate(g, 600, seed=3)["generator"]
    g2 = copy.deepcopy(g); g2["smoking_episodes"]["rate_per_min"] = 1.0
    g2["phone_episodes"]["duration"]["mean_s"] = 12.0
    other = SG.generate(g2, 600, seed=3)["generator"]
    assert other["drowsy_episodes"] == base["drowsy_episodes"]
    assert other["seatbelt_off_episodes"] == base["seatbelt_off_episodes"]
    assert other["smoking_episodes"] != base["smoking_episodes"]
    # kinds without target_bias (smoking, seatbelt-off) leave the target chain alone
    g3 = copy.deepcopy(g); g3["smoking_episodes"]["rate_per_min"] = 0; g3["seatbelt_off_episodes"]["rate_per_min"] = 2.0
    t0, t3 = _targets(g), _targets(g3)
    merge = lambda ts: [t for i, t in enumerate(ts) if i == 0 or ts[i - 1] != t]
    assert merge(t0) == merge(t3)


def test_scenario_gen_episode_rate_is_rate_per_min():
    g = SG.load_gen_config()
    c = g["drowsy_episodes"]
    res, T, S = float(g["time_resolution_s"]), 3600.0, 120
    n = sum(len(SG._episodes(np.random.default_rng([s, 5]), c, int(T / res), res, None)) for s in range(S))
    rate = n / (S * T / 60)
    assert rate == pytest.approx(c["rate_per_min"], rel=0.06)     # was 1 / (60 / rate + mean duration)


def test_scenario_gen_unreachable_rate_warns():
    c = {"rate_per_min": 10.0, "duration": {"mean_s": 10.0, "sigma": 0.1, "min_s": 5.0, "max_s": 20.0}}
    with pytest.warns(UserWarning, match="back to back"):
        eps = SG._episodes(np.random.default_rng(0), c, 600, 0.1, None)
    assert all(b1 <= a2 for (_, b1, _), (a2, _, _) in zip(eps, eps[1:]))


def test_scenario_gen_duration_rounding_and_minimum():
    g = SG.load_gen_config()
    for dur, exp in ((5.05, 5.1), (5.0, 5.0), (0.06, 0.1)):
        s = sum(x["duration"] for x in SG.generate(g, dur, seed=3)["segments"])
        assert s == pytest.approx(exp, abs=1e-6)
    with pytest.raises(ValueError, match="at least one tick"):
        SG.generate(g, 0.04, seed=3)

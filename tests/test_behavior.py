import numpy as np
import pytest
from dms_model_simulator import scenario_gen
from dms_model_simulator.behavior import BehaviorGenerator
from dms_model_simulator.config import deep_merge, load_configs, load_scenario
from dms_model_simulator.events import label_events
from dms_model_simulator.simulate import Simulation
from dms_model_simulator.world import Cabin


def _frames(scenario, seed=0):
    cfgs = load_configs(); cabin = Cabin(cfgs["cabin"])
    beh = BehaviorGenerator(cabin, cfgs["behavior"], scenario, np.random.default_rng(seed))
    return cabin, beh, beh.generate()


def test_yawn_raises_mouth_openness():
    base = {"name": "t", "segments": [{"target": "road_ahead", "duration": 6.0}]}
    _, beh, quiet = _frames(base)
    assert not any(f.yawning for f in quiet)
    assert max(f.mouth_openness for f in quiet) == pytest.approx(beh.cfg["mouth_openness"])
    yawny = {"name": "t", "segments": [{"target": "road_ahead", "duration": 6.0, "yawn": {"rate_per_min": 14}}]}
    _, beh, frames = _frames(yawny)
    assert any(f.yawning for f in frames)
    peak = max(f.mouth_openness for f in frames)
    assert peak > beh.cfg["mouth_openness"] + 0.5
    assert peak <= beh.cfg["yawn"]["peak_openness"] + 1e-9
    assert all(f.mouth_openness == pytest.approx(beh.cfg["mouth_openness"]) for f in frames if not f.yawning)


LABEL_KEYS = {"eyes_off_road", "off_road_s", "eyes_closed", "closed_s", "phone_use", "yawning", "smoking",
              "seatbelt_off"}


def test_glance_to_phone_lap_labels_off_road_and_counts_its_duration():
    sc = {"name": "t", "segments": [{"target": "road_ahead", "duration": 2.0},
                                    {"target": "phone_lap", "duration": 3.0, "hand": "phone", "phone": "lap"},
                                    {"target": "road_ahead", "duration": 2.0}]}
    cabin, beh, frames = _frames(sc)
    ev = label_events(frames, cabin, beh.cfg)
    assert len(ev) == len(frames) and set(ev[0]) == LABEL_KEYS
    dt = 1.0 / beh.fps
    k = 0
    for f, e in zip(frames, ev):
        if f.target_name == "road_ahead":
            k = 0
            assert not e["eyes_off_road"] and e["off_road_s"] == 0
        else:
            k += 1                                                   # counts the current frame
            assert e["eyes_off_road"] and e["phone_use"]
            assert e["off_road_s"] == pytest.approx(k * dt, abs=1e-6)
    assert max(e["off_road_s"] for e in ev) == pytest.approx(3.0, abs=1e-6)
    assert not ev[0]["phone_use"] and not any(e["smoking"] or e["seatbelt_off"] for e in ev)


def test_drowsy_scenario_records_carry_raw_behaviour_labels():
    sim = Simulation("drowsy", "cluster", seed=0); sim.run()
    ev = [r["truth"]["events"] for r in sim.records]
    assert set(ev[0]) == LABEL_KEYS | {"phone_use_visible"}          # raw labels only, no alert-like decisions
    fps = sim.behavior.fps
    assert any(e["yawning"] for e in ev)
    assert max(e["closed_s"] for e in ev) >= 1.5                     # the scripted 2 s micro-sleep
    assert max(e["closed_s"] for e in ev[: int(3 * fps)]) < 0.5      # alert first segment: blinks only
    for e in ev:
        assert e["eyes_closed"] == (e["closed_s"] > 0)


def test_generator_reproducible_and_valid():
    cfg = scenario_gen.load_gen_config()
    a = scenario_gen.generate(cfg, 120, seed=1)
    b = scenario_gen.generate(cfg, 120, seed=1)
    c = scenario_gen.generate(cfg, 120, seed=2)
    assert a == b and a["segments"] != c["segments"]
    cabin = Cabin(load_configs()["cabin"])
    for seg in a["segments"]:
        assert seg["target"] in cabin.targets and seg["duration"] > 0
    assert sum(s["duration"] for s in a["segments"]) == pytest.approx(120.0, abs=1e-6)
    assert len({s["target"] for s in a["segments"]}) > 3


def test_generated_scenario_roundtrips_and_runs(tmp_path):
    cfg = scenario_gen.load_gen_config()
    sc = scenario_gen.generate(cfg, 20, seed=4)
    p = scenario_gen.save_yaml(sc, tmp_path / "rnd.yaml")
    sim = Simulation(str(p), "steering_column", seed=0); sim.run()
    assert len(sim.records) == int(round(20 * sim.behavior.fps))
    assert sim.scenario["segments"] == sc["segments"]


def _label_run(cabin, cfg, n_before, n_run, n_after, **run_fields):
    """Hand-made frames: n_run frames with run_fields (e.g. target_name / openness) between plain frames."""
    from dms_model_simulator.behavior import DriverFrame
    dt = 1.0 / cfg["fps"]
    frames = []
    for i in range(n_before + n_run + n_after):
        f = DriverFrame(i * dt, 0, 0, 0, cabin.targets["road_ahead"], "road_ahead", 1.0, 1.0, False, False, "wheel", "none")
        if n_before <= i < n_before + n_run:
            for k, v in run_fields.items():
                setattr(f, k, v)
        frames.append(f)
    return label_events(frames, cabin, cfg)


@pytest.mark.parametrize("n_run", [1, 2, 45])
def test_off_road_run_duration_is_frame_exact(n_run):
    """Every off-road frame is labelled (no minimum duration); off_road_s counts the run's frames and resets."""
    cfgs = load_configs(); cabin = Cabin(cfgs["cabin"]); cfg = cfgs["behavior"]
    dt = 1.0 / cfg["fps"]
    ev = _label_run(cabin, cfg, 10, n_run, 10, target_name="phone_lap")
    assert [e["eyes_off_road"] for e in ev] == [False] * 10 + [True] * n_run + [False] * 10
    assert [e["off_road_s"] for e in ev[10:10 + n_run]] == pytest.approx([(i + 1) * dt for i in range(n_run)], abs=1e-6)
    assert ev[10 + n_run]["off_road_s"] == 0
    on_road = _label_run(cabin, cfg, 2, 3, 2, target_name=cfg["events"]["on_road_targets"][-1])
    assert not any(e["eyes_off_road"] for e in on_road)


def test_eyes_closed_threshold_is_strict_and_closed_s_counts_frames():
    cfgs = load_configs(); cabin = Cabin(cfgs["cabin"]); cfg = cfgs["behavior"]
    thr, dt = cfg["events"]["closed_openness_threshold"], 1.0 / cfg["fps"]
    at = _label_run(cabin, cfg, 5, 10, 5, openness_left=thr, openness_right=thr)
    assert not any(e["eyes_closed"] for e in at)                     # exactly at the threshold: open
    ev = _label_run(cabin, cfg, 5, 10, 5, openness_left=thr - 0.01, openness_right=thr - 0.01)
    assert [e["eyes_closed"] for e in ev] == [False] * 5 + [True] * 10 + [False] * 5
    assert ev[14]["closed_s"] == pytest.approx(10 * dt, abs=1e-6) and ev[15]["closed_s"] == 0
    one = _label_run(cabin, cfg, 5, 10, 5, openness_left=0.0, openness_right=2 * thr)   # mean of both eyes
    assert not any(e["eyes_closed"] for e in one)


def test_phone_distraction_off_road_runs_match_the_script():
    """Off-road runs are exactly the scripted glance segments (2-5 s and 6-8 s to the lap, 11-12.5 s to the
    passenger), with no one-frame labels at the segment boundaries."""
    _, beh, frames = _frames(load_scenario("phone_distraction"))
    from dms_model_simulator.metrics import segments
    ev = label_events(frames, None, beh.cfg)
    runs = segments([e["eyes_off_road"] for e in ev])
    fps = beh.fps
    assert [(round(a / fps, 3), round(b / fps, 3)) for a, b in runs] == [(2.0, 5.0), (6.0, 8.0), (11.0, 12.5)]
    phone = segments([e["phone_use"] for e in ev])
    assert [(round(a / fps, 3), round(b / fps, 3)) for a, b in phone] == [(2.0, 12.5)]


def test_event_durations_are_frame_exact():
    # rates are events per minute of driving, so they must fit: 60 x 0.3 s + 10 x 1.5 s = 33 s per minute
    sc = {"name": "t", "behavior": {"blink": {"rate_per_min": 60, "duration_s": 0.3}},
          "segments": [{"target": "road_ahead", "duration": 60.0, "closure": {"rate_per_min": 10, "duration_s": 1.5},
                        "yawn": {"rate_per_min": 6, "duration_s": 4.0}}]}
    _, beh, frames = _frames(sc, seed=5)
    from dms_model_simulator.metrics import segments
    for attr, dur in (("blink", 0.3), ("closure", 1.5), ("yawning", 4.0)):
        n = int(round(dur * beh.fps))
        runs = [b - a for a, b in segments([getattr(f, attr) for f in frames]) if b < len(frames)]   # not cut off
        assert runs and min(runs) == n and all(r % n == 0 for r in runs), (attr, runs[:5])   # back-to-back events merge
    # a blink reaches its minimum in the middle and every blink frame is actually (partly) closed
    i = next(k for k, f in enumerate(frames) if f.blink)
    ops = [f.openness_left for f in frames[i:i + 9]]
    assert max(ops) < beh.cfg["eye_openness"] and ops.index(min(ops)) == 4


def test_saccades_are_smooth_in_angle_and_eyes_stay_within_the_limit():
    from dms_model_simulator import geometry as G
    cabin, beh, frames = _frames(load_scenario("phone_distraction"))
    lim = beh.cfg["eye_in_head_limit_deg"]
    dirs = []
    for f in frames:
        head = cabin.head_state(f.head_yaw, f.head_pitch, f.head_roll)
        d = G.normalize(f.gaze_point - head.origin)
        e_yaw, e_pitch = G.dir_to_yaw_pitch(head.R.T @ d)
        assert abs(e_yaw) <= lim["yaw"] + 1e-6 and abs(e_pitch) <= lim["pitch"] + 1e-6
        dirs.append(d)
    steps = [G.angle_between(a, b) for a, b in zip(dirs, dirs[1:])]
    assert max(steps) < 25.0                                    # was ~70 deg in one frame (3D point lerp)
    # steady state (end of every segment): exactly at the scripted target
    for s in beh.segments:
        f = frames[s["end_frame"] - 1]
        o = cabin.head_state(f.head_yaw, f.head_pitch, f.head_roll).origin
        assert G.angle_between(f.gaze_point - o, cabin.targets[s["target"]] - o) < 1e-6


def test_segment_boundaries_are_frame_exact():
    from fractions import Fraction
    cfgs = load_configs(); cabin = Cabin(cfgs["cabin"]); fps = cfgs["behavior"]["fps"]
    for durs in ([0.1, 0.1, 0.1, 2.1, 1.0], [0.35, 0.35, 0.35], [1.0, 0.01, 1.0], [4.7, 0.8, 4.6, 0.7, 2.9, 1.4]):
        sc = {"name": "t", "segments": [{"target": "road_ahead" if i % 2 == 0 else "left_mirror", "duration": d}
                                        for i, d in enumerate(durs)]}
        beh = BehaviorGenerator(cabin, cfgs["behavior"], sc, np.random.default_rng(0))
        ends = np.cumsum([Fraction(str(d)) for d in durs])
        assert beh.n_frames == -(-ends[-1] * fps // 1)                  # every instant k / fps < duration
        expect = [next((k for k, e in enumerate(ends) if Fraction(i, fps) < e), len(durs) - 1)   # start <= k/fps < end
                  for i in range(beh.n_frames)]
        assert [beh._seg_at_frame(i) for i in range(beh.n_frames)] == expect
        assert [beh._seg_at(i / fps) for i in range(beh.n_frames)] == expect
    _, beh, frames = _frames({"name": "t", "segments": [{"target": "road_ahead", "duration": 0.1}] * 3 +
                              [{"target": "left_mirror", "duration": 2.1}, {"target": "road_ahead", "duration": 1.0}]})
    names = [f.target_name for f in frames]
    assert names.index("left_mirror") == 9 and names.count("left_mirror") == 63


def test_event_rates_are_events_per_minute():
    """rate_per_min is the measured long-run event rate (hazard over event-free time), with blinks and closures
    excluding each other and yawns independent."""
    from dms_model_simulator.metrics import segments
    minutes = 40
    sc = {"name": "t", "segments": [{"target": "road_ahead", "duration": 60.0 * minutes,
                                     "closure": {"rate_per_min": 12, "duration_s": 1.5}, "yawn": {"rate_per_min": 6}}]}
    _, beh, frames = _frames(sc, seed=3)
    for attr, rate in (("blink", beh.cfg["blink"]["rate_per_min"]), ("closure", 12), ("yawning", 6)):
        measured = len(segments([getattr(f, attr) for f in frames])) / minutes
        assert abs(measured / rate - 1) < 0.08, (attr, measured, rate)
    assert not any(f.blink and f.closure for f in frames)


def test_even_length_blink_reaches_min_openness():
    sc = {"name": "t", "behavior": {"blink": {"rate_per_min": 30, "duration_s": 0.2}},
          "segments": [{"target": "road_ahead", "duration": 30.0}]}
    _, beh, frames = _frames(sc, seed=2)
    assert beh._n_frames(0.2) % 2 == 0
    from dms_model_simulator.metrics import segments
    for a, b in segments([f.blink for f in frames]):
        ops = [f.openness_left for f in frames[a:b]]
        if b - a == beh._n_frames(0.2):
            assert min(ops) == pytest.approx(beh.cfg["blink"]["min_openness"]) and max(ops) < beh.cfg["eye_openness"]


def test_event_kinds_use_independent_random_streams():
    segs = [{"target": "road_ahead", "duration": 4.0}, {"target": "left_mirror", "duration": 1.0},
            {"target": "road_ahead", "duration": 4.0}, {"target": "road_ahead", "duration": 6.0}]
    _, _, quiet = _frames({"name": "t", "segments": segs}, seed=7)
    segs[2] = {**segs[2], "yawn": {"rate_per_min": 12}}
    _, _, yawny = _frames({"name": "t", "segments": segs}, seed=7)
    assert any(f.yawning for f in yawny)
    assert [(f.blink, f.openness_left) for f in quiet] == [(f.blink, f.openness_left) for f in yawny]


def test_short_segment_head_moves_without_jumps():
    """A segment shorter than transition_head_s: the next transition starts from the actual head pose."""
    sc = {"name": "t", "segments": [{"target": "road_ahead", "duration": 1.0}, {"target": "left_mirror", "duration": 0.2},
                                    {"target": "road_ahead", "duration": 1.0}]}
    cabin, beh, frames = _frames(sc)
    yaw = np.array([f.head_yaw for f in frames])
    amp = abs(beh._head_pose_for(sc["segments"][1])[0] - beh._head_pose_for(sc["segments"][0])[0])
    max_step = 1.5 * amp / beh.cfg["transition_head_s"] / beh.fps          # smoothstep peak slope
    assert np.abs(np.diff(yaw)).max() <= max_step * 1.01
    gz = np.array([f.gaze_point for f in frames])
    assert np.isfinite(gz).all()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_drowsy_scenario_has_a_long_closure_for_any_seed(seed):
    _, beh, frames = _frames(load_scenario("drowsy"), seed=seed)
    assert max(e["closed_s"] for e in label_events(frames, None, beh.cfg)) >= 1.5   # scripted micro-sleep


def test_generator_smoking_and_seatbelt_episodes():
    cfg = scenario_gen.load_gen_config()
    sc = scenario_gen.generate(cfg, 300, seed=3)
    gen = sc["generator"]
    assert gen["smoking_episodes"] and gen["seatbelt_off_episodes"]
    segs = sc["segments"]
    assert {s.get("cigarette") for s in segs} >= {"mouth", "hand"}
    assert any(s.get("seatbelt") == "off" for s in segs)
    modes = set(cfg["smoking_episodes"]["modes"])
    assert all(m in modes for _, _, m in gen["smoking_episodes"])
    # a phone episode keeps its hand state where a smoking episode overlaps it
    for s in segs:
        if s.get("phone") == "lap":
            assert s["hand"] == "phone"
    # the new kinds use their own random streams: targets / phone / drowsy episodes of a seed are unchanged
    old = scenario_gen.generate({k: v for k, v in cfg.items() if k not in ("smoking_episodes", "seatbelt_off_episodes")},
                                300, seed=3)
    assert [s["target"] for s in old["segments"]] == [s["target"] for s in _merge_same(segs, old["segments"])]
    assert old["generator"]["phone_episodes"] == gen["phone_episodes"]
    assert scenario_gen.generate(cfg, 300, seed=3) == sc
    # the generated scenario runs (seatbelt 'off' survives the YAML round trip)
    sim_cfg = load_configs(); cabin = Cabin(sim_cfg["cabin"])
    frames = BehaviorGenerator(cabin, sim_cfg["behavior"], sc, np.random.default_rng(0)).generate()
    ev = label_events(frames, cabin, sim_cfg["behavior"])
    assert any(e["smoking"] for e in ev) and any(e["seatbelt_off"] for e in ev)


def _merge_same(segs, ref):
    """Re-split `segs` (extra cuts at smoking / seatbelt boundaries) onto the visits of `ref` by time."""
    out, t, edges = [], 0.0, []
    for s in segs:
        edges.append((round(t, 6), s)); t += s["duration"]
    t = 0.0
    for r in ref:
        out.append(next(s for t0, s in edges if t0 >= round(t, 6) - 1e-9))
        t += r["duration"]
    return out


def test_object_held_in_the_mouth_needs_no_hand():
    _, _, frames = _frames({"name": "t", "segments": [{"target": "road_ahead", "duration": 1.0, "held": "toothpick"}]})
    assert {f.held for f in frames} == {"toothpick"} and {f.hand for f in frames} == {"wheel"}


def test_held_object_follows_the_segment_and_needs_the_hand_at_the_mouth():
    sc = {"name": "t", "segments": [{"target": "road_ahead", "duration": 1.0},
                                    {"target": "road_ahead", "duration": 1.0, "hand": "at_mouth", "held": "cup"}]}
    _, beh, frames = _frames(sc)
    n = len(frames) // 2
    assert {f.held for f in frames[:n]} == {"none"} and {f.held for f in frames[n:]} == {"cup"}
    with pytest.raises(ValueError, match="at_mouth"):
        _frames({"name": "t", "segments": [{"target": "road_ahead", "duration": 1.0, "hand": "lap", "held": "cup"}]})
    with pytest.raises(ValueError, match="mug"):
        _frames({"name": "t", "segments": [{"target": "road_ahead", "duration": 1.0, "hand": "at_mouth", "held": "mug"}]})


def test_head_turned_beyond_the_eyes_reach_is_off_road_even_with_a_road_target():
    # head 60 deg right, scripted target road_ahead: the eyes stop at the eye-in-head limit, the gaze leaves the
    # road box (for longer than gaze_miss_min_s) -> eyes_off_road; a 20 deg turn keeps the gaze on the road
    def run(yaw):
        sc = {"name": "t", "behavior": {"transition_head_s": 0.1},
              "segments": [{"target": "road_ahead", "duration": 2.0},
                           {"target": "road_ahead", "duration": 2.0, "head_pose": [yaw, 0, 0]},
                           {"target": "road_ahead", "duration": 2.0}]}
        cabin, beh, frames = _frames(sc)
        return label_events(frames, cabin, beh.cfg), label_events(frames, None, beh.cfg)
    geo, named = run(-90)
    assert max(e["off_road_s"] for e in geo) > 1.5 and not any(e["eyes_off_road"] for e in named)
    geo, _ = run(-20)
    assert not any(e["eyes_off_road"] for e in geo)


def test_generator_extra_episode_kinds_and_mode_durations():
    from dms_model_simulator import scenario_gen as G
    base = G.load_gen_config()
    assert [k for k, _ in G.episode_kinds(base)][-1] == "habit_episodes"
    plain = G.generate(base, 120, 3)
    assert "habit_episodes" not in plain["generator"]                    # rate 0: no trace at all
    on = G.generate(deep_merge(base, {"habit_episodes": {"rate_per_min": 6}}), 120, 3)
    eps = on["generator"]["habit_episodes"]
    assert eps and all(len(e) == 3 for e in eps)
    # the built-in kinds draw exactly as before (own streams); habits only add attributes
    for k in ("phone_episodes", "drowsy_episodes", "smoking_episodes", "seatbelt_off_episodes"):
        assert on["generator"][k] == plain["generator"][k]
    # per-mode durations: talking lasts longer than rubbing an eye
    talk = [b - a for a, b, m in eps if m == "talk"]
    rub = [b - a for a, b, m in eps if m == "rub_eye"]
    if talk and rub:
        assert min(talk) >= 2.0 - 1e-9 and max(rub) <= 2.0 + 1e-9
    # habits never overwrite what an earlier kind set: a phone call keeps hand: call
    for s in on["segments"]:
        if s.get("phone") == "call":
            assert s["hand"] == "call"

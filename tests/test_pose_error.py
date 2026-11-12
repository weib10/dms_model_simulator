"""Structured head-pose / face-direction / eye-gaze error (Emulator._pose_error): per-run bias, fast jitter,
outlier runs, the per-run eye-openness bias, and the pg profiles. All of it is off by default."""
import copy
import json
import numpy as np
import pytest
from dms_model_simulator import catalog as C
from dms_model_simulator.camera import Camera
from dms_model_simulator.config import deep_merge, load_configs
from dms_model_simulator.emulator import Emulator, at_fps, capped_markov_step
from dms_model_simulator.simulate import Simulation
from dms_model_simulator.world import Cabin

CAM = "steering_column"
CFGS = load_configs()
PROFILES = C.load_profiles()
# every box found, no drops, slow noise / view terms / constant bias of all three quantities at 0 (pose0)
QUIET = deep_merge(deep_merge(PROFILES["ideal"], PROFILES["pose0"]), {"perfect_detection": True})


@pytest.fixture(scope="module")
def base():
    s = Simulation({"name": "t", "segments": [{"target": "road_ahead", "duration": 0.1}]}, CAM, seed=0)
    s.run()
    return s.records[0]["truth"]


@pytest.fixture(scope="module")
def rig():
    cab = Cabin(CFGS["cabin"])
    return cab, Camera.from_config(CAM, CFGS["cameras"], cab.nominal_head_center())


def _emu(rig, over, seed=0, fps=None):
    cfg = deep_merge(deep_merge(CFGS["emulator"], QUIET), over)
    if fps is not None:
        cfg = at_fps(cfg, fps)
    return Emulator(cfg, *rig, np.random.default_rng(seed))


def _errors(out, truth):
    """Per quantity the error vector of one frame's output against a static truth."""
    hp, tp = out["head_pose_camera"], truth["head_pose_camera"]
    g, fd = out["gaze"], out["face_dir"]
    return {"head_pose": np.array([hp["yaw"] - tp["yaw"], hp["pitch"] - tp["pitch"], hp["roll"] - tp["roll"]]),
            "gaze": np.array([g["yaw_camera"] - truth["gaze"]["yaw_camera"],
                              g["pitch_camera"] - truth["gaze"]["pitch_camera"]]),
            "face_dir": np.array([fd["yaw_camera"] - truth["face_dir"]["yaw_camera"],
                                  fd["pitch_camera"] - truth["face_dir"]["pitch_camera"]])}


# ---- default output unchanged ---------------------------------------------------------------------------------
# pred values of mirror_check (seed 3) before the structured error existed: frame -> (gaze yaw / pitch camera, head
# yaw / pitch, face-direction yaw camera, face box x1, left eye openness, gaze conf)
PINNED = {
    ("nominal", 20): {
        10: (-3.9268678178364462, 8.617925138778988, 1.5933074754877312, 14.980421896939236, 0.8733141323437968, 521.1122655455431, 0.9573567711685178, 0.646709262380557),
        40: (-2.3666210576939433, 11.209912949482796, 0.7777401951189463, 10.797653079954134, -1.6979521878960513, 503.9644380546462, 1.0, 0.646709262380557),
        70: (1.687787458975165, 9.08570902940002, 1.8643026114613772, 13.411404011382524, 4.13720194823909, 499.1330777693585, 0.9799145742773254, 0.646709262380557),
    },
    ("nominal", 25): {
        10: (-3.1617745102578114, 8.284546182747622, 1.6497160940600346, 14.966252616422484, 1.033980973377562, 520.9962171038039, 0.9578578330763935, 0.646709262380557),
        40: (-3.11170499983225, 11.208360715031038, 0.9543387803535796, 10.98714221136914, -1.5504647591764018, 502.5757319329538, 1.0, 0.646709262380557),
        70: (57.213821094220116, -4.283367804334878, 40.45178576317884, 9.119202909137584, 43.24990058873135, 545.002891945069, 0.9804581764361909, 0.4068119778089192),
    },
    ("ideal", 20): {
        10: (-3.9268678178364462, 8.617925138778988, 1.5933074754877312, 14.980421896939236, 0.8733141323437968, 521.1122655455431, 1.0, 0.646709262380557),
        40: (0.794943209565871, 9.082067030828236, 0.8328180255644967, 9.96747968165192, -1.4651990593452464, 511.3852765451221, 1.0, 0.646709262380557),
        70: (1.7406642099444287, 9.971758187766815, 1.2718126678169013, 12.587262944382143, 2.697001837263318, 501.4327860434855, 1.0, 0.646709262380557),
    },
    ("ideal", 25): {
        10: (-3.1617745102578114, 8.284546182747622, 1.6497160940600346, 14.966252616422484, 1.033980973377562, 520.9962171038039, 1.0, 0.646709262380557),
        40: (-0.22767697757578595, 9.269175566755042, 1.0137679685538847, 10.230998145726508, -1.309659986087122, 509.6694575960946, 1.0, 0.646709262380557),
        70: (57.733851528419216, -3.007856648990925, 39.60478383147995, 8.011498003563924, 41.38742385556165, 546.3046107148551, 1.0, 0.4068119778089192),
    },
    ("jit3", 20): {
        10: (-3.9268678178364462, 8.617925138778988, 1.5933074754877312, 14.980421896939236, 0.8733141323437968, 554.3986380311934, 0.9573567711685178, 0.646709262380557),
        40: (-2.3666210576939433, 11.209912949482796, 0.7777401951189463, 10.797653079954134, -1.6979521878960513, 519.4955918662876, 1.0, 0.646709262380557),
        70: (1.9190508255324301, 8.648996329742383, 1.8643026114613772, 13.411404011382524, 4.13720194823909, 437.7334708523523, 1.0, 0.646709262380557),
    },
    ("jit3", 25): {
        10: (-3.1617745102578114, 8.284546182747622, 1.6497160940600346, 14.966252616422484, 1.033980973377562, 561.3652616339668, 0.9578578330763935, 0.646709262380557),
        40: (-3.11170499983225, 11.208360715031038, 0.9543387803535796, 10.98714221136914, -1.5504647591764018, 516.3477268484214, 1.0, 0.646709262380557),
        70: (57.213821094220116, -4.283367804334878, 40.45178576317884, 9.119202909137584, 43.24990058873135, 401.3020042557649, 0.9804581764361909, 0.4068119778089192),
    },
    ("exact", 20): {
        10: (0.0, 11.474291689029384, 0.0, 11.474291689029384, 0.0, 521.1122655455431, 1.0, 0.9),
        40: (0.0, 11.474291689029384, 0.0, 11.474291689029384, 0.0, 511.3852765451221, 1.0, 0.9),
        70: (0.0, 11.474291689029384, 0.0, 11.474291689029384, 0.0, 501.4327860434855, 1.0, 0.9),
    },
    ("exact", 25): {
        10: (0.0, 11.474291689029384, 0.0, 11.474291689029384, 0.0, 520.9962171038039, 1.0, 0.9),
        40: (0.0, 11.474291689029384, 0.0, 11.474291689029384, 0.0, 509.6694575960946, 1.0, 0.9),
        70: (54.43887825449708, 1.0340115719619898, 37.35910014714313, 6.608097825809692, 37.35910014714313, 546.3046107148551, 1.0, 0.9),
    },
}


@pytest.mark.parametrize("prof,fps", sorted(PINNED))
def test_default_output_is_unchanged(prof, fps):
    s = Simulation("mirror_check", CAM, seed=3, fps=fps, emulator_overrides=C.resolve_profile(prof, PROFILES))
    s.frames = s.frames[:80]
    s.run()
    for k, want in PINNED[(prof, fps)].items():
        p = s.records[k]["pred"]
        g, fd, hp = p["gaze"], p["face_dir"], p["head_pose_camera"]
        face = next(d for d in p["detections"] if d["class"] == "face")
        got = (g["yaw_camera"], g["pitch_camera"], hp["yaw"], hp["pitch"], fd["yaw_camera"], face["bbox"][0],
               p["eye_openness"]["left"], g["conf"])
        assert got == pytest.approx(want, rel=1e-9, abs=1e-9), (prof, fps, k)
        assert not any("outlier" in (x or {}) for x in (g, fd, hp))


def test_structured_error_leaves_the_other_streams_alone():
    """pg3 changes head pose / gaze / face direction only: every other output and every flag stay as they were."""
    def rest(records):
        return [json.dumps({k: v for k, v in r["pred"].items() if k not in ("head_pose_camera", "gaze", "face_dir")},
                           sort_keys=True, default=float) for r in records]
    a = Simulation("mirror_check", CAM, seed=7); a.frames = a.frames[:200]; a.run()
    b = Simulation("mirror_check", CAM, seed=7, emulator_overrides=PROFILES["pg3"]); b.frames = b.frames[:200]; b.run()
    assert rest(a.records) == rest(b.records)
    assert [r["pred"]["gaze"] for r in a.records] != [r["pred"]["gaze"] for r in b.records]


# ---- per-run bias ---------------------------------------------------------------------------------------------
def test_per_run_bias_is_constant_within_a_run_and_has_the_configured_std(rig, base):
    std = {"head_pose": [1.0, 2.0, 0.5], "gaze": [3.0, 1.5], "face_dir": [2.0, 1.0]}
    over = {k: {"bias_std_deg": v} for k, v in std.items()}
    first = {k: [] for k in std}
    for seed in range(1500):
        e = _emu(rig, over, seed=seed)
        runs = [_errors(e._infer(base), base) for _ in range(4)]
        for k in std:
            assert np.ptp([r[k] for r in runs], axis=0) == pytest.approx(0.0, abs=1e-9)
            first[k].append(runs[0][k])
    for k, want in std.items():
        x = np.array(first[k])
        assert x.std(axis=0) == pytest.approx(want, rel=0.06), k
        assert np.all(np.abs(x.mean(axis=0)) < 0.12 * np.array(want) + 0.02), k
    assert np.corrcoef(np.array(first["head_pose"])[:, 0], np.array(first["face_dir"])[:, 0])[0, 1] == \
        pytest.approx(0.0, abs=0.08)                     # independent draws per quantity


def test_bias_std_scalar_means_every_axis(rig, base):
    a = _errors(_emu(rig, {"gaze": {"bias_std_deg": 2.0}}, seed=4)._infer(base), base)["gaze"]
    b = _errors(_emu(rig, {"gaze": {"bias_std_deg": [2.0, 2.0]}}, seed=4)._infer(base), base)["gaze"]
    assert np.allclose(a, b) and np.all(a != 0)
    with pytest.raises(ValueError):
        _emu(rig, {"gaze": {"bias_std_deg": [1.0, 2.0, 3.0]}})


def test_eye_openness_per_run_bias(rig, base):
    half = copy.deepcopy(base)
    half["eye_openness"] = {"left": 0.5, "right": 0.5}
    x = []
    for seed in range(1500):
        e = _emu(rig, {"eye_openness": {"bias_std": 0.08}}, seed=seed)
        o = [e._infer(half)["eye_openness"] for _ in range(3)]
        assert o[0]["left"] == o[0]["right"] == o[1]["left"] == o[2]["right"]   # one level, both eyes, whole run
        x.append(o[0]["left"] - 0.5)
    assert np.std(x) == pytest.approx(0.08, rel=0.06)


def test_reported_std_includes_bias_and_fast_jitter(rig, base):
    q = _emu(rig, {})._infer(base)["gaze"]["std_deg"]
    assert q == 0.0
    out = _emu(rig, {"gaze": {"bias_std_deg": [3.0, 1.0], "fast_std_deg": 2.0}})._infer(base)["gaze"]
    assert out["std_deg"] == pytest.approx(np.sqrt(4.0 + (9.0 + 1.0) / 2))


# ---- fast jitter ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("fps", [20, 40])
def test_fast_jitter_lag1_correlation_follows_fast_rho_per_second(rig, base, fps):
    rho, sd = 0.6, 1.5
    e = _emu(rig, {k: {"fast_std_deg": sd, "fast_rho": rho} for k in ("head_pose", "gaze", "face_dir")},
             seed=11, fps=fps)
    x = np.array([[_errors(o, base)[k][0] for k in ("head_pose", "gaze", "face_dir")]
                  for o in (e._infer(base) for _ in range(12000))])
    want = rho ** (20 / fps)                              # fast_rho is given at reference_fps 20
    for j in range(3):
        assert np.corrcoef(x[:-1, j], x[1:, j])[0, 1] == pytest.approx(want, abs=0.03)
        assert x[:, j].std() == pytest.approx(sd, rel=0.06)
    assert abs(np.corrcoef(x[:, 0], x[:, 1])[0, 1]) < 0.06           # one stream per quantity


# ---- outliers -------------------------------------------------------------------------------------------------
def _run_lengths(flags):
    runs, cur = [], 0
    for on in list(flags) + [False]:
        if on:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    return runs


@pytest.mark.parametrize("r,mean,cap", [(0.02, 2, 3), (0.05, 2, 3), (0.2, 3, 5), (0.2, 3, None), (0.6, 3, 2)])
def test_capped_markov_step_keeps_share_and_caps_runs(r, mean, cap):
    u = np.random.default_rng(1).random(400000)
    n, flags = None, []
    for x in u:
        n = capped_markov_step(n, r, mean, cap, x)
        flags.append(n > 0)
    runs = _run_lengths(flags)
    b = 1 / mean
    L = 1 / b if cap is None else (1 - (1 - b) ** cap) / b
    a = r / ((1 - r) * L)
    want = r if a <= 1 else L / (L + 1)                  # entry rate capped at 1: share below r (documented)
    assert np.mean(flags) == pytest.approx(want, rel=0.06)
    assert np.mean(runs) == pytest.approx(L, rel=0.04)
    if cap is not None:
        assert max(runs) == cap


@pytest.mark.parametrize("fps,cap", [(20, None), (40, None), (20, 5), (40, 5)])
def test_outlier_share_run_length_and_size(rig, base, fps, cap):
    c = {"outlier_p": 0.2, "outlier_mean_frames": 3, "outlier_max_frames": cap, "outlier_min_deg": 10.0,
         "outlier_max_deg": 20.0}
    e = _emu(rig, {"gaze": c, "head_pose": c}, seed=5, fps=fps)
    flags, mags, offs, cur = [], [], [], 0
    for _ in range(20000):
        o = e._infer(base)
        err = _errors(o, base)
        on = bool(o["gaze"].get("outlier"))
        assert on == (np.linalg.norm(err["gaze"]) > 1e-9)
        if on:
            mags.append(float(np.linalg.norm(err["gaze"])))
            assert not cur or np.allclose(err["gaze"], offs[-1])      # one offset per run
            offs.append(err["gaze"])
        cur = cur + 1 if on else 0
        flags.append(on)
    runs = _run_lengths(flags)
    k = fps / 20                                         # outlier_mean_frames / _max_frames are given at 20 fps
    mean = 3 * k
    L = mean if cap is None else mean * (1 - (1 - 1 / mean) ** round(cap * k))
    assert np.mean(flags) == pytest.approx(0.2, abs=0.02)
    assert np.mean(runs) == pytest.approx(L, rel=0.1)
    if cap is not None:
        assert max(runs) <= round(cap * k)
    assert 10.0 - 1e-9 <= min(mags) and max(mags) <= 20.0 + 1e-9
    assert np.mean(mags) == pytest.approx(15.0, abs=0.6)
    assert "outlier" not in _emu(rig, {}, seed=5)._infer(base)["face_dir"]


def test_profile_outlier_runs_last_1_to_3_frames(rig, base):
    e = _emu(rig, {"gaze": {k: v for k, v in PROFILES["pg3"]["gaze"].items() if k.startswith("outlier")}}, seed=8)
    flags = [bool(e._infer(base)["gaze"].get("outlier")) for _ in range(40000)]
    runs = _run_lengths(flags)
    assert max(runs) == 3 and np.mean(flags) == pytest.approx(PROFILES["pg3"]["gaze"]["outlier_p"], rel=0.12)


# ---- profiles -------------------------------------------------------------------------------------------------
def test_pg_profiles_load_stack_and_pose0_stays_exact(rig, base):
    for name in ("pg1", "pg2", "pg3"):
        assert {"head_pose", "gaze", "face_dir"} <= set(PROFILES[name])
    bad, worst = C.resolve_profile("bad", PROFILES), C.resolve_profile("worst", PROFILES)
    assert bad["gaze"]["outlier_p"] == PROFILES["pg2"]["gaze"]["outlier_p"] and bad["box_jump"]["enabled"]
    assert worst["gaze"]["fast_std_deg"] == PROFILES["pg3"]["gaze"]["fast_std_deg"]
    assert worst["pipeline"]["latency_frames"] == 2
    for stack in ("pg3+exact", "worst+pose0", "exact"):
        ov = deep_merge(C.resolve_profile(stack, PROFILES), {"perfect_detection": True,
                                                             "pipeline": {"drop_frame_prob": 0.0, "latency_frames": 0}})
        e = Emulator(deep_merge(CFGS["emulator"], ov), *rig, np.random.default_rng(2))
        for _ in range(200):
            o = e._infer(base)
            assert o["gaze"]["angular_error_deg"] < 1e-9 and o["face_dir"]["angular_error_deg"] < 1e-9, stack
            assert np.abs(_errors(o, base)["head_pose"]).max() < 1e-9, stack
    e = Emulator(deep_merge(CFGS["emulator"], deep_merge(QUIET, PROFILES["pg2"])), *rig, np.random.default_rng(2))
    assert np.abs(_errors(e._infer(base), base)["gaze"]).max() > 0


def test_exact_stays_exact_with_an_eye_openness_bias_underneath(rig, base):
    """A per-run openness level set in an override below `exact` is switched off by it (ideal sets bias_std 0)."""
    half = copy.deepcopy(base)
    half["eye_openness"] = {"left": 0.5, "right": 0.5}
    over = deep_merge({"eye_openness": {"bias_std": 0.1}}, C.resolve_profile("pg3+exact", PROFILES))
    for seed in range(20):
        e = Emulator(deep_merge(CFGS["emulator"], over), *rig, np.random.default_rng(seed))
        assert e._infer(half)["eye_openness"] == {"left": 0.5, "right": 0.5}
    no_exact = deep_merge(over, {"eye_openness": {"bias_std": 0.1}})              # the bias itself does apply
    e = Emulator(deep_merge(CFGS["emulator"], no_exact), *rig, np.random.default_rng(0))
    assert e._infer(half)["eye_openness"]["left"] != 0.5


def test_outliers_show_in_the_behaviour_card_and_timeline():
    from dms_model_simulator import viz
    from dms_model_simulator.behavior_card import injected_errors, markdown
    c = {"outlier_p": 0.05, "outlier_min_deg": 10.0, "outlier_max_deg": 20.0}
    s = Simulation("mirror_check", CAM, seed=2, fps=20,
                   emulator_overrides={"gaze": c, "pipeline": {"drop_frame_prob": 0.0}})
    s.frames = s.frames[:400]
    s.run()
    e = injected_errors(s.records)
    flags = [bool((r["pred"]["gaze"] or {}).get("outlier")) for r in s.records]
    assert set(e["outlier"]) == {"gaze"} and sum(b - a + 1 for a, b in e["outlier"]["gaze"]) == sum(flags) > 0
    assert all(b - a + 1 <= 3 for a, b in e["outlier"]["gaze"])           # outlier_max_frames 3 at 20 fps
    assert viz._error_rows(s.records)["pose/gaze outlier"] == flags
    c0 = {"fps": 20.0, "frames": 400, "duration_s": 20, "row_id": None, "scenario": "x", "camera": CAM, "seed": 2,
          "emulator_seed": None, "timeline": [], "errors": e}
    assert "離群" in markdown(c0) and "`gaze`" in markdown(c0)
    plain = Simulation("mirror_check", CAM, seed=2); plain.frames = plain.frames[:100]; plain.run()
    assert injected_errors(plain.records)["outlier"] == {} and "pose/gaze outlier" not in viz._error_rows(plain.records)


def test_outlier_cap_never_longer_in_seconds_than_at_20_fps():
    import math
    from dms_model_simulator import emulator as E
    for fps in (20, 25, 30, 40):
        cap = E.at_fps({"reference_fps": 20, "gaze": {"outlier_max_frames": 3}}, fps)["gaze"]["outlier_max_frames"]
        n, longest, rng = None, 0, np.random.default_rng(fps)
        for k in range(20000):
            n = E.capped_markov_step(n, 0.3, 2 * fps / 20, cap, rng.random())
            longest = max(longest, n)
        assert longest / fps <= 3 / 20 + 1e-9, (fps, longest)
        assert longest == math.floor(3 * fps / 20 + 1e-9)

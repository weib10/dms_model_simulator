"""Emulator error model: false-positive counts, AR(1) noise, openness availability, latency bookkeeping."""
import copy
import numpy as np
import pytest
from dms_model_simulator.camera import Camera
from dms_model_simulator.config import deep_merge, load_configs
from dms_model_simulator.emulator import AR1, Emulator
from dms_model_simulator.simulate import Simulation
from dms_model_simulator.world import Cabin

CAM = "steering_column"


@pytest.fixture(scope="module")
def base():
    s = Simulation({"name": "t", "segments": [{"target": "road_ahead", "duration": 0.1}]}, CAM, seed=0)
    s.run()
    return s.records[0]["truth"]


def _emu(over=None, seed=0):
    cfgs = load_configs()
    cab = Cabin(cfgs["cabin"])
    cam = Camera.from_config(CAM, cfgs["cameras"], cab.nominal_head_center())
    return Emulator(deep_merge(cfgs["emulator"], over or {}), cab, cam, np.random.default_rng(seed))


@pytest.mark.parametrize("rate", [0.0, 0.3, 1.0, 2.5])
def test_false_positive_count_is_poisson_with_mean_rate(rate):
    e = _emu({"detection": {"false_positives": {"rate_per_frame": rate}}}, seed=1)
    n = np.array([len(e._false_positives()) for _ in range(20000)])      # rate >= 1 terminates too
    assert n.mean() == pytest.approx(rate, abs=0.05 + 0.03 * rate)
    assert n.var() == pytest.approx(rate, abs=0.06 + 0.06 * rate)       # Poisson: var == mean


def test_ar1_stationary_from_first_sample_and_gap_decorrelates():
    rho = 0.9
    first, a, b = [], [], []
    for s in range(3000):
        n = AR1(np.random.default_rng(s), rho, 1)
        x0 = float(n.step(0)[0])
        first.append(x0)
        assert float(n.step(0)[0]) == x0                 # same frame: same sample
        a.append(float(n.step(1)[0]))
        b.append(float(n.step(46)[0]))                   # 45-frame gap
    assert np.std(first) == pytest.approx(1.0, abs=0.05)
    assert np.corrcoef(first, a)[0, 1] == pytest.approx(rho, abs=0.03)
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.06           # rho**45 ~ 0.009
    assert np.std(b) == pytest.approx(1.0, abs=0.05)


def test_gaze_noise_decorrelates_across_a_closure(base):
    closed = copy.deepcopy(base)
    closed["eye_openness"] = {"left": 0.05, "right": 0.05}
    exact = {"pipeline": {"drop_frame_prob": 0.0},
             "detection": {k: {"p_max": 1.0} for k in ("face", "eye", "mouth")}}
    x, y = [], []
    for s in range(400):
        e = _emu(exact, seed=100 + s)
        o1 = e._infer(base)
        for _ in range(45):
            assert not e._infer(closed)["gaze"]["valid"]
        o2 = e._infer(base)
        if o1["gaze"]["valid"] and o2["gaze"]["valid"]:
            x.append(o1["gaze"]["yaw_camera"]); y.append(o2["gaze"]["yaw_camera"])
    assert len(x) > 300 and abs(np.corrcoef(x, y)[0, 1]) < 0.15        # was ~rho = 0.9 before gap tracking


def test_openness_null_for_undetected_eye_or_mouth(base):
    e = _emu({"detection": {"eye": {"p_max": 0.0}, "mouth": {"p_max": 0.0}, "face": {"p_max": 1.0}}})
    o = e._infer(base)
    assert o["head_pose_camera"] is not None                 # face found ...
    assert o["eye_openness"] is None and o["mouth_openness"] is None    # ... but no eye / mouth detected
    one = copy.deepcopy(base)
    one["detections"] = [d for d in one["detections"] if d["class"] != "right_eye"]   # right eye out of view
    e = _emu({"detection": {k: {"p_max": 1.0, "min_width_px": 0.0} for k in ("face", "eye", "mouth")}})
    o = e._infer(one)
    assert o["eye_openness"]["right"] is None and 0.0 <= o["eye_openness"]["left"] <= 1.0
    assert o["mouth_openness"] is not None


EXACT_DET = {"pipeline": {"drop_frame_prob": 0.0},
             "detection": {k: {"p_max": 1.0, "min_width_px": 0.0} for k in ("face", "eye", "mouth")}}


def test_face_dir_without_eye_gaze_and_none_without_face(base):
    closed = copy.deepcopy(base)
    closed["eye_openness"] = {"left": 0.05, "right": 0.05}
    o = _emu(EXACT_DET)._infer(closed)
    assert not o["gaze"]["valid"] and o["gaze"]["reason"] == "eyes_closed"
    fd = o["face_dir"]
    assert fd["valid"] and 0.0 <= fd["conf"] <= 1.0 and fd["std_deg"] > 0
    assert abs(np.linalg.norm(fd["dir_vehicle"]) - 1) < 1e-9
    noface = copy.deepcopy(base)
    noface["detections"] = [d for d in noface["detections"] if d["class"] != "face"]
    o = _emu(EXACT_DET)._infer(noface)
    assert o["face_dir"] is None and o["gaze"]["reason"] == "no_face" and o["head_pose_camera"] is None
    e = _emu({"pipeline": {"drop_frame_prob": 1.0}})
    assert e.emulate(base)["face_dir"] is None


def test_face_dir_zero_noise_is_exact(base):
    zero = deep_merge(EXACT_DET, {"face_dir": {"base_std_deg": 0.0, "per_deg_rel_yaw": 0.0,
                                               "per_px_inv_face_width": 0.0, "bias_deg": [0.0, 0.0]}})
    fd = _emu(zero)._infer(base)["face_dir"]
    tf = base["face_dir"]
    assert fd["angular_error_deg"] < 1e-9 and fd["zone"] == tf["zone"]
    assert fd["yaw_camera"] == pytest.approx(tf["yaw_camera"], abs=1e-9)
    assert fd["yaw_vehicle"] == pytest.approx(tf["yaw_vehicle"], abs=1e-9)
    fd = _emu(EXACT_DET)._infer(base)["face_dir"]               # nominal noise: a few degrees off
    assert 0 < fd["angular_error_deg"] < 6 * fd["std_deg"]


def test_no_unused_eye_noise_stream():
    e = _emu()
    assert "eye" not in e.box_noise and {"left_eye", "right_eye"} <= set(e.box_noise)


def test_latency_tags_source_frame():
    L = 4
    s = Simulation("mirror_check", CAM, seed=1, emulator_overrides={"pipeline": {"latency_frames": L, "drop_frame_prob": 0.0}})
    s.frames = s.frames[:30]
    s.run()
    assert all(r["pred"]["source_frame"] is None and r["pred"]["latency_fill"] for r in s.records[:L])
    assert all(r["pred"]["source_frame"] == r["frame"] - L for r in s.records[L:])
    s0 = Simulation("mirror_check", CAM, seed=1, emulator_overrides={"pipeline": {"drop_frame_prob": 0.0}})
    s0.frames = s0.frames[:5]
    s0.run()
    assert all("source_frame" not in r["pred"] for r in s0.records)

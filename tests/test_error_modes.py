"""Optional error modes: burst misses / drops, eye and mouth class flips, contextual false positives."""
import copy
import json
import numpy as np
import pytest
from dms_model_simulator.config import deep_merge, load_configs, load_yaml
from dms_model_simulator.emulator import markov_step
from dms_model_simulator.metrics import segments
from dms_model_simulator.schema import mapped_detections
from dms_model_simulator.simulate import Simulation, _json_default
from tests.conftest import MAPPED_LAYOUT
from tests.test_emulator import _emu, base  # noqa: F401  (fixture)

CAM = "steering_column"


def _chain(r, L, n=200000, seed=0):
    rng = np.random.default_rng(seed)
    s, out = None, np.empty(n, bool)
    for i in range(n):
        s = out[i] = markov_step(s, r, L, rng.random())
    return out


@pytest.mark.parametrize("r,L", [(0.1, 5), (0.3, 20), (0.9, 3), (0.05, 1)])
def test_markov_step_rate_and_run_length(r, L):
    x = _chain(r, L)
    assert x.mean() == pytest.approx(r, abs=0.02)
    runs = [j - i for i, j in segments(x)]
    expect = L if r <= L / (L + 1) else 1.0 / (1.0 - r)       # clipped branch: runs of 1 / b = r / (1 - r) + ...
    if r <= L / (L + 1):
        assert np.mean(runs) == pytest.approx(expect, rel=0.1)
    else:
        assert np.mean(runs) > L                                 # longer runs than asked, rate kept


def test_markov_step_edges():
    assert not any(_chain(0.0, 5, n=1000))
    assert all(_chain(1.0, 5, n=1000))


def test_burst_misses_keep_rate_and_lengthen_runs(base):
    exact = {"pipeline": {"drop_frame_prob": 0.0}, "detection": {"face": {"p_max": 0.8, "min_width_px": 0.0}}}
    iid = _emu(exact, seed=3)
    burst = _emu({**exact, "detection": {"face": {"p_max": 0.8, "min_width_px": 0.0, "mean_miss_frames": 12}}}, seed=3)
    miss = {}
    for name, e in (("iid", iid), ("burst", burst)):
        miss[name] = np.array([not any(d["class"] == "face" for d in e._infer(base)["detections"]) for _ in range(20000)])
    p = burst._p_detect(burst.cfg["detection"]["face"], next(d for d in base["detections"] if d["class"] == "face")["width_px"],
                        base["head_pose_camera"]["yaw"], next(d for d in base["detections"] if d["class"] == "face")["occlusion"])
    for x in miss.values():
        assert x.mean() == pytest.approx(1 - p, abs=0.02)
    assert np.mean([j - i for i, j in segments(miss["iid"])]) < 1.5
    assert np.mean([j - i for i, j in segments(miss["burst"])]) == pytest.approx(12, rel=0.15)


def test_burst_drops():
    e = _emu({"pipeline": {"drop_frame_prob": 0.05, "drop_mean_frames": 8}}, seed=2)
    truth = Simulation({"name": "t", "segments": [{"target": "road_ahead", "duration": 0.1}]}, CAM, seed=0)
    truth.run()
    t = truth.records[0]["truth"]
    d = np.array([e.emulate(t)["dropped"] for _ in range(20000)])
    assert d.mean() == pytest.approx(0.05, abs=0.015)
    assert np.mean([j - i for i, j in segments(d)]) == pytest.approx(8, rel=0.2)


def test_defaults_unchanged_by_new_keys():
    """The shipped config (new modes off) gives the same records as a config without the new keys at all."""
    cfgs = load_configs()
    stripped = copy.deepcopy(cfgs)
    em = stripped["emulator"]
    for k in ("eye_state_flip", "mouth_state_flip", "contextual_false_positives"):
        em.pop(k)
    em["pipeline"].pop("drop_mean_frames")
    out = []
    for c in (cfgs, stripped):
        s = Simulation("scenarios/hand_to_face.yaml", CAM, seed=4, cfgs=c); s.run()
        out.append(json.dumps(s.records, default=_json_default, sort_keys=True))
    assert out[0] == out[1]


def test_flip_values_land_on_the_other_side_of_the_class_boundary():
    """A flipped eye / mouth is reported with closed_value / open_value: on the wrong side of the emulator's own
    threshold and of the test layout's class_map threshold (the flip tests below sync the two)."""
    em = load_configs()["emulator"]
    spec = load_yaml(MAPPED_LAYOUT)
    for k, thr in (("eye_state_flip", spec["lids_threshold"]), ("mouth_state_flip", spec["jaw_threshold"])):
        assert em[k]["closed_value"] < em[k]["threshold"] <= em[k]["open_value"]
        assert em[k]["closed_value"] < thr <= em[k]["open_value"]


def _classes(ov, scenario, seed=1):
    """Run with emulator overrides ov (flip thresholds set to the test layout's class boundaries) and map every
    frame's detections with the test layout."""
    spec = load_yaml(MAPPED_LAYOUT)
    sync = {"eye_state_flip": {"threshold": spec["lids_threshold"]},
            "mouth_state_flip": {"threshold": spec["jaw_threshold"]}}
    s = Simulation(scenario, CAM, seed=seed, emulator_overrides=deep_merge(sync, ov or {})); s.run()
    return s, [[d["class"] for d in mapped_detections(r, spec)[0]] for r in s.records]


def test_eye_flip_reports_closed_eyes_when_looking_into_the_lap():
    sc = {"name": "lap", "segments": [{"target": "road_ahead", "duration": 1.0}, {"target": "phone_lap", "duration": 5.0}]}
    _, off = _classes(None, sc)
    s, on = _classes({"eye_state_flip": {"enabled": True}}, sc)
    lap = range(60, len(s.records))
    assert sum(c.count("lids_shut") for c in (off[i] for i in lap)) == 0
    n_closed = sum(c.count("lids_shut") for c in (on[i] for i in lap))
    n_eye = sum(c.count("lids_shut") + c.count("lids_open") for c in (on[i] for i in lap))
    assert 0.15 < n_closed / n_eye < 0.6                      # q ~ 0.3 - 0.4 while looking down
    # flips come in runs: both eyes of a frame flip together
    assert all(c.count("lids_shut") in (0, c.count("lids_shut") + c.count("lids_open")) for c in on)


def test_flip_reset_on_truth_class_change(base):
    e = _emu({"eye_state_flip": {"enabled": True, "mean_frames": 1000, "open_to_closed": {"p_base": 1.0},
                                 "closed_to_open": {"p_base": 0.0}}})
    assert e._flip("eye", 1.0, base, 20.0) == e.cfg["eye_state_flip"]["closed_value"]     # open reported closed
    assert e._flip("eye", 0.0, base, 20.0) is None           # blink: truth closed, reported closed (reset)
    assert e._flip("eye", 1.0, base, 20.0) is not None       # open again: re-entered (p_base 1)


def test_contextual_false_positives_follow_the_hand():
    ov = {"contextual_false_positives": {"enabled": True}}
    s, on = _classes(ov, "scenarios/hand_to_face.yaml")
    _, off = _classes(None, "scenarios/hand_to_face.yaml")
    ear = [i for i, r in enumerate(s.records) if r["truth"]["hand"] == "at_ear"]
    mouth = [i for i, r in enumerate(s.records) if r["truth"]["hand"] == "at_mouth"]
    other = [i for i, r in enumerate(s.records) if r["truth"]["hand"] not in ("at_ear", "at_mouth")]
    assert sum("handset" in on[i] for i in ear) > 0.15 * len(ear)
    assert sum("smoke_stick" in on[i] for i in mouth) > 0.1 * len(mouth)
    assert not any("smoke_stick" in on[i] for i in other)
    assert sum("handset" in off[i] for i in ear) < 0.05 * len(ear)
    assert not any(r["truth"]["events"].get("phone_use") or r["truth"]["events"].get("smoking") for r in s.records)
    fp = [d for r in s.records for d in r["pred"]["detections"] if d.get("context")]
    assert fp and all(d["false_positive"] for d in fp)


def test_mouth_flip_reports_open_mouth_with_hand_at_mouth():
    sc = {"name": "m", "segments": [{"target": "road_ahead", "duration": 6.0, "hand": "at_mouth"}]}
    ov = {"mouth_state_flip": {"enabled": True, "closed_to_open": {"p_base": 0.0, "p_hand_state": {"at_mouth": 0.5}}},
          "detection": {"mouth": {"occlusion_power": 0.0}}}
    _, off = _classes({"detection": {"mouth": {"occlusion_power": 0.0}}}, sc)
    _, on = _classes(ov, sc)
    assert sum(c.count("jaw_open") for c in off) == 0
    n_open = sum(c.count("jaw_open") for c in on)
    n = sum(c.count("jaw_open") + c.count("jaw_shut") for c in on)
    assert n > 50 and 0.25 < n_open / n < 0.75


def test_burst_drops_through_simulation():
    s = Simulation("scenarios/mirror_check.yaml", CAM, seed=0,
                   emulator_overrides={"pipeline": {"drop_frame_prob": 0.1, "drop_mean_frames": 6}}); s.run()
    d = [r["pred"]["dropped"] for r in s.records]
    runs = [j - i for i, j in segments(d)]
    assert sum(d) > 0 and max(runs) > 2


def test_burst_miss_state_resets_when_the_class_leaves_the_truth(base):
    e = _emu({"detection": {"phone": {"p_max": 0.5, "min_width_px": 0.0, "mean_miss_frames": 50}}})
    e.miss_state["phone"] = True
    e._infer(base)                                           # no phone in this truth frame
    assert "phone" not in e.miss_state


def test_eye_flip_resets_when_the_eyes_leave_the_truth(base):
    # a flip latched before the eyes left the image (e.g. a long head turn) does not come back with them
    e = _emu({"eye_state_flip": {"enabled": True, "mean_frames": 1000,
                                 "open_to_closed": {"p_base": 1.0, "p_gaze_down": 0.0, "p_small": 0.0}}}, seed=0)
    flipped = lambda out: out["eye_openness"] is not None and max(v for v in out["eye_openness"].values() if v is not None) \
        == e.cfg["eye_state_flip"]["closed_value"]
    assert flipped(e._infer(base))
    no_eyes = copy.deepcopy(base)
    no_eyes["detections"] = [d for d in no_eyes["detections"] if not d["class"].endswith("_eye")]
    for _ in range(20):
        e._infer(no_eyes)
    e.cfg["eye_state_flip"]["open_to_closed"]["p_base"] = 1e-6
    assert not flipped(e._infer(base))


def test_contextual_false_positive_can_require_a_held_object():
    rule = {"name": "bottle_as_phone", "when_hand": ["at_mouth"], "when_held": ["bottle"], "class": "phone",
            "p": 0.6, "mean_frames": 5, "box_scale": [1.0, 1.0], "jitter_frac": 0.0, "conf_range": [0.6, 0.8]}
    sc = {"name": "d", "segments": [{"target": "road_ahead", "duration": 4.0, "hand": "at_mouth"},
                                    {"target": "road_ahead", "duration": 4.0, "hand": "at_mouth", "held": "bottle"}]}
    s, cls = _classes({"contextual_false_positives": {"enabled": True, "rules": [rule]}}, sc)
    bottle = [i for i, r in enumerate(s.records) if r["truth"]["held"] == "bottle"]
    empty = [i for i, r in enumerate(s.records) if r["truth"]["held"] == "none"]
    assert sum("handset" in cls[i] for i in bottle) > 0.3 * len(bottle)
    assert not any("handset" in cls[i] for i in empty)


def test_box_jump_and_conditions_off_leave_output_unchanged():
    """Enabled with no classes, box_jump draws nothing visible; zero box_conditions gains scale by exactly 1."""
    out = []
    for over in ({}, {"box_jump": {"enabled": True, "classes": []}, "box_conditions": {"occlusion": 0.0}}):
        s = Simulation("scenarios/hand_to_face.yaml", CAM, seed=4, emulator_overrides=over); s.run()
        out.append(json.dumps([r["pred"] for r in s.records], default=_json_default, sort_keys=True))
    assert out[0] == out[1]


def test_box_jump_rate_and_only_jumped_class_moves(base):
    over = {"pipeline": {"drop_frame_prob": 0.0}, "detection": {"face": {"p_max": 1.0, "min_width_px": 0.0}},
            "box_jump": {"enabled": True, "classes": ["face"], "p": 0.2, "mean_frames": 2.0, "shift_frac": 0.5}}
    on, off = _emu(over, seed=7), _emu({**over, "box_jump": {"enabled": False}}, seed=7)
    jumped, eye_same = [], []
    for _ in range(4000):
        a, b = on._infer(base), off._infer(base)
        fa = next(d for d in a["detections"] if d["class"] == "face")
        jumped.append(bool(fa.get("jump")))
        ea = [d["bbox"] for d in a["detections"] if d["class"].endswith("_eye")]
        eb = [d["bbox"] for d in b["detections"] if d["class"].endswith("_eye")]
        eye_same.append(ea == eb)
    assert np.mean(jumped) == pytest.approx(0.2, abs=0.04)
    assert np.mean([j - i for i, j in segments(np.array(jumped))]) == pytest.approx(2.0, rel=0.2)
    assert all(eye_same)                         # the jump stream is separate: other classes are untouched


def test_features_follow_a_jumped_face(base):
    jump = {"enabled": True, "classes": ["face"], "p": 1.0, "mean_frames": 5.0, "shift_frac": 0.3,
            "scale_range": [1.0, 1.0]}
    over = {"pipeline": {"drop_frame_prob": 0.0}, "detection": {"face": {"p_max": 1.0, "min_width_px": 0.0},
                                                                 "eye": {"p_max": 1.0, "min_width_px": 0.0}}}
    follow = _emu({**over, "box_jump": {**jump, "features_follow_face": True}}, seed=1)
    alone = _emu({**over, "box_jump": jump}, seed=1)
    a, b = follow._infer(base), alone._infer(base)
    shift = follow.face_shift
    assert shift is not None and np.abs(shift).max() > 0
    ea = {d["class"]: np.array(d["bbox"]) for d in a["detections"] if d["class"].endswith("_eye")}
    eb = {d["class"]: np.array(d["bbox"]) for d in b["detections"] if d["class"].endswith("_eye")}
    assert ea and set(ea) == set(eb)
    for c in ea:                                 # inside the image, so not clipped: moved by exactly the face shift
        np.testing.assert_allclose(ea[c] - eb[c], np.tile(shift, 2), atol=1e-6)


def test_box_conditions_hardness():
    from dms_model_simulator.emulator import Emulator
    spec = {"min_width_px": 10}
    det = {"occlusion": 0.5, "width_px": 5.0}
    assert Emulator._hardness({}, spec, det, 30.0, "face") == 1.0
    g = {"occlusion": 1.0, "rel_yaw": 1.0, "small": 1.0}
    assert Emulator._hardness(g, spec, det, 45.0, "face") == pytest.approx(1 + 0.5 + 1.0 + 2.0)
    assert Emulator._hardness(g, spec, det, 45.0, "phone") == pytest.approx(1 + 0.5 + 2.0)     # yaw: face parts only
    assert Emulator._hardness(g, spec, det, 200.0, "eye") == pytest.approx(1 + 0.5 + 2.0 + 2.0)  # capped


# ---- class_flip flag -------------------------------------------------------------------------------
_FLIP_SC = {"name": "f", "segments": [{"target": "road_ahead", "duration": 1.5},
                                      {"target": "phone_lap", "duration": 1.5, "hand": "at_mouth"}]}


def _flip_run(ov):
    s = Simulation(_FLIP_SC, CAM, seed=3, emulator_overrides=ov); s.run()
    return s.records


def test_class_flip_flag_absent_by_default():
    assert not any("class_flip" in r["pred"] for r in _flip_run(None))
    off = {"eye_state_flip": {"enabled": True, "open_to_closed": {"p_base": 0.0, "p_gaze_down": 0.0, "p_small": 0.0},
                              "closed_to_open": {"p_base": 0.0}}}
    assert not any("class_flip" in r["pred"] for r in _flip_run(off))   # enabled, never flipped: still absent


def test_class_flip_flag_marks_exactly_the_flipped_frames():
    fc = load_configs()["emulator"]["eye_state_flip"]
    recs = _flip_run({"eye_state_flip": {"enabled": True, "mean_frames": 4, "open_to_closed": {"p_base": 0.4}}})
    n_flag = 0
    for r in recs:
        eo = r["pred"]["eye_openness"]
        vals = [v for v in (eo or {}).values() if v is not None]
        flipped = bool(vals) and all(v in (fc["closed_value"], fc["open_value"]) for v in vals)
        assert ("class_flip" in r["pred"]) == flipped, r["frame"]
        if flipped:
            assert r["pred"]["class_flip"] == ["eye"]
            n_flag += 1
    assert 0 < n_flag < sum(r["pred"]["eye_openness"] is not None for r in recs)


def test_class_flip_flag_for_eye_and_mouth():
    em = load_configs()["emulator"]
    always = {"eye_state_flip": {"enabled": True, "mean_frames": 1000,
                                 "open_to_closed": {"p_base": 1.0, "p_gaze_down": 0.0, "p_small": 0.0},
                                 "closed_to_open": {"p_base": 1.0}},
              "mouth_state_flip": {"enabled": True, "mean_frames": 1000, "closed_to_open": {"p_base": 1.0},
                                   "open_to_closed": {"p_base": 1.0}},
              "detection": {"mouth": {"occlusion_power": 0.0}}}
    recs = _flip_run(always)
    for r in recs:
        pr, tr = r["pred"], r["truth"]
        want = [k for k, on in (("eye", pr["eye_openness"] is not None), ("mouth", pr["mouth_openness"] is not None))
                if on]
        assert pr.get("class_flip", []) == want, r["frame"]          # evaluated every frame, flipped every frame
        if pr["mouth_openness"] is not None:
            closed = tr["mouth_openness"] < em["mouth_state_flip"]["threshold"]
            assert pr["mouth_openness"] == em["mouth_state_flip"]["open_value" if closed else "closed_value"]
    assert any(r["pred"].get("class_flip") == ["eye", "mouth"] for r in recs)


def test_contextual_false_positive_anchored_on_a_held_object_in_the_mouth():
    # a straw held in the mouth (no hand): the rule's box sits on the straw's box, not on a hand
    rule = {"name": "straw_as_cig", "anchor": "held", "when_held": ["straw"], "class": "cigarette",
            "p": 0.6, "mean_frames": 5, "box_scale": [1.0, 1.0], "jitter_frac": 0.0, "conf_range": [0.6, 0.8]}
    sc = {"name": "d", "segments": [{"target": "road_ahead", "duration": 3.0},
                                    {"target": "road_ahead", "duration": 4.0, "held": "straw"}]}
    s = Simulation(sc, CAM, seed=1, emulator_overrides={"contextual_false_positives": {"enabled": True, "rules": [rule]}})
    s.run()
    straw = [r for r in s.records if r["truth"]["held"] == "straw"]
    empty = [r for r in s.records if r["truth"]["held"] == "none"]
    assert all(r["truth"]["hand"] == "wheel" and r["truth"]["held_bbox"] for r in straw)
    hits = [(r, d) for r in straw for d in r["pred"]["detections"] if d.get("context") == "straw_as_cig"]
    assert len(hits) > 0.3 * len(straw)
    for r, d in hits:                                   # jitter 0, scale 1: the straw's own box
        assert d["bbox"] == pytest.approx(r["truth"]["held_bbox"])
    assert not any(d.get("context") for r in empty for d in r["pred"]["detections"])
    assert all(r["truth"]["held_bbox"] is None for r in empty)

"""cigarette and seatbelt classes: geometry, behaviour labels, pipeline output and dms.bin."""
import numpy as np
import pytest
from dms_model_simulator.behavior import BehaviorGenerator, DriverFrame
from dms_model_simulator.camera import Camera
from dms_model_simulator.config import load_configs
from dms_model_simulator.binlog import DmsBinLayout, read_dms_bin, write_dms_bin
from dms_model_simulator.events import label_events
from dms_model_simulator.schema import FIELDS, LABELS, to_pipeline, to_truth
from dms_model_simulator.simulate import Simulation
from dms_model_simulator.truth import compute_truth
from dms_model_simulator.world import Cabin

F = FIELDS
EXACT = {"pipeline": {"drop_frame_prob": 0},
         "detection": {c: {"p_max": 1, "min_width_px": 0} for c in ("face", "eye", "mouth", "hand", "cigarette", "seatbelt")}}


def _setup(cam="steering_column"):
    cfgs = load_configs(); cabin = Cabin(cfgs["cabin"])
    return cabin, Camera.from_config(cam, cfgs["cameras"], cabin.nominal_head_center())


def _frame(cabin, hand="wheel", cigarette="none", seatbelt="on"):
    f = DriverFrame(0, 0, 0, 0, cabin.targets["road_ahead"], "road_ahead", 1, 1, False, False, hand, "none")
    f.cigarette, f.seatbelt = cigarette, seatbelt
    return f


def _det(tr, cls):
    return next((d for d in tr["detections"] if d["class"] == cls), None)


def _center(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


@pytest.fixture(scope="module")
def smoking():
    s = Simulation("smoking", "steering_column", seed=0, emulator_overrides=EXACT)
    s.run()
    return s


def test_driver_frame_defaults_keep_positional_constructor():
    cabin, _ = _setup()
    f = DriverFrame(0, 0, 0, 0, cabin.targets["road_ahead"], "road_ahead", 1, 1, False, False, "wheel", "none")
    assert (f.cigarette, f.seatbelt, f.mouth_openness, f.yawning) == ("none", "on", 0.0, False)


@pytest.mark.parametrize("hand", ["wheel", "smoke"])
def test_cigarette_at_mouth_is_next_to_mouth_box(hand):
    cabin, cam = _setup()
    assert _det(compute_truth(cabin, cam, _frame(cabin, hand)), "cigarette") is None
    tr = compute_truth(cabin, cam, _frame(cabin, hand, cigarette="mouth"))
    cig, mouth = _det(tr, "cigarette"), _det(tr, "mouth")
    assert cig and mouth and cig["occlusion"] == 0.0 and cig["state"] == "mouth"
    mb, cb = mouth["bbox"], cig["bbox"]
    mouth_w = mb[2] - mb[0]
    assert np.linalg.norm(_center(cb) - _center(mb)) < mouth_w            # centred near the lips
    assert cb[0] < mb[2] and cb[2] > mb[0]                                 # overlaps the mouth columns
    assert 0 < cig["width_px"] < 2 * mouth_w                               # small object
    # mouth box itself is unchanged by the cigarette
    assert mb == _det(compute_truth(cabin, cam, _frame(cabin, hand)), "mouth")["bbox"]


def test_cigarette_in_hand_is_at_the_hand():
    cabin, cam = _setup()
    tr = compute_truth(cabin, cam, _frame(cabin, "smoke", cigarette="hand"))
    cig, hand = _det(tr, "cigarette"), _det(tr, "hand")
    assert cig and hand and hand["state"] == "smoke"
    hb = hand["bbox"]
    assert hb[0] <= _center(cig["bbox"])[0] <= hb[2] and hb[1] <= _center(cig["bbox"])[1] <= hb[3]
    # cigarette "hand" without a hand in the frame: no box
    assert _det(compute_truth(cabin, cam, _frame(cabin, "none", cigarette="hand")), "cigarette") is None


def test_cigarette_in_head_frame_hand_turns_with_the_head():
    """hand: smoke is a head-frame hand, so the cigarette it holds keeps its pose relative to the head; a hand on
    the wheel keeps the vehicle-frame pose."""
    cabin, _ = _setup()
    rest, turned = cabin.head_state(0, 0, 0), cabin.head_state(40, -10, 5)
    def rel(head, hand):
        a, b = cabin.cigarette_segment("hand", hand, head)
        hp = cabin.hand_position(hand, head)
        return head.R.T @ (a - hp), head.R.T @ (b - hp)
    assert np.allclose(rel(rest, "smoke"), rel(turned, "smoke"))
    a0, b0 = cabin.cigarette_segment("hand", "wheel", rest)
    a1, b1 = cabin.cigarette_segment("hand", "wheel", turned)
    assert np.allclose(a0, a1) and np.allclose(b0, b1)


def test_seatbelt_present_when_on_absent_when_off():
    cabin, cam = _setup("steering_column")
    on = compute_truth(cabin, cam, _frame(cabin, seatbelt="on"))
    off = compute_truth(cabin, cam, _frame(cabin, seatbelt="off"))
    sb = _det(on, "seatbelt")
    assert sb is not None and _det(off, "seatbelt") is None
    x1, y1, x2, y2 = sb["bbox"]
    assert 0 <= x1 < x2 <= cam.width and 0 <= y1 < y2 <= cam.height and sb["width_px"] == x2 - x1
    assert 0.0 <= sb["occlusion"] <= 1.0
    # strap starts at the driver's left shoulder: image right of the face, below it, for a frontal camera
    face = _det(on, "face")["bbox"]
    assert _center(sb["bbox"])[0] > _center(face)[0] and y1 > face[1]
    # the would-be strap box is still known when the belt is off (used for emulator false positives)
    assert off["seatbelt_strap_bbox"] == sb["bbox"] and (on["seatbelt"], off["seatbelt"]) == ("on", "off")


def test_seatbelt_out_of_view_has_no_box():
    cfgs = load_configs()
    cfgs["cabin"]["seatbelt_strap"].update({"top": [-0.3, 3.0, -0.15], "bottom": [-0.3, 2.6, -0.55], "pillar_path": []})
    cabin = Cabin(cfgs["cabin"])
    cam = Camera.from_config("steering_column", cfgs["cameras"], cabin.nominal_head_center())
    tr = compute_truth(cabin, cam, _frame(cabin))
    assert _det(tr, "seatbelt") is None and tr["seatbelt_strap_bbox"] is None


def test_smoking_and_seatbelt_off_event_labels():
    cfgs = load_configs(); cabin = Cabin(cfgs["cabin"])
    sc = {"name": "t", "segments": [{"target": "road_ahead", "duration": 1.0},
                                    {"target": "road_ahead", "duration": 1.0, "hand": "smoke", "cigarette": "mouth"},
                                    {"target": "road_ahead", "duration": 1.0, "cigarette": "hand"},
                                    {"target": "road_ahead", "duration": 1.0, "seatbelt": False},   # YAML `off`
                                    {"target": "road_ahead", "duration": 1.0, "seatbelt": "off"}]}
    beh = BehaviorGenerator(cabin, cfgs["behavior"], sc, np.random.default_rng(0))
    frames = beh.generate()
    ev = label_events(frames, cabin, beh.cfg)
    assert {f.seatbelt for f in frames} == {"on", "off"}
    for f, e in zip(frames, ev):
        seg = int(f.t)
        assert e["smoking"] == (seg in (1, 2)) == (f.cigarette != "none")
        assert e["seatbelt_off"] == (seg >= 3) == (f.seatbelt == "off")
        assert not e["phone_use"]


def test_smoking_scenario_records_and_pipeline_labels(smoking):
    ev = [r["truth"]["events"] for r in smoking.records]
    assert any(e["smoking"] for e in ev) and any(e["seatbelt_off"] for e in ev)
    pipe = [to_pipeline(r, smoking.camera) for r in smoking.records]
    labels = {o[F["label"]] for p in pipe for o in p[F["od"]]}
    assert {LABELS["cigarette"], LABELS["seatbelt"]} <= labels
    for r, p in zip(smoking.records, pipe):
        has_belt = any(o[F["label"]] == LABELS["seatbelt"] for o in p[F["od"]])
        if r["truth"]["seatbelt"] == "on":
            assert has_belt                                 # p_detect ~1 with EXACT overrides
        elif has_belt:
            assert any(d.get("false_positive") for d in r["pred"]["detections"] if d["class"] == "seatbelt")
        if r["truth"]["cigarette"] == "mouth":
            assert any(o[F["label"]] == LABELS["cigarette"] for o in p[F["od"]])
    t = to_truth(smoking.records[100], smoking.camera)
    assert t[F["cigarette"]] == "mouth" and t[F["seatbelt"]] == "on"


@pytest.mark.parametrize("rate", [0.0, 1.0])
def test_seatbelt_off_false_positive_rate(rate):
    sc = {"name": "t", "segments": [{"target": "road_ahead", "duration": 1.0, "seatbelt": "off"}]}
    ov = {**EXACT, "detection": {**EXACT["detection"], "seatbelt": {"seatbelt_off_false_positive_rate": rate}}}
    s = Simulation(sc, "steering_column", seed=1, emulator_overrides=ov); s.run()
    belts = [[d for d in r["pred"]["detections"] if d["class"] == "seatbelt"] for r in s.records]
    if rate == 0.0:
        assert not any(belts)
    else:
        assert all(len(b) == 1 and b[0]["false_positive"] for b in belts)
        assert all(abs(_center(b[0]["bbox"]) - _center(r["truth"]["seatbelt_strap_bbox"])).max() < 50
                   for b, r in zip(belts, s.records))


def test_dms_bin_roundtrip_has_cigarette_and_seatbelt_ids(smoking, tmp_path):
    layout = DmsBinLayout.from_yaml()                   # generic default layout
    st = write_dms_bin(smoking.records, smoking.camera, layout, tmp_path / "dms.bin")
    assert st["skipped"] == {}
    frames = list(read_dms_bin(tmp_path / "dms.bin", layout))
    ids = {o["class_id"] for f in frames for o in f["objects"]}
    assert {layout.class_ids["cigarette"], layout.class_ids["seatbelt"]} <= ids
    for cls in ("cigarette", "seatbelt"):
        assert layout.class_names[layout.class_ids[cls]] == cls
    names = {o["class"] for f in frames for o in f["objects"]}
    assert {"cigarette", "seatbelt"} <= names
    # only model outputs are written: the header holds no behaviour labels
    assert set(frames[0]["header"]) == {"timestamp", "frame_index", "width", "height", "num_objects"}
    for fr, r in zip(frames, smoking.records):
        if r["truth"]["cigarette"] == "mouth":                # p_detect ~1 with EXACT overrides
            assert "cigarette" in {o["class"] for o in fr["objects"]}

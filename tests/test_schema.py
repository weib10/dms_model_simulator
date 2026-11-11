import dataclasses
import json
import time
import numpy as np
from dms_model_simulator.behavior import DriverFrame
from dms_model_simulator.camera import Camera
from dms_model_simulator.config import load_configs
from dms_model_simulator.replay import iter_pairs, iter_pipeline
from dms_model_simulator.schema import FIELDS, to_pipeline, to_truth
from dms_model_simulator.simulate import Simulation
from dms_model_simulator.truth import compute_truth
from dms_model_simulator.world import Cabin

F = FIELDS


def _setup(cam="steering_column"):
    cfgs = load_configs(); cabin = Cabin(cfgs["cabin"])
    return cabin, Camera.from_config(cam, cfgs["cameras"], cabin.nominal_head_center())


@dataclasses.dataclass
class _MouthFrame(DriverFrame):      # a DriverFrame with a fixed mouth_openness default
    mouth_openness: float = 0.0


def _frame(cabin, mouth):
    return _MouthFrame(0, 0, 0, 0, cabin.targets["road_ahead"], "road_ahead", 1, 1, False, False, "wheel", "none", mouth)


def _box(tr, cls):
    return next(d["bbox"] for d in tr["detections"] if d["class"] == cls)


def _sim(**ov):
    s = Simulation("mirror_check", "steering_column", seed=2, emulator_overrides=ov or None); s.run()
    return s


def test_pipeline_record_keys_and_types():
    s = _sim(pipeline={"drop_frame_prob": 0})
    for r in s.records[:50]:
        p = json.loads(json.dumps(to_pipeline(r, s.camera)))
        assert set(p) == {F[k] for k in ("frame_id", "timestamp_ms", "camera_id", "od", "head_pose", "gaze",
                                         "eye_openness", "mouth_openness")}
        assert isinstance(p[F["frame_id"]], int) and isinstance(p[F["timestamp_ms"]], int)
        assert p[F["camera_id"]] == "steering_column"
        for o in p[F["od"]]:
            assert o[F["label"]] in ("left_eye", "right_eye", "mouth", "hand", "phone", "cigarette", "seatbelt")
            assert len(o[F["bbox"]]) == 4 and all(isinstance(v, float) for v in o[F["bbox"]])
            assert 0 <= o[F["score"]] <= 1
        if p[F["head_pose"]] is not None:
            assert set(p[F["head_pose"]]) == {F["yaw"], F["pitch"], F["roll"], F["score"]}
        g = p[F["gaze"]]
        assert (g is None) == (p[F["head_pose"]] is None)          # both need the face
        if g is not None:
            assert set(g) == {F["face"], F["eye"]}
            for part in (F["face"], F["eye"]):
                d = g[part]
                assert set(d) == {F[k] for k in ("yaw", "pitch", "vector", "valid", "score", "yaw_vehicle",
                                                 "pitch_vehicle", "zone")}
                if d[F["valid"]]:
                    assert abs(np.linalg.norm(d[F["vector"]]) - 1) < 1e-6 and isinstance(d[F["zone"]], str)
            assert g[F["face"]][F["valid"]]                         # with a face, the face direction is there
            assert g[F["eye"]][F["valid"]] == bool(r["pred"]["gaze"]["valid"])
        if p[F["eye_openness"]] is not None:
            assert set(p[F["eye_openness"]]) == {F["left"], F["right"]}
            assert any(v is not None for v in p[F["eye_openness"]].values())
            found = {o[F["label"]] for o in p[F["od"]]}
            for side in ("left", "right"):          # a side is reported only for a detected eye
                assert (p[F["eye_openness"]][F[side]] is not None) == (f"{side}_eye" in found)
        if p[F["mouth_openness"]] is not None:
            assert 0 <= p[F["mouth_openness"]] <= 1
    labels = {o[F["label"]] for r in s.records for o in to_pipeline(r, s.camera)[F["od"]]}
    assert "mouth" in labels and "face" not in labels
    assert any(o[F["label"]] == "face" for o in to_pipeline(s.records[0], s.camera, include_face=True)[F["od"]])


def test_mouth_box_inside_face_and_grows_with_openness():
    cabin, cam = _setup()
    heights = []
    for m in (0.0, 0.5, 1.0):
        tr = compute_truth(cabin, cam, _frame(cabin, m))
        fx1, fy1, fx2, fy2 = _box(tr, "face")
        mx1, my1, mx2, my2 = _box(tr, "mouth")
        assert fx1 <= mx1 < mx2 <= fx2 and fy1 <= my1 < my2 <= fy2
        assert tr["mouth_openness"] == m
        heights.append(my2 - my1)
    assert heights[0] < heights[1] < heights[2]


def test_plain_driver_frame_defaults_to_closed_mouth():
    cabin, cam = _setup()
    f = DriverFrame(0, 0, 0, 0, cabin.targets["road_ahead"], "road_ahead", 1, 1, False, False, "wheel", "none")
    assert compute_truth(cabin, cam, f)["mouth_openness"] == 0.0


def test_dropped_frames_produce_nulls():
    s = _sim(pipeline={"drop_frame_prob": 1.0})
    p = to_pipeline(s.records[10], s.camera)
    assert p[F["od"]] == []
    assert all(p[F[k]] is None for k in ("head_pose", "gaze", "eye_openness", "mouth_openness"))
    assert p[F["frame_id"]] == 10


def test_eye_refused_face_direction_still_given():
    s = Simulation({"name": "closed", "behavior": {"blink": {"rate_per_min": 0}},
                    "segments": [{"target": "road_ahead", "duration": 0.5, "eye_openness": 0.05}]}, "steering_column",
                   seed=1, emulator_overrides={"pipeline": {"drop_frame_prob": 0}})
    s.run()
    seen = 0
    for r in s.records:
        g = to_pipeline(r, s.camera)[F["gaze"]]
        if g is None:
            continue
        seen += 1
        e, f = g[F["eye"]], g[F["face"]]
        assert not e[F["valid"]] and e[F["score"]] == 0.0
        assert all(e[F[k]] is None for k in ("yaw", "pitch", "vector", "yaw_vehicle", "pitch_vehicle", "zone"))
        assert f[F["valid"]] and f[F["score"]] > 0 and isinstance(f[F["zone"]], str)
        assert f[F["yaw"]] == r["pred"]["face_dir"]["yaw_camera"]
    assert seen > 0


def test_record_without_face_dir_keeps_the_eye_gaze():
    s = _sim(pipeline={"drop_frame_prob": 0})
    r = next(r for r in s.records if r["pred"]["face_dir"] is not None)
    old = {**r, "pred": {k: v for k, v in r["pred"].items() if k != "face_dir"}}     # stored before face_dir existed
    g, g_new = to_pipeline(old, s.camera)[F["gaze"]], to_pipeline(r, s.camera)[F["gaze"]]
    assert g[F["face"]] is None and g[F["eye"]] == g_new[F["eye"]]


def test_truth_gaze_face_and_eye():
    s = _sim()
    for r in s.records[:20]:
        g = to_truth(r, s.camera)[F["gaze"]]
        common = {F[k] for k in ("yaw", "pitch", "vector", "yaw_vehicle", "pitch_vehicle", "vector_vehicle", "zone")}
        assert set(g[F["face"]]) == common and set(g[F["eye"]]) == common | {F["target"]}
        assert g[F["face"]][F["vector_vehicle"]] == r["truth"]["face_dir"]["dir_vehicle"]
        assert g[F["eye"]][F["vector_vehicle"]] == r["truth"]["gaze"]["dir_vehicle"]


def test_truth_record_matches_frame_and_carries_events():
    s = _sim()
    r = dict(s.records[5]); r["truth"] = {**r["truth"], "events": {"eyes_off_road": True}}
    t = to_truth(r, s.camera)
    assert t[F["frame_id"]] == 5 and t[F["events"]] == {"eyes_off_road": True}
    assert t[F["gaze"]][F["eye"]][F["zone"]] == s.records[5]["truth"]["gaze"]["zone"]
    ev = to_truth(s.records[5], s.camera)[F["events"]]       # real labels from events.py (+ phone_use_visible)
    assert set(ev) == {"eyes_off_road", "off_road_s", "eyes_closed", "closed_s", "phone_use", "yawning", "smoking",
                       "seatbelt_off", "phone_use_visible"}


def test_phone_visible_label():
    s = Simulation("phone_distraction", "steering_column", seed=0); s.run()
    tr = [to_truth(r, s.camera) for r in s.records]
    for r, t in zip(s.records, tr):
        vis = any(d["class"] == "phone" for d in r["truth"]["detections"])
        assert t[F["phone_visible"]] == vis
        assert t[F["events"]]["phone_use_visible"] == (t[F["events"]]["phone_use"] and vis)
    assert any(t[F["events"]]["phone_use"] and not t[F["phone_visible"]] for t in tr)   # lap phone out of view
    assert any(t[F["events"]]["phone_use_visible"] for t in tr)                           # call at the ear


def test_realtime_replay_tolerates_null_timestamps(tmp_path):
    p = tmp_path / "n.pipeline.jsonl"
    p.write_text(json.dumps({F["frame_id"]: 0, F["timestamp_ms"]: None}) + "\n" + json.dumps({F["frame_id"]: 1}) + "\n")
    assert [r[F["frame_id"]] for r in iter_pipeline(p, realtime=True)] == [0, 1]


def test_write_and_replay_pairs_align(tmp_path):
    s = _sim(pipeline={"drop_frame_prob": 0.3})
    paths = s.write(tmp_path, fmt="both")
    names = sorted(p.name for p in paths)
    assert names == ["mirror_check__steering_column.jsonl", "mirror_check__steering_column.pipeline.jsonl",
                     "mirror_check__steering_column.truth.jsonl"]
    pipe, truth = tmp_path / names[1], tmp_path / names[2]
    recs = list(iter_pipeline(pipe))
    assert len(recs) == len(s.records)
    ts = [r[F["timestamp_ms"]] for r in recs]
    assert ts == sorted(ts)
    pairs = list(iter_pairs(pipe, truth))
    assert len(pairs) == len(s.records)
    assert all(p[F["frame_id"]] == t[F["frame_id"]] for p, t in pairs)
    assert any(p[F["head_pose"]] is None for p, _ in pairs)   # some dropped frames, still aligned


def test_realtime_replay_sleeps_to_timestamps(tmp_path):
    p = tmp_path / "x.pipeline.jsonl"
    p.write_text("".join(json.dumps({F["frame_id"]: i, F["timestamp_ms"]: 1000 + 20 * i}) + "\n" for i in range(6)))
    t0 = time.monotonic()
    assert [r[F["frame_id"]] for r in iter_pipeline(p, realtime=True)] == list(range(6))
    # 100 ms of timestamps; the slack covers the ~16 ms tick of time.monotonic on Windows (seen failing at 0.095)
    assert time.monotonic() - t0 >= 0.08


def test_mapped_label_mode():
    from tests.conftest import MAPPED_CLASSES, MAPPED_LAYOUT
    from dms_model_simulator.config import load_yaml
    spec = load_yaml(MAPPED_LAYOUT)
    s = Simulation("phone_distraction", "steering_column", seed=1); s.run()
    seen = set()
    for r in s.records:
        p = to_pipeline(r, s.camera, labels="mapped", label_map=spec)
        sim = to_pipeline(r, s.camera, include_face=True)
        assert set(p) == set(sim)                               # same record shape; only od differs
        for k in ("head_pose", "gaze", "eye_openness", "mouth_openness"):
            assert p[F[k]] == sim[F[k]]                         # simulator-internal fields kept as they are
        labels = [o[F["label"]] for o in p[F["od"]]]
        seen |= set(labels)
        assert set(labels) <= MAPPED_CLASSES
        for o in p[F["od"]]:
            assert o[F["score"]] >= spec["conf_threshold"][o[F["label"]]]
        # an eye box is lids_open / lids_shut by its own predicted openness (the truth when the prediction is null)
        eo = r["pred"]["eye_openness"] or {}
        for d in r["pred"]["detections"]:
            if d["class"].endswith("_eye") and d["conf"] >= spec["conf_threshold"]["lids_open"]:
                side = d["class"][:-4]
                v = eo.get(side) if eo.get(side) is not None else r["truth"]["eye_openness"][side]
                want = "lids_shut" if v < spec["lids_threshold"] else "lids_open"
                assert any(x[F["bbox"]] == d["bbox"] and x[F["label"]] == want for x in p[F["od"]])
        n_hand = sum(o[F["label"]] == "hand" for o in sim[F["od"]])
        assert len(p[F["od"]]) <= len(sim[F["od"]]) - n_hand
    assert {"face_box", "lids_open", "jaw_shut", "handset"} <= seen
    r0 = s.records[0]
    assert "face_box" not in {o[F["label"]] for o in to_pipeline(r0, s.camera, False, labels="mapped",
                                                                  label_map=spec)[F["od"]]}


def test_mapped_mode_dropped_frame_bad_mode_and_layout_without_class_map():
    import pytest
    from tests.conftest import MAPPED_LAYOUT
    from dms_model_simulator.config import load_yaml
    s = _sim(pipeline={"drop_frame_prob": 1.0})
    p = to_pipeline(s.records[3], s.camera, labels="mapped", label_map=load_yaml(MAPPED_LAYOUT))
    assert p[F["od"]] == [] and all(p[F[k]] is None for k in ("head_pose", "gaze", "eye_openness", "mouth_openness"))
    with pytest.raises(ValueError, match="labels"):
        to_pipeline(s.records[3], s.camera, labels="yolo")
    ok = _sim()
    with pytest.raises(ValueError, match="class_map"):                  # default layout: generic, no class_map
        to_pipeline(ok.records[3], ok.camera, labels="mapped")
    with pytest.raises(ValueError, match="class_map"):
        to_pipeline(ok.records[3], ok.camera, labels="mapped", label_map={"class_ids": {"face": 0}})


def test_simulation_write_mapped_labels(tmp_path):
    import pytest
    from tests.conftest import MAPPED_CLASSES, MAPPED_LAYOUT
    s = _sim()
    s.write(tmp_path, fmt="pipeline", labels="mapped", dms_layout=MAPPED_LAYOUT)
    recs = list(iter_pipeline(tmp_path / "mirror_check__steering_column.pipeline.jsonl"))
    labels = {o[F["label"]] for r in recs for o in r[F["od"]]}
    assert labels <= MAPPED_CLASSES and {"face_box", "lids_open"} <= labels     # face kept by default in mapped mode
    with pytest.raises(ValueError, match="class_map"):
        s.write(tmp_path / "d", fmt="pipeline", labels="mapped")


def test_cli_labels_switch(tmp_path):
    from tests.conftest import MAPPED_CLASSES, MAPPED_LAYOUT
    from dms_model_simulator import cli
    sc = {"name": "short", "segments": [{"target": "road_ahead", "duration": 0.5}]}
    p = tmp_path / "short.yaml"; p.write_text(json.dumps(sc))
    cli.main(["run", "--scenario", str(p), "--format", "pipeline", "--labels", "sim", "--out", str(tmp_path / "sim")])
    cli.main(["run", "--scenario", str(p), "--format", "pipeline", "--labels", "mapped", "--layout", str(MAPPED_LAYOUT),
              "--out", str(tmp_path / "mapped")])
    lab = lambda m: {o["label"] for l in (tmp_path / m / "short__steering_column.pipeline.jsonl").read_text().splitlines()
                     for o in json.loads(l)["od"]}
    assert "left_eye" in lab("sim") and "face" not in lab("sim")
    assert lab("mapped") <= MAPPED_CLASSES and {"face_box", "lids_open"} <= lab("mapped")

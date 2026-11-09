import json
import pytest
from dms_model_simulator.binlog import DEFAULT_LAYOUT, DmsBinLayout, read_dms_bin, write_dms_bin
from dms_model_simulator.config import load_yaml
from dms_model_simulator.replay import iter_dms_bin
from dms_model_simulator.schema import mapped_detections, to_pipeline
from dms_model_simulator.simulate import Simulation
from tests.conftest import MAPPED_CLASSES, MAPPED_LAYOUT


@pytest.fixture(scope="module")
def sim():
    s = Simulation("phone_distraction", "steering_column", seed=3)
    s.records = s.run()[:150]
    return s


def _expected(sim, layout):
    """Per frame: list of (class_id, xyxy, score) the writer should emit."""
    out = []
    for r in sim.records:
        if layout.class_map is not None:
            dets, _, _ = mapped_detections(r, layout.spec)
            out.append([(layout.class_ids[d["class"]], d["bbox"], d["conf"]) for d in dets])
            continue
        od = to_pipeline(r, sim.camera, include_face=True)["od"]
        out.append([(layout.class_ids[o["label"]], o["bbox"], o["score"]) for o in od if o["label"] in layout.class_ids])
    return out


def _check_roundtrip(sim, layout, path):
    st = write_dms_bin(sim.records, sim.camera, layout, path)
    frames = list(read_dms_bin(path, layout))
    exp = _expected(sim, layout)
    assert len(frames) == len(sim.records) == st["frames"]
    assert sum(len(f["objects"]) for f in frames) == st["objects"] == sum(len(e) for e in exp)
    for fr, r, objs in zip(frames, sim.records, exp):
        h = fr["header"]
        assert fr["frame_id"] == r["frame"]
        assert abs(fr["timestamp_ms"] - r["t"] * 1000.0) <= 1.0
        if layout.ts_ms_is_int:                  # integer ms field: exactly the pipeline timestamp, as an int
            assert isinstance(fr["timestamp_ms"], int) and fr["timestamp_ms"] == to_pipeline(r, sim.camera)["timestamp_ms"]
        assert (h["width"], h["height"]) == (sim.camera.width, sim.camera.height)
        assert h["num_objects"] == len(objs) == len(fr["objects"])
        for o, (cid, xyxy, score) in zip(fr["objects"], objs):
            assert o["class_id"] == cid
            assert o["bbox_xyxy"] == pytest.approx(xyxy, abs=1e-3)
            assert o["conf"] == pytest.approx(score, abs=1e-3)
    return st, frames


def test_default_layout_is_generic():
    layout = DmsBinLayout.from_yaml()
    assert layout.source == str(DEFAULT_LAYOUT)
    assert (layout.header.size, layout.obj.size) == (18, 21)
    assert layout.header.roles == ["timestamp", "frame_index", "width", "height", "num_objects"]
    assert layout.class_map is None and layout.conf_threshold == {}
    assert layout.magic == b"" and layout.version is None and layout.file_header_bytes() == b""
    assert layout.class_names[layout.class_ids["left_eye"]] == "eye" == layout.class_names[layout.class_ids["right_eye"]]


def test_roundtrip_generic_layout(sim, tmp_path):
    layout = DmsBinLayout.from_yaml()
    st, frames = _check_roundtrip(sim, layout, tmp_path / "dms.bin")
    assert st["skipped"] == {} and st["below_threshold"] == {}
    # xywh on disk: w/h are sizes, not corners
    o = next(o for f in frames for o in f["objects"])
    assert o["w"] == pytest.approx(o["bbox_xyxy"][2] - o["x"], abs=1e-3)
    # both eyes share one id, read back as "eye"
    cls = {o["class"] for f in frames for o in f["objects"]}
    assert "eye" in cls and "left_eye" not in cls and "hand" in cls
    # file size = frames * header + objects * object (no file header in the default layout)
    size = (tmp_path / "dms.bin").stat().st_size
    assert size == len(frames) * layout.header.size + st["objects"] * layout.obj.size
    # replay iterator yields the same frames
    assert [f["frame_id"] for f in iter_dms_bin(tmp_path / "dms.bin", layout)] == [f["frame_id"] for f in frames]


def test_roundtrip_other_endianness_and_order(sim, tmp_path):
    spec = load_yaml(DEFAULT_LAYOUT)
    spec.update({
        "endian": "big", "bbox_format": "xyxy", "timestamp_unit": "us", "conf_scale": 10000,
        "file_header": {"magic": "DMSB", "version": 7, "version_type": "u16"},
        "header_fields": [{"name": "num_objects", "type": "u32"}, {"name": "flags", "type": "u8"},
                          {"name": "pad", "type": "pad", "size": 3}, {"name": "frame_index", "type": "u64"},
                          {"name": "height", "type": "i32"}, {"name": "width", "type": "i32"},
                          {"name": "timestamp", "type": "f64"}, {"name": "reserved", "type": "u32"},
                          {"name": "slots", "type": "u16", "count": 3}],
        "object_fields": [{"name": "score", "type": "u16"}, {"name": "x2", "type": "f64"},
                          {"name": "y2", "type": "f64"}, {"name": "x1", "type": "f64"},
                          {"name": "y1", "type": "f64"}, {"name": "class_id", "type": "i32"}],
    })
    layout = DmsBinLayout(spec)
    assert layout.header.size == 4 + 1 + 3 + 8 + 4 + 4 + 8 + 4 + 6
    path = tmp_path / "dms_be.bin"
    _, frames = _check_roundtrip(sim, layout, path)
    assert path.read_bytes()[:6] == b"DMSB\x00\x07"                      # magic + big-endian u16 version
    # header fields the simulator does not fill are written as their default (0) and read back raw
    assert all(f["header"]["reserved"] == 0 and f["header"]["flags"] == 0 and f["header"]["slots"] == [0, 0, 0]
               for f in frames)
    # wrong layout (little endian default) does not silently parse this file
    with pytest.raises(ValueError):
        list(read_dms_bin(path, DmsBinLayout.from_yaml()))


def test_mapped_layout_roundtrip(sim, tmp_path):
    layout = DmsBinLayout.from_yaml(MAPPED_LAYOUT)
    assert (layout.header.size, layout.obj.size) == (21, 20)
    path = tmp_path / "mapped.bin"
    st, frames = _check_roundtrip(sim, layout, path)
    data = path.read_bytes()
    assert data[:6] == b"TLY1\x03\x00"                                   # magic + little-endian u16 version 3
    assert len(data) == 6 + len(frames) * layout.header.size + st["objects"] * layout.obj.size
    assert all(f["header"]["spare"] == [0, 0, 0, 0] for f in frames)
    names = {o["class"] for f in frames for o in f["objects"]}
    assert names <= MAPPED_CLASSES and {"face_box", "lids_open", "jaw_shut"} <= names
    n_hand = sum(d["class"] == "hand" for r in sim.records if not r["pred"]["dropped"] for d in r["pred"]["detections"])
    assert n_hand > 0 and st["skipped"] == {"hand": n_hand}
    thr = layout.conf_threshold
    assert all(o["conf"] >= thr[o["class"]] - 1e-4 for f in frames for o in f["objects"])
    # bad magic / version mismatch are refused
    other = load_yaml(MAPPED_LAYOUT); other["file_header"]["version"] = 4
    with pytest.raises(ValueError, match="version mismatch"):
        list(read_dms_bin(path, other))
    other["file_header"]["magic"] = "XXXX"
    with pytest.raises(ValueError, match="bad magic"):
        list(read_dms_bin(path, other))


def test_mapped_layout_eye_and_mouth_class_follow_predicted_openness(sim, tmp_path):
    spec = load_yaml(MAPPED_LAYOUT)
    path = tmp_path / "m.bin"
    write_dms_bin(sim.records, sim.camera, spec, path)
    n_eye = 0
    for fr, r in zip(read_dms_bin(path, spec), sim.records):
        eo, mo = r["pred"]["eye_openness"] or {}, r["pred"]["mouth_openness"]
        if mo is None:                                   # null prediction: the class map uses the truth value
            mo = r["truth"]["mouth_openness"]
        for o in fr["objects"]:
            if o["class"].startswith("lids_"):
                d = next(d for d in r["pred"]["detections"] if d["class"].endswith("_eye")
                         and d["bbox"] == pytest.approx(o["bbox_xyxy"], abs=1e-3))
                side = d["class"][:-4]
                v = eo.get(side) if eo.get(side) is not None else r["truth"]["eye_openness"][side]
                assert o["class"] == ("lids_shut" if v < spec["lids_threshold"] else "lids_open")
                n_eye += 1
            elif o["class"].startswith("jaw_"):
                assert o["class"] == ("jaw_shut" if mo < spec["jaw_threshold"] else "jaw_open")
    assert n_eye > 100


def test_timestamp_ms_type(sim, tmp_path):
    p = tmp_path / "a.bin"
    write_dms_bin(sim.records, sim.camera, None, p)
    ts = [f["timestamp_ms"] for f in read_dms_bin(p)]
    assert all(type(t) is int for t in ts) and ts[121] == 4033          # was 4033.0000000000005
    spec = load_yaml(DEFAULT_LAYOUT); spec["timestamp_unit"] = "us"
    write_dms_bin(sim.records, sim.camera, spec, p)
    ts = [f["timestamp_ms"] for f in read_dms_bin(p, spec)]
    assert ts[1] == 33.333 and ts[121] == 4033.333                       # float, rounded to 1e-6 ms


def test_truncated_file(sim, tmp_path, capsys):
    from dms_model_simulator import cli
    p = tmp_path / "t.bin"
    write_dms_bin(sim.records, sim.camera, None, p)
    data = p.read_bytes()
    p.write_bytes(data[:-3])
    with pytest.raises(ValueError, match="truncated object records"):
        list(read_dms_bin(p))
    with pytest.warns(UserWarning, match="truncated"):
        frames = list(read_dms_bin(p, strict=False))
    assert len(frames) == len(sim.records) - 1
    with pytest.warns(UserWarning):
        assert len(list(iter_dms_bin(p, strict=False))) == len(frames)
    capsys.readouterr()
    cli.main(["inspect-bin", str(p), "--head", "1"])
    out = capsys.readouterr().out
    assert f"-- {len(frames)} frames" in out and "warning: truncated" in out and out.startswith("[frame 0]")
    # a cut frame header is reported too
    p.write_bytes(data[:7])
    with pytest.raises(ValueError, match="truncated frame header"):
        list(read_dms_bin(p))


def test_inspect_bin_with_a_layout(sim, tmp_path, capsys):
    from dms_model_simulator import cli
    p = tmp_path / "m.bin"
    write_dms_bin(sim.records[:3], sim.camera, MAPPED_LAYOUT, p)
    capsys.readouterr()
    cli.main(["inspect-bin", str(p), "--layout", str(MAPPED_LAYOUT)])
    out = capsys.readouterr().out
    assert "-- 3 frames" in out and "header 21 B, object 20 B" in out and "spare: [0, 0, 0, 0]" in out


def test_layout_validation_and_pack_errors(sim, tmp_path):
    spec = load_yaml(DEFAULT_LAYOUT); spec["conf_scale"] = 0
    with pytest.raises(ValueError, match="conf_scale"):
        DmsBinLayout(spec)
    spec = load_yaml(DEFAULT_LAYOUT); spec["header_fields"][0] = {"name": "timestamp", "type": "u32"}
    spec["timestamp_unit"] = "us"
    rec = dict(sim.records[0], t=4295.0)                                 # 4295 s in us overflows u32
    with pytest.raises(ValueError, match="'timestamp'"):
        write_dms_bin([rec], sim.camera, spec, tmp_path / "o.bin")
    spec = load_yaml(DEFAULT_LAYOUT); spec["header_fields"] = spec["header_fields"][:-1]
    with pytest.raises(ValueError, match="num_objects"):
        DmsBinLayout(spec)
    spec = load_yaml(DEFAULT_LAYOUT); spec["bbox_format"] = "xyxy"         # object fields are x, y, w, h
    with pytest.raises(ValueError, match="box fields"):
        DmsBinLayout(spec)
    spec = load_yaml(MAPPED_LAYOUT); spec["class_map"]["phone"] = "pocket_phone"
    with pytest.raises(ValueError, match="not in class_ids"):
        DmsBinLayout(spec)
    spec = load_yaml(MAPPED_LAYOUT); del spec["lids_threshold"]
    with pytest.raises(ValueError, match="threshold key"):
        DmsBinLayout(spec)


def test_unknown_class_skipped(sim, tmp_path):
    spec = load_yaml(DEFAULT_LAYOUT)
    del spec["class_ids"]["hand"]
    layout = DmsBinLayout(spec)
    st, frames = _check_roundtrip(sim, layout, tmp_path / "dms.bin")
    n_hand = sum(1 for r in sim.records for o in to_pipeline(r, sim.camera, include_face=True)["od"] if o["label"] == "hand")
    assert n_hand > 0 and st["skipped"] == {"hand": n_hand}
    assert all(o["class_id"] != 3 for f in frames for o in f["objects"])


def test_conf_threshold_without_class_map(sim, tmp_path):
    spec = load_yaml(DEFAULT_LAYOUT)
    spec["conf_threshold"] = {"hand": 0.99}
    st = write_dms_bin(sim.records, sim.camera, spec, tmp_path / "dms.bin")
    frames = list(read_dms_bin(tmp_path / "dms.bin", spec))
    assert st["below_threshold"].get("hand", 0) > 0
    assert all(o["conf"] >= 0.99 for f in frames for o in f["objects"] if o["class"] == "hand")


@pytest.mark.parametrize("flag, pipe_face, bin_face", [([], False, True), (["--include-face"], True, True),
                                                      (["--no-face"], False, False)])
def test_run_face_switch_applies_to_pipeline_and_dmsbin(tmp_path, flag, pipe_face, bin_face):
    from dms_model_simulator import cli
    sc = {"name": "short", "segments": [{"target": "road_ahead", "duration": 0.5}]}
    p = tmp_path / "short.yaml"
    p.write_text(json.dumps(sc))
    cli.main(["run", "--scenario", str(p), "--format", "all", "--out", str(tmp_path)] + flag)
    labels = {o["label"] for line in (tmp_path / "short__steering_column.pipeline.jsonl").read_text().splitlines()
              for o in json.loads(line)["od"]}
    assert ("face" in labels) == pipe_face and "left_eye" in labels
    layout = DmsBinLayout.from_yaml()                     # generic: face 0, both eyes 1
    frames = list(read_dms_bin(tmp_path / "short__steering_column.dms.bin", layout))
    ids = {o["class_id"] for f in frames for o in f["objects"]}
    assert (layout.class_ids["face"] in ids) == bin_face and layout.class_ids["left_eye"] in ids

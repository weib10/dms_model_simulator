"""Export of simulated model outputs: frame_yaml / dms.bin exporters and readers, export_scenario, `dms-sim export`."""
import json
import pytest
import yaml
from dms_model_simulator import cli
from dms_model_simulator import export as X
from dms_model_simulator.binlog import read_dms_bin, write_dms_bin
from dms_model_simulator.config import deep_merge
from dms_model_simulator.export import ExportError
from tests.conftest import MAPPED_CLASSES, MAPPED_LAYOUT

CAM = {"name": "dash_right", "width": 1920, "height": 1080}
SHORT = {"name": "short", "segments": [{"target": "road_ahead", "duration": 0.5}]}


def _det(cls, box, conf):
    return {"class": cls, "bbox": list(box), "conf": conf}


def _rec(frame, t, dets, dropped=False):
    pred = {"dropped": dropped, "detections": [] if dropped else dets,
            "eye_openness": None if dropped else {"left": 0.9, "right": 0.1},
            "mouth_openness": None if dropped else 0.6}
    truth = {"eye_openness": {"left": 1.0, "right": 1.0}, "mouth_openness": 0.05}
    return {"frame": frame, "t": t, "camera": "dash_right", "truth": truth, "pred": pred}


FACE = (100.0, 200.0, 300.0, 450.0)


def _records():
    full = [_det("face", FACE, 0.9), _det("left_eye", (140, 300, 180, 320), 0.8),     # lids_open (0.9)
            _det("right_eye", (220, 300, 260, 320), 0.6),                             # lids_shut (0.1)
            _det("mouth", (180, 380, 260, 410), 0.5),                                 # jaw_open (0.6)
            _det("hand", (400, 500, 480, 580), 0.9),                                  # not in the mapped layout
            _det("phone", (500, 500, 560, 600), 0.4)]                                 # handset below 0.5
    return [_rec(0, 0.0, full), _rec(1, 0.05, [], dropped=True),
            _rec(2, 0.10, [_det("face", FACE, 0.95), _det("phone", (500, 500, 560, 600), 0.8)])]


def _same_frames(bin_frames, yaml_frames):
    bin_frames, yaml_frames = list(bin_frames), list(yaml_frames)
    assert len(bin_frames) == len(yaml_frames)
    for b, y in zip(bin_frames, yaml_frames):
        assert (y["frame_id"], y["timestamp_ms"]) == (b["frame_id"], b["timestamp_ms"])
        assert [o["label"] for o in y["od"]] == [o["label"] for o in b["od"]]
        for oy, ob in zip(y["od"], b["od"]):
            assert oy["bbox"] == pytest.approx(ob["bbox"], abs=1e-3) and oy["score"] == pytest.approx(ob["score"], abs=1e-4)


# ---- frame_yaml exporter / reader ------------------------------------------------------------------
def test_frame_yaml_export_default_layout(tmp_path):
    recs = _records()
    d = tmp_path / "a.frames"
    (d / "labels").mkdir(parents=True)
    (d / "labels" / "00000099.yaml").write_text("detections: []\n")          # stale frame of an earlier export
    (d / "labels" / "notes.yaml").write_text("keep: me\n")
    st = X.export_frame_yaml(recs, CAM, d)
    assert st["frames"] == 3 and st["objects"] == 6 + 0 + 2 and st["images"] == 0 and st["format"] == "frame_yaml"
    assert st["skipped"] == {} and st["below_threshold"] == {}
    names = sorted(f.name for f in (d / "labels").iterdir())
    assert names == ["00000000.yaml", "00000001.yaml", "00000002.yaml", "notes.yaml"]
    assert (d / "images").is_dir() and not any((d / "images").iterdir())
    f0 = yaml.safe_load((d / "labels" / "00000000.yaml").read_text())
    assert f0["timestamp_msec"] == 0 and len(f0["detections"]) == 6
    assert f0["detections"][0] == {"class_id": 0, "confidence": 0.9, "left": 100.0, "top": 200.0, "width": 200.0,
                                   "height": 250.0, "class_name": "face"}
    assert [o["class_name"] for o in f0["detections"]] == ["face", "eye", "eye", "mouth", "hand", "phone"]
    assert (d / "labels" / "00000001.yaml").read_text() == "detections: []\ntimestamp_msec: 50\n"   # zero-object frame
    assert yaml.safe_load((d / "labels" / "00000002.yaml").read_text())["timestamp_msec"] == 100
    # same objects as the dms.bin writer (shared encoder)
    write_dms_bin(recs, CAM, None, tmp_path / "a.dms.bin")
    _same_frames(read_dms_bin(tmp_path / "a.dms.bin"), X.read_frame_yaml(d))
    assert [f["frame_id"] for f in X.read_frame_yaml(d / "labels")] == [0, 1, 2]


def test_frame_yaml_export_mapped_layout(tmp_path):
    recs = _records()
    d = tmp_path / "m.frames"
    st = X.export_frame_yaml(recs, CAM, d, layout=MAPPED_LAYOUT)
    assert st["frames"] == 3 and st["objects"] == 4 + 0 + 2
    assert st["skipped"] == {"hand": 1} and st["below_threshold"] == {"handset": 1}
    f0 = yaml.safe_load((d / "labels" / "00000000.yaml").read_text())
    assert [o["class_name"] for o in f0["detections"]] == ["face_box", "lids_open", "lids_shut", "jaw_open"]
    assert [o["class_id"] for o in f0["detections"]] == [4, 1, 6, 2]
    f2 = yaml.safe_load((d / "labels" / "00000002.yaml").read_text())
    assert [o["class_name"] for o in f2["detections"]] == ["face_box", "handset"]
    write_dms_bin(recs, CAM, MAPPED_LAYOUT, tmp_path / "m.dms.bin")
    _same_frames(read_dms_bin(tmp_path / "m.dms.bin", MAPPED_LAYOUT), X.read_frame_yaml(d, MAPPED_LAYOUT))
    # the layout can also come from the export config
    st2 = X.export(recs, CAM, "frame_yaml", tmp_path / "c.frames", {"layout": str(MAPPED_LAYOUT)})
    assert (st2["objects"], st2["skipped"]) == (st["objects"], st["skipped"])
    assert {o["label"] for f in X.read_frame_yaml(tmp_path / "c.frames", MAPPED_LAYOUT) for o in f["od"]} <= MAPPED_CLASSES


def test_frame_yaml_reader_and_option_errors(tmp_path):
    recs = _records()
    d = tmp_path / "a.frames"
    X.export_frame_yaml(recs, CAM, d, layout=MAPPED_LAYOUT)
    (d / "labels" / "00000001.yaml").write_text("detections:\n  - {class_id: 4, class_name: belt_strap, confidence: 1,"
                                                " left: 0, top: 0, width: 1, height: 1}\n")
    with pytest.raises(ExportError, match=r"00000001\.yaml.*does not match class_id 4"):
        list(X.read_frame_yaml(d, MAPPED_LAYOUT))
    (d / "labels" / "00000001.yaml").write_text("detections:\n  - {class_id: 9, confidence: 1, left: 0, top: 0,"
                                                " width: 1, height: 1}\n")
    with pytest.raises(ExportError, match="class_id 9 not in the layout"):
        list(X.read_frame_yaml(d, MAPPED_LAYOUT))
    (d / "labels" / "00000001.yaml").write_text("detections: 3\n")
    with pytest.raises(ExportError, match="detections must be a list"):
        list(X.read_frame_yaml(d, MAPPED_LAYOUT))
    with pytest.raises(ExportError, match="no frame YAML"):
        list(X.read_frame_yaml(tmp_path))
    with pytest.raises(ExportError, match="needs the simulation's Camera and Cabin"):
        X.export_frame_yaml(recs, CAM, d, image="render")
    with pytest.raises(ExportError, match="image must be one of"):
        X.export_frame_yaml(recs, CAM, d, image="video")
    with pytest.raises(ExportError, match="needs format frame_yaml"):
        X.export(recs, CAM, "dms_bin", tmp_path / "x.bin", image="render")
    with pytest.raises(ExportError, match="format must be one of"):
        X.export(recs, CAM, "csv", tmp_path / "x.csv")
    with pytest.raises(ExportError, match="digits 1 is too small"):
        X.export_frame_yaml([_rec(i, i * 0.05, []) for i in range(11)], CAM, tmp_path / "d1",
                            deep_merge(X.load_config(), {"frame_yaml": {"digits": 1}}))


def test_read_frame_yaml_orders_files_by_frame_number(tmp_path):
    d = tmp_path / "labels"
    d.mkdir()
    for i in (2, 9, 10, 11):
        (d / f"{i}.yaml").write_text("detections: []\n")
    assert [f["frame_id"] for f in X.read_frame_yaml(d)] == [2, 9, 10, 11]
    assert all(f["timestamp_ms"] is None for f in X.read_frame_yaml(d))


def test_dms_bin_export_and_sniff_format(tmp_path):
    st = X.export(_records(), CAM, "dms_bin", tmp_path / "a.dms.bin", layout=MAPPED_LAYOUT)
    assert st["format"] == "dms_bin" and st["frames"] == 3 and st["objects"] == 6 and st["skipped"] == {"hand": 1}
    X.export(_records(), CAM, "frame_yaml", tmp_path / "a.frames", layout=MAPPED_LAYOUT)
    _same_frames(read_dms_bin(st["path"], MAPPED_LAYOUT), X.read_frame_yaml(tmp_path / "a.frames", MAPPED_LAYOUT))
    assert X.sniff_format(tmp_path / "a.frames") == "frame_yaml" and X.sniff_format(st["path"]) == "dms_bin"


def test_frame_yaml_render_images(tmp_path):
    from PIL import Image
    sim, st, _ = X.export_scenario(SHORT, "dash_right", "frame_yaml", tmp_path, fps=10, image="render")
    imgs = sorted((st["image_dir"]).iterdir())
    assert st["images"] == st["frames"] == len(imgs) == 5
    assert [p.stem for p in imgs] == [p.stem for p in sorted(st["labels_dir"].iterdir())]
    im = Image.open(imgs[0])
    assert im.size == (sim.camera.width, sim.camera.height) and im.mode == "RGB"


# ---- export_scenario --------------------------------------------------------------------------------
def test_export_scenario_writes_pipeline_and_truth(tmp_path):
    sim, st, truth = X.export_scenario(SHORT, "dash_right", "dms_bin", tmp_path / "a", fps=10)
    stem = "short__dash_right"
    assert truth == (tmp_path / "a" / f"{stem}.truth.jsonl").resolve() and truth.exists()
    assert (tmp_path / "a" / f"{stem}.pipeline.jsonl").exists() and not (tmp_path / "a" / f"{stem}.jsonl").exists()
    assert st["path"].name == f"{stem}.dms.bin" and st["frames"] == len(sim.records) == 5
    _, _, _ = X.export_scenario(SHORT, "dash_right", "frame_yaml", tmp_path / "b", fps=10, full=True)
    full = [json.loads(ln) for ln in (tmp_path / "b" / f"{stem}.jsonl").read_text().splitlines()]
    assert len(full) == 5 and {"truth", "pred"} <= set(full[0])
    with pytest.raises(ExportError, match="format must be one of"):
        X.export_scenario(SHORT, "dash_right", "csv", tmp_path / "c", fps=10)


def test_export_scenario_overrides_and_mapped_labels(tmp_path):
    sim, st, _ = X.export_scenario(SHORT, "dash_right", "frame_yaml", tmp_path / "d", fps=10,
                                   emulator_overrides={"pipeline": {"drop_frame_prob": 1.0}})
    assert st["objects"] == 0 and all(r["pred"]["dropped"] for r in sim.records)
    cfg = deep_merge(X.load_config(), {"simulation": {"labels": "mapped", "fps": 10}, "layout": str(MAPPED_LAYOUT)})
    sim, st, _ = X.export_scenario(SHORT, "dash_right", "frame_yaml", tmp_path / "m", config=cfg)
    assert sim.behavior.fps == 10 and st["frames"] == 5
    pipe = [json.loads(ln) for ln in (tmp_path / "m" / "short__dash_right.pipeline.jsonl").read_text().splitlines()]
    labels = {o["label"] for p in pipe for o in p["od"]}
    assert labels <= MAPPED_CLASSES and "face_box" in labels
    exported = {o["label"] for f in X.read_frame_yaml(st["path"], MAPPED_LAYOUT) for o in f["od"]}
    assert exported == labels                                            # JSONL and export use the same layout


# ---- CLI ---------------------------------------------------------------------------------------------
def test_cli_export_writes_export_truth_and_card(tmp_path, capsys):
    capsys.readouterr()
    cli.main(["export", "--scenario", "phone_distraction", "--camera", "dash_right", "--format", "frame_yaml",
              "--fps", "10", "--out", str(tmp_path / "fy"), "--json"])
    fy = json.loads(capsys.readouterr().out)
    assert fy["ok"] and fy["frames"] == 145 and fy["fps"] == 10
    assert fy["input"].endswith(".frames") and fy["labels_dir"].endswith("labels") and fy["images"] == 0
    assert fy["truth"].endswith(".truth.jsonl") and fy["card"].endswith("card.md")
    from pathlib import Path
    card = Path(fy["card"])
    assert card.exists() and (card.parent / "card.json").exists()
    assert card.parent == tmp_path / "fy" / "phone_distraction__dash_right__card"
    cj = json.loads((card.parent / "card.json").read_text(encoding="utf-8"))
    assert cj["frames"] == 145 and cj["scenario"] == "phone_distraction" and cj["camera"] == "dash_right"
    assert (tmp_path / "fy" / "phone_distraction__dash_right.jsonl").exists()          # full JSONL with error flags
    cli.main(["export", "--scenario", "phone_distraction", "--camera", "dash_right", "--format", "dms_bin",
              "--fps", "10", "--out", str(tmp_path / "bin"), "--profile", "drops"])
    out = capsys.readouterr().out
    assert out.startswith("wrote ") and ".dms.bin (dms_bin, 145 frames" in out and "card.md" in out
    cj = json.loads((tmp_path / "bin" / "phone_distraction__dash_right__card" / "card.json").read_text(encoding="utf-8"))
    assert cj["profile"] == "drops"


def test_cli_export_with_a_layout_config(tmp_path, capsys):
    cfg = deep_merge(X.load_config(), {"layout": str(MAPPED_LAYOUT), "simulation": {"labels": "mapped"}})
    p = tmp_path / "export.yaml"
    p.write_text(yaml.safe_dump(cfg))
    sc = tmp_path / "short.yaml"
    sc.write_text(json.dumps(SHORT))
    capsys.readouterr()
    cli.main(["export", "--scenario", str(sc), "--camera", "dash_right", "--format", "dms_bin", "--config", str(p),
              "--out", str(tmp_path / "o"), "--json"])
    info = json.loads(capsys.readouterr().out)
    assert info["skipped"].keys() <= {"hand"}
    assert {o["class"] for f in read_dms_bin(info["input"], MAPPED_LAYOUT) for o in f["objects"]} <= MAPPED_CLASSES


def test_cli_export_errors(tmp_path, capsys):
    capsys.readouterr()
    for argv in (["export", "--scenario", "phone_distraction", "--camera", "nope", "--format", "dms_bin"],
                 ["export", "--scenario", "no_such_scenario", "--camera", "dash_right", "--format", "frame_yaml"]):
        with pytest.raises(SystemExit) as e:
            cli.main(argv + ["--out", str(tmp_path / "o"), "--json"])
        assert e.value.code == 2, argv
        err = json.loads(capsys.readouterr().out)
        assert err["ok"] is False and err["error"], argv
    with pytest.raises(SystemExit) as e:
        cli.main(["export", "--scenario", "phone_distraction", "--camera", "nope", "--format", "dms_bin",
                  "--out", str(tmp_path / "o")])
    assert e.value.code == 2 and "dms-sim export: KeyError: unknown camera 'nope'" in capsys.readouterr().err
    with pytest.raises(SystemExit):                                      # unknown format
        cli.main(["export", "--scenario", "phone_distraction", "--camera", "dash_right", "--format", "csv"])

import json
import numpy as np
import pytest
from dms_model_simulator import cli
from dms_model_simulator.camera import Camera
from dms_model_simulator.config import load_configs
from dms_model_simulator.simulate import Simulation
from dms_model_simulator.variation import sample_variant
from dms_model_simulator.world import Cabin

SC = {"name": "t", "segments": [{"target": "road_ahead", "duration": 0.2}]}
CAM = "steering_column"


def test_variants_are_deterministic_within_range_and_distinct():
    cfg = {"driver": {"eye_point_offset": {"x": [-0.1, 0.1], "z": [-0.08, 0.08]}, "head_scale": [0.9, 1.1]},
           "camera": {"mount_rotation_deg": {"yaw": [-2, 2]}}}
    a, b, c = sample_variant(3, cfg), sample_variant(3, cfg), sample_variant(4, cfg)
    assert a == b and a != c and a["variant"] == 3
    x, y, z = a["cabin"]["driver"]["eye_point_offset"]
    assert -0.1 <= x <= 0.1 and y == 0.0 and -0.08 <= z <= 0.08
    assert 0.9 <= a["cabin"]["driver"]["head_scale"] <= 1.1
    assert a["camera"]["mount_rotation_deg"][1:] == [0.0, 0.0]
    with pytest.raises(ValueError):
        sample_variant(0, {"driver": {"eye_point_offset": {"w": [0, 1]}}})
    assert sample_variant(0)["cabin"]["driver"]          # default configs/variation.yaml loads
    with pytest.raises(ValueError):
        sample_variant(-1)


def test_head_scale_scales_head_geometry():
    cfgs = load_configs()
    base = Cabin(cfgs["cabin"])
    cfgs["cabin"]["driver"]["head_scale"] = 1.1
    big = Cabin(cfgs["cabin"])
    h0, h1 = base.head_state(0, 0, 0), big.head_state(0, 0, 0)
    d = lambda h: np.linalg.norm(h.landmarks["left_eye_center"] - h.landmarks["right_eye_center"])
    assert np.isclose(d(h1), 1.1 * d(h0)) and np.allclose(h1.origin, h0.origin)
    assert np.isclose(big.eye_half_height, 1.1 * base.eye_half_height)
    assert np.isclose(big.self_occluders[0]["radius"], 1.1 * base.self_occluders[0]["radius"])
    assert np.allclose(big.reference_head_center, base.reference_head_center)
    assert np.isclose(big.annotation_boxes["eye_top"], 1.1 * base.annotation_boxes["eye_top"])


def test_moved_driver_keeps_the_fixed_camera_aim():
    nominal = Simulation(SC, CAM)
    moved = Simulation({**SC, "cabin": {"driver": {"eye_point_offset": [0.0, 0.0, 0.05], "head_scale": 0.95}}}, CAM)
    assert np.allclose(moved.camera.R, nominal.camera.R) and np.allclose(moved.camera.position, nominal.camera.position)
    assert np.allclose(moved.cabin.eye_point, [0, 0, 0.05])
    # the moved eye point projects off the image centre (camera did not follow the driver)
    assert not np.allclose(moved.camera.project(moved.cabin.eye_point)[0], [moved.camera.cx, moved.camera.cy])


def test_mount_offset_and_rotation():
    cfgs = load_configs(); cabin = Cabin(cfgs["cabin"])
    spec = {**cfgs["cameras"]["defaults"], **cfgs["cameras"]["cameras"][CAM]}
    cam = Camera(CAM, spec, cabin.reference_head_center)
    eye = cabin.eye_point
    shifted = Camera(CAM, {**spec, "mount_offset": [0.0, 0.0, 0.01]}, cabin.reference_head_center)
    assert np.allclose(shifted.R, cam.R) and np.allclose(shifted.position, cam.position + [0, 0, 0.01])
    rot = lambda r: Camera(CAM, {**spec, "mount_rotation_deg": r}, cabin.reference_head_center)
    u, v = rot([2, 0, 0]).project(eye)[0]
    assert u < cam.cx - 1 and abs(v - cam.cy) < 1e-6          # pan right: the scene moves left
    u, v = rot([0, 2, 0]).project(eye)[0]
    assert v > cam.cy + 1 and abs(u - cam.cx) < 1e-6          # tilt up: the scene moves down
    right = cam.position + cam.R.T @ np.array([0.1, 0.0, 0.6])  # a point right of the image centre
    assert rot([0, 0, 5]).project(right)[0][1] < cam.cy - 1    # roll clockwise: the scene turns anticlockwise
    assert rot([0, 0, 0]).describe()["mount_rotation_deg"] == [0, 0, 0]


def test_variant_and_scenario_camera_block_in_simulation(tmp_path):
    cfgs = load_configs()
    s = Simulation({**SC, "camera": {"mount_rotation_deg": [1.0, 0.0, 0.0]}}, CAM, cfgs=cfgs, variant=2)
    var = sample_variant(2)
    assert s.scenario["name"] == "t_v2"
    assert np.allclose(s.camera.mount_rotation, var["camera"]["mount_rotation_deg"])   # variant beats scenario
    assert np.allclose(s.cabin.eye_point, var["cabin"]["driver"]["eye_point_offset"])
    assert cfgs == load_configs()                                                       # caller's configs untouched
    s.run(); s.write(tmp_path, fmt="pipeline")
    desc = json.loads((tmp_path / f"t_v2__{CAM}__camera.json").read_text())
    assert desc["variant"]["variant"] == 2 and desc["driver"]["head_scale"] == s.cabin.head_scale
    plain = Simulation({**SC, "camera": {"mount_rotation_deg": [1.0, 0.0, 0.0]}}, CAM)
    assert plain.scenario["name"] == "t" and np.allclose(plain.camera.mount_rotation, [1, 0, 0])


def test_batch_command(tmp_path):
    cli.main(["batch", "--gen", "1", "--duration", "1", "--camera", CAM, "--variants", "2", "--out", str(tmp_path),
              "--format", "dmsbin", "--fps", "10"])
    recs = json.loads((tmp_path / "batch_summary.json").read_text())
    assert [r["variant"] for r in recs] == [None, 0, 1]
    assert (tmp_path / "scenarios" / "random_1s_seed0.yaml").exists()
    assert len(list((tmp_path / "data").glob("*.dms.bin"))) == 3
    assert "Driver and mount per run" in (tmp_path / "batch_summary.md").read_text()


def test_demo_and_compare_with_variant(tmp_path):
    sc = tmp_path / "tiny.yaml"
    sc.write_text("name: tiny\nsegments:\n  - {target: road_ahead, duration: 0.5}\n")
    cli.main(["demo", "--scenario", str(sc), "--cameras", CAM, "--variant", "1", "--fps", "10",
              "--out", str(tmp_path / "demo")])
    demo = tmp_path / "demo"
    assert (demo / "data" / f"tiny_v1__{CAM}.pipeline.jsonl").exists()
    assert (demo / "data" / f"tiny_v1__{CAM}.dms.bin").exists()
    card = demo / "data" / f"tiny_v1__{CAM}__card"
    assert (card / "card.md").exists() and json.loads((card / "card.json").read_text(encoding="utf-8"))["frames"] == 5
    assert (demo / f"tiny_v1__{CAM}__timeline.png").exists() and (demo / f"tiny_v1__{CAM}.gif").exists()
    assert f"data/tiny_v1__{CAM}__card/card.md" in (demo / "report.md").read_text(encoding="utf-8")
    cli.main(["compare", "--scenario", str(sc), "--variant", "1", "--out", str(tmp_path / "cmp")])
    assert (tmp_path / "cmp" / "compare__tiny_v1.md").exists()

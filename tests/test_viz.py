import copy
import os
import time
import numpy as np
import pytest
from dms_model_simulator import viz
from dms_model_simulator.camera import Camera
from dms_model_simulator.simulate import Simulation


@pytest.fixture(scope="module")
def sim():
    s = Simulation("mirror_check", "steering_column", seed=0)
    s.run()
    return s


@pytest.fixture(scope="module")
def all_cams(sim):
    return [Camera.from_config(n, sim.cfgs["cameras"], sim.cabin.nominal_head_center())
            for n in sim.cfgs["cameras"]["cameras"]]


def test_composite_size(sim, all_cams):
    r = sim.records[10]
    view = viz.draw_camera_view(r, sim.camera, sim.cabin, 0.5)
    assert view.size == (int(sim.camera.width * 0.5), int(sim.camera.height * 0.5))
    img = viz.composite(r, sim.camera, sim.cabin, all_cams, 0.5)
    bg = viz.cabin_background(sim.cabin, all_cams, sim.camera.name, view.height)
    assert img.size == (view.width + bg.image.width, view.height)
    # cabin panel keeps the 13:6 figure aspect
    assert abs(bg.image.width / bg.image.height - 13 / 6) < 0.02
    # the overlay changes pixels relative to the static background
    cab = np.asarray(img)[:, view.width:]
    assert np.any(cab != np.asarray(bg.image))


def test_cabin_transform_matches_matplotlib(sim, all_cams):
    bg = viz.cabin_background(sim.cabin, all_cams, sim.camera.name, 400)
    for a in bg.axes:
        x0, y0, x1, y1 = a["box"]
        hc = sim.cabin.nominal_head_center()
        u, v = bg.to_px(a, hc)
        assert x0 <= u <= x1 and y0 <= v <= y1


@pytest.mark.skipif(os.environ.get("DMS_SKIP_TIMING") == "1", reason="timing test disabled (slow CI)")
def test_composite_speed(sim, all_cams):
    viz.composite(sim.records[0], sim.camera, sim.cabin, all_cams, 0.5)   # warm the static cache
    t0 = time.perf_counter()
    for r in sim.records[1:21]:
        viz.composite(r, sim.camera, sim.cabin, all_cams, 0.5)
    assert time.perf_counter() - t0 < 4.0


def test_dropped_frame(sim):
    r = copy.deepcopy(sim.records[5])
    r["pred"] = {"dropped": True, "detections": [], "head_pose_camera": None, "gaze": None, "eye_openness": None}
    img = viz.draw_camera_view(r, sim.camera, sim.cabin, 0.5)
    assert img.size == (sim.camera.width // 2, sim.camera.height // 2)


def test_no_face_box_and_extra_fields(sim, all_cams):
    r = copy.deepcopy(sim.records[5])
    r["truth"]["detections"] = [d for d in r["truth"]["detections"] if d["class"] != "face"]
    r["truth"]["landmarks_px"] = {k: [float("nan"), float("nan")] for k in r["truth"]["landmarks_px"]}
    r["truth"]["gaze"]["origin_px"] = [float("nan")] * 2
    r["truth"]["gaze"]["tip_px"] = [float("nan")] * 2
    r["pred"]["detections"] = []
    img = viz.draw_camera_view(r, sim.camera, sim.cabin, 0.5)
    assert img.size == (sim.camera.width // 2, sim.camera.height // 2)
    # mouth openness and events (fields produced by other modules) are optional but drawn when present
    r2 = copy.deepcopy(sim.records[5])
    r2["truth"]["mouth_openness"] = 0.9
    r2["truth"]["events"] = {"eyes_off_road": True, "eyes_closed": False, "yawning": True, "off_road_s": 1.5}
    assert viz._active_events(r2["truth"]) == ["eyes_off_road", "yawning"]     # durations are not labels
    a = np.asarray(viz.draw_camera_view(r2, sim.camera, sim.cabin, 0.5))
    b = np.asarray(viz.draw_camera_view(sim.records[5], sim.camera, sim.cabin, 0.5))
    assert np.any(a != b)
    viz.composite(r2, sim.camera, sim.cabin, all_cams, 0.5)


def test_plot_cabin_standalone(sim, all_cams):
    fig = viz.plot_cabin(sim.cabin, all_cams, sim.records[3], highlight=sim.camera.name)
    assert len(fig.axes) == 2
    viz.plt.close(fig)


def test_pred_gaze_drawn_with_lens_distortion():
    """With exact gaze output the predicted gaze tip lands on the GT tip, also through a distorted lens."""
    from dms_model_simulator.config import load_configs
    cfgs = load_configs()
    cfgs["cameras"]["cameras"]["steering_column"].update({"distortion": [-0.35, 0.0], "look_at_offset": [0.0, 0.2, 0.12]})
    ov = {"pipeline": {"drop_frame_prob": 0},
          "gaze": {"base_std_deg": 0, "per_deg_gaze_to_camera": 0, "per_deg_rel_head_yaw": 0, "per_px_inv_eye_width": 0},
          "detection": {"face": {"p_max": 1, "min_width_px": 0}, "eye": {"p_max": 1, "min_width_px": 0},
                        "false_positives": {"rate_per_frame": 0}}}
    s = Simulation("mirror_check", "steering_column", seed=0, cfgs=cfgs, emulator_overrides=ov)
    s.frames = s.frames[:5]; s.run()
    r = s.records[2]
    assert r["pred"]["gaze"]["valid"]
    u, v = r["truth"]["gaze"]["tip_px"]
    img = np.asarray(viz.draw_camera_view(r, s.camera, s.cabin, 1.0))
    assert tuple(img[int(round(v)), int(round(u))]) == viz.PRED_GAZE_COLOR      # drawn over the GT tip


def test_error_rows_and_plot_timeline(tmp_path):
    from dms_model_simulator.schema import to_pipeline, to_truth
    sc = {"name": "t", "segments": [{"target": "road_ahead", "duration": 1.0}, {"target": "phone_lap", "duration": 1.0}]}
    ov = {"pipeline": {"drop_frame_prob": 0.2},
          "eye_state_flip": {"enabled": True, "mean_frames": 5, "open_to_closed": {"p_base": 0.5}}}
    s = Simulation(sc, "steering_column", seed=0, emulator_overrides=ov); s.run()
    rows = viz._error_rows(s.records)
    assert rows["dropped"] == [r["pred"]["dropped"] for r in s.records]
    assert rows["eye class flip"] == ["class_flip" in r["pred"] for r in s.records] and any(rows["eye class flip"])
    assert "mouth class flip" not in rows                                # rows that never occur are left out
    fig = viz.plot_timeline(s.records, tmp_path / "tl.png", title="t / steering_column")
    assert (tmp_path / "tl.png").stat().st_size > 0 and fig.axes
    viz.plt.close(fig)
    # truth-only records (no error rows) with the pipeline records for the model openness
    truth = [to_truth(r, s.camera) for r in s.records]
    pipe = [to_pipeline(r, s.camera) for r in s.records]
    assert viz._error_rows(truth) == {}
    fig = viz.plot_timeline(truth, tmp_path / "tl2.png", pipeline=pipe)
    assert (tmp_path / "tl2.png").exists()
    viz.plt.close(fig)

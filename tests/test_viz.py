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
    r["truth"]["face_dir"]["origin_px"] = [float("nan")] * 2       # a face direction off the image is not drawn
    r["truth"]["face_dir"]["tip_px"] = [float("nan")] * 2
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


def test_face_direction_arrows_drawn(sim, all_cams):
    # the frame where head and eyes point furthest apart (else the gaze arrow covers the face arrow)
    from dms_model_simulator import geometry as G
    r = max(sim.records, key=lambda x: G.angle_between(x["truth"]["face_dir"]["dir_vehicle"], x["truth"]["gaze"]["dir_vehicle"]))
    assert r["pred"]["face_dir"]["valid"]
    bare = copy.deepcopy(r)
    del bare["truth"]["face_dir"]; bare["pred"]["face_dir"] = None     # records without a face direction still draw
    a = np.asarray(viz.draw_camera_view(r, sim.camera, sim.cabin, 0.5))
    b = np.asarray(viz.draw_camera_view(bare, sim.camera, sim.cabin, 0.5))
    assert np.any(a != b)
    n_face = lambda img, col: int(np.all(img == col, axis=-1).sum())     # the legend has the colour in both
    assert n_face(a, viz.FACE_COLOR) > n_face(b, viz.FACE_COLOR)
    bg = viz.cabin_background(sim.cabin, all_cams, sim.camera.name, 300)
    ca, cb = np.asarray(bg.render(r)), np.asarray(bg.render(bare))
    assert n_face(ca, viz.CABIN_GT_FACE) > n_face(cb, viz.CABIN_GT_FACE)
    viz.plt.close(viz.plot_cabin(sim.cabin, all_cams, bare, highlight=sim.camera.name))


def test_face_direction_drawn_on_top_of_an_aligned_gaze(sim, all_cams):
    """Head and eyes pointing the same way: the thinner face arrows lie on the gaze arrows and stay visible."""
    from dms_model_simulator import geometry as G
    r = copy.deepcopy(min(sim.records, key=lambda x: G.angle_between(x["truth"]["face_dir"]["dir_vehicle"],
                                                                     x["truth"]["gaze"]["dir_vehicle"])))
    r["truth"]["face_dir"] = {**r["truth"]["face_dir"], "dir_vehicle": r["truth"]["gaze"]["dir_vehicle"],
                              "origin_px": r["truth"]["gaze"]["origin_px"], "tip_px": r["truth"]["gaze"]["tip_px"]}
    u, v = np.round(0.5 * np.add(r["truth"]["gaze"]["origin_px"], r["truth"]["gaze"]["tip_px"]) * 0.5).astype(int)
    img = np.asarray(viz.draw_camera_view(r, sim.camera, sim.cabin, 0.5, show_pred=False))
    assert tuple(img[v, u]) == viz.FACE_COLOR                        # midpoint of the GT gaze line: face on top
    near = img[v - 3:v + 4, u - 3:u + 4].reshape(-1, 3)
    assert any(tuple(p) == viz.GAZE_COLOR for p in near)             # the wider gaze line still shows beside it
    bg = viz.cabin_background(sim.cabin, all_cams, sim.camera.name, 300)
    bare = copy.deepcopy(r); del bare["truth"]["face_dir"]; bare["pred"]["face_dir"] = None
    ca, cb = np.asarray(bg.render(r)), np.asarray(bg.render(bare))
    n = lambda img, col: int(np.all(img == col, axis=-1).sum())
    assert n(ca, viz.CABIN_GT_FACE) > n(cb, viz.CABIN_GT_FACE)       # visible over the GT gaze arrow
    assert n(ca, viz.CABIN_GT_GAZE) > 0


def test_legend_wraps_in_a_narrow_view(sim):
    """At a small scale the six legend entries do not fit one row: the last (PRED face) wraps onto a row of its own
    instead of being cut off at the image edge."""
    r = copy.deepcopy(sim.records[10])
    del r["truth"]["face_dir"]; r["pred"]["face_dir"] = None        # PRED_FACE_COLOR then only in the legend
    for scale in (0.5, 0.15):
        img = np.asarray(viz.draw_camera_view(r, sim.camera, sim.cabin, scale))
        rows = np.nonzero(np.all(img == viz.PRED_FACE_COLOR, axis=-1).any(axis=1))[0]
        gaze_rows = np.nonzero(np.all(img == viz.GAZE_COLOR, axis=-1)[-40:].any(axis=1))[0]
        assert len(rows) > 0                                        # the PRED face legend entry is drawn
        if scale == 0.15:
            assert rows.min() > img.shape[0] - 40 + gaze_rows.max()   # on a row below the GT gaze entry
        else:                                                       # wide enough: one row, as before
            assert rows.max() <= img.shape[0] - 40 + gaze_rows.max() and rows.min() >= img.shape[0] - 40 + gaze_rows.min()


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
    r = copy.deepcopy(s.records[2])
    assert r["pred"]["gaze"]["valid"]
    # the face-direction arrows are drawn on top of the gaze; without them the GT gaze tip shows the PRED gaze
    del r["truth"]["face_dir"]; r["pred"]["face_dir"] = None
    u, v = r["truth"]["gaze"]["tip_px"]
    img = np.asarray(viz.draw_camera_view(r, s.camera, s.cabin, 1.0))
    assert tuple(img[int(round(v)), int(round(u))]) == viz.PRED_GAZE_COLOR      # drawn over the GT tip


def test_pred_face_drawn_with_lens_distortion():
    """With exact face-direction output the predicted face tip lands on the GT face tip (truth.py: the same 0.15 m
    ray, projected through the distorted lens), and the arrow ends there. The frame is the one where the
    distortion moves the tip most, so a ray of another length or a projection without distortion misses."""
    from dms_model_simulator.config import load_configs
    cfgs = load_configs()
    cfgs["cameras"]["cameras"]["steering_column"].update({"distortion": [-0.35, 0.0], "look_at_offset": [0.0, 0.2, 0.12]})
    ov = {"pipeline": {"drop_frame_prob": 0},
          "face_dir": {"base_std_deg": 0, "per_deg_rel_yaw": 0, "per_px_inv_face_width": 0, "bias_deg": [0, 0]},
          "detection": {"face": {"p_max": 1, "min_width_px": 0}, "false_positives": {"rate_per_frame": 0}}}
    s = Simulation("mirror_check", "steering_column", seed=0, cfgs=cfgs, emulator_overrides=ov); s.run()
    flat = copy.copy(s.camera); flat.k1 = flat.k2 = 0.0            # the same camera without distortion

    def shift(r):                                  # how far the distortion moves the face tip (px)
        t = r["truth"]["face_dir"]
        tip = np.array(t["origin_vehicle"]) + 0.15 * np.array(t["dir_vehicle"])
        return float(np.linalg.norm(np.subtract(t["tip_px"], flat.project(tip)[0])))

    H = s.camera.height
    ok = [r for r in s.records if r["pred"]["face_dir"] and 0.3 * H < r["truth"]["face_dir"]["tip_px"][1] < 0.7 * H]
    r = copy.deepcopy(max(ok, key=shift))
    assert shift(r) > 6
    assert r["pred"]["face_dir"]["angular_error_deg"] < 1e-9
    o, tip = np.array(r["truth"]["face_dir"]["origin_px"]), np.array(r["truth"]["face_dir"]["tip_px"])
    del r["truth"]["face_dir"]                     # only the PRED face arrow is drawn (it is on top of the rest)
    img = np.asarray(viz.draw_camera_view(r, s.camera, s.cabin, 1.0))
    px = lambda p: tuple(img[int(round(p[1])), int(round(p[0]))])
    assert px(tip) == viz.PRED_FACE_COLOR                            # tip on the GT tip
    beyond = tip + 6 * (tip - o) / np.linalg.norm(tip - o)
    assert px(beyond) != viz.PRED_FACE_COLOR                         # and the arrow ends there


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


def test_view_indices_play_back_in_real_time():
    assert viz.view_indices(10, 25, 25) == list(range(10))           # same rate: every record once
    assert viz.view_indices(30, 30, 25) == [int(k * 30 / 25) for k in range(25)]   # 1 s at 30 fps -> 25 frames
    assert viz.view_indices(4, 20, 40) == [0, 0, 1, 1, 2, 2, 3, 3]    # higher view rate: records repeat
    with pytest.raises(ValueError):
        viz.view_indices(4, 20, 0)


def test_render_uses_view_fps(tmp_path, capsys):
    from dms_model_simulator import cli
    cli.main(["render", "--scenario", "mirror_check", "--camera", "steering_column", "--fps", "10", "--view-fps", "5",
              "--no-cabin", "--scale", "0.15", "--out", str(tmp_path)])
    assert "@ 5.0 fps" in capsys.readouterr().out
    gif = next(tmp_path.rglob("preview.gif"))
    from PIL import Image
    im = Image.open(gif)
    assert im.info["duration"] == 200 and im.n_frames == 57           # 11.5 s scenario at 5 fps

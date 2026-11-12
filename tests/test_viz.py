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
    assert n_face(ca, viz.FACE_COLOR) > n_face(cb, viz.FACE_COLOR)
    viz.plt.close(viz.plot_cabin(sim.cabin, all_cams, bare, highlight=sim.camera.name))


def test_face_wedge_and_eye_line_both_visible_when_collinear(sim, all_cams):
    """Head and eyes pointing the same way: the eye line lies on the wider face wedge and both stay visible, in the
    camera view and in both cabin views."""
    from dms_model_simulator import geometry as G
    r = copy.deepcopy(min(sim.records, key=lambda x: G.angle_between(x["truth"]["face_dir"]["dir_vehicle"],
                                                                     x["truth"]["gaze"]["dir_vehicle"])))
    r["truth"]["face_dir"] = {**r["truth"]["face_dir"], "dir_vehicle": r["truth"]["gaze"]["dir_vehicle"],
                              "origin_px": r["truth"]["gaze"]["origin_px"], "tip_px": r["truth"]["gaze"]["tip_px"]}
    u, v = np.round(0.5 * np.add(r["truth"]["gaze"]["origin_px"], r["truth"]["gaze"]["tip_px"]) * 0.5).astype(int)
    length = np.linalg.norm(np.subtract(r["truth"]["gaze"]["tip_px"], r["truth"]["gaze"]["origin_px"])) * 0.5
    assert length > viz.TOWARD_CAM_PX                                # an arrow, not a toward-camera mark
    img = np.asarray(viz.draw_camera_view(r, sim.camera, sim.cabin, 0.5, show_pred=False))
    assert tuple(img[v, u]) == viz.EYES_COLOR                        # midpoint of the GT eye line: eyes on top
    near = img[v - 4:v + 5, u - 4:u + 5].reshape(-1, 3)
    assert any(tuple(p) == viz.FACE_COLOR for p in near)             # the wider face wedge shows beside it
    bg = viz.cabin_background(sim.cabin, all_cams, sim.camera.name, 300)
    bare = copy.deepcopy(r); del bare["truth"]["face_dir"]; bare["pred"]["face_dir"] = None
    ca, cb = np.asarray(bg.render(r)), np.asarray(bg.render(bare))
    n = lambda img, col: int(np.all(img == col, axis=-1).sum())
    for a in bg.axes:                                                # in each cabin view, the wedge beside the line
        x0, y0, x1, y1 = a["box"]
        sa, sb = ca[y0:y1, x0:x1], cb[y0:y1, x0:x1]
        assert n(sa, viz.FACE_COLOR) > n(sb, viz.FACE_COLOR) + 20
        assert n(sa, viz.EYES_COLOR) > 20


def test_legend_wraps_in_a_narrow_view(sim):
    """At a small scale the six legend entries do not fit one row: they wrap onto more rows instead of being cut off
    at the image edge; the last entry (model face) is drawn on the last row."""
    from PIL import Image, ImageDraw
    r = copy.deepcopy(sim.records[10])
    del r["truth"]["face_dir"]; r["pred"]["face_dir"] = None        # FACE_COLOR then only in the legend
    for scale in (0.5, 0.15):
        img = np.asarray(viz.draw_camera_view(r, sim.camera, sim.cabin, scale))
        H, W = img.shape[:2]
        fs = max(9, round(W / 58))                                  # draw_camera_view's base font size
        d = ImageDraw.Draw(Image.new("RGB", (W, H)))
        rows = viz.camera_legend_rows(d, W, fs)
        f_small = viz._font(max(8, fs - 2))
        for row in rows:                                            # nothing runs off the right edge
            x, (_k, _m, label) = row[-1]
            assert len(row) == 1 or x + viz._legend_swatch_w(fs) + 4 + d.textlength(label, font=f_small) <= W - 6
        assert rows[-1][-1][1][2] == "face: model"
        y_last = H - (fs + 4) - 4                                   # top of the last legend row
        face_rows = np.nonzero(np.all(img == viz.FACE_COLOR, axis=-1).any(axis=1))[0]
        assert np.any(face_rows >= y_last - 2)                      # the model face swatch is drawn there
        assert len(rows) > 1 if scale == 0.15 else len(rows) == 1   # wide enough: one row, as before


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
    # truth and model share the eyes colour: only the model eye line is drawn (truth gaze off the image, no faces)
    del r["truth"]["face_dir"]; r["pred"]["face_dir"] = None
    u, v = r["truth"]["gaze"]["tip_px"]
    r["truth"]["gaze"]["origin_px"] = r["truth"]["gaze"]["tip_px"] = [float("nan")] * 2
    img = np.asarray(viz.draw_camera_view(r, s.camera, s.cabin, 1.0))
    assert tuple(img[int(round(v)), int(round(u))]) == viz.EYES_COLOR           # model tip on the GT tip


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
    assert px(tip) == viz.FACE_COLOR                                 # tip on the GT tip
    beyond = tip + 6 * (tip - o) / np.linalg.norm(tip - o)
    assert px(beyond) != viz.FACE_COLOR                              # and the wedge ends there


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


# ---- one visual coding, toward-camera marks, GIF palette, story view -------------------------------------------
@pytest.fixture(scope="module")
def fve():
    s = Simulation("face_vs_eyes", "steering_column", seed=0)
    s.run()
    return s


def _count(img, col):
    return int(np.all(np.asarray(img) == col, axis=-1).sum())


def test_toward_camera_marker(sim):
    """A direction that projects to a stub in the camera view (pointing at the camera) is drawn as a ring with a dot
    at its origin, not as a short arrow."""
    r = copy.deepcopy(sim.records[10])
    del r["truth"]["face_dir"]; r["pred"]["face_dir"] = None
    g = r["truth"]["gaze"]
    g["tip_px"] = list(np.add(g["origin_px"], [2.0, 1.0]))          # 1 px long at scale 0.5
    g["dir_camera"] = [0.0, 0.0, -1.0]                               # towards the camera
    img = np.asarray(viz.draw_camera_view(r, sim.camera, sim.cabin, 0.5, show_pred=False))
    u, v = np.round(np.multiply(g["origin_px"], 0.5)).astype(int)
    assert tuple(img[v, u]) == viz.EYES_COLOR                        # the dot
    ring = np.all(img[v - 12:v + 13, u - 12:u + 13] == viz.EYES_COLOR, axis=-1)
    assert ring.sum() > 30 and not ring[12, 12 + 4:12 + 6].all()     # a ring around it, a gap between ring and dot
    g["dir_camera"] = [0.0, 0.0, 1.0]                                # away: a cross, no dot in the centre ring gap
    img2 = np.asarray(viz.draw_camera_view(r, sim.camera, sim.cabin, 0.5, show_pred=False))
    assert np.any(img2[v - 12:v + 13, u - 12:u + 13] != img[v - 12:v + 13, u - 12:u + 13])


def test_cabin_out_of_plane_marker(sim, all_cams):
    """Eyes straight down: in the top-down view the sight line becomes an into-the-picture mark at the head, with no
    long arrow."""
    r = copy.deepcopy(sim.records[10])
    r["truth"]["gaze"]["dir_vehicle"] = [0.0, 0.0, -1.0]
    r["pred"] = {"dropped": True}
    bg = viz.cabin_background(sim.cabin, all_cams, sim.camera.name, 400)
    img = np.asarray(bg.render(r))
    a = bg.axes[0]
    hc = np.array(r["truth"]["head_center_vehicle"])
    hx, hy = np.round(bg.to_px(a, hc)).astype(int)
    near = np.all(img[hy - 15:hy + 16, hx - 15:hx + 16] == viz.EYES_COLOR, axis=-1).sum()
    x0, y0, x1, y1 = a["box"]
    box = np.all(img[y0:y1, x0:x1] == viz.EYES_COLOR, axis=-1).sum()
    legend = np.all(img[y0:y1, x0:x0 + 60] == viz.EYES_COLOR, axis=-1).sum()
    assert near > 10 and box - legend - near == 0                    # only the mark (and the legend swatches)


def test_gif_palette_keeps_legend_colours(sim, tmp_path):
    """A frame with no orange outside the legend keeps the legend's exact colours in the written GIF."""
    from PIL import Image
    r = copy.deepcopy(sim.records[10])
    bare = copy.deepcopy(r)
    del bare["truth"]["face_dir"]; bare["pred"] = {"dropped": True}
    bare["truth"]["gaze"]["origin_px"] = bare["truth"]["gaze"]["tip_px"] = [float("nan")] * 2
    frames = [viz.draw_camera_view(x, sim.camera, sim.cabin, 0.5) for x in (r, bare, r)]
    path = tmp_path / "p.gif"
    viz.save_gif(frames, path, 10)
    im = Image.open(path)
    for i, fr in enumerate(frames):
        im.seek(i)
        got, want = np.asarray(im.convert("RGB")), np.asarray(fr)
        for col in (viz.EYES_COLOR, viz.FACE_COLOR, viz.GT_BOX_COLOR, viz.PRED_BOX_COLOR, viz.HUD_TRUTH):
            m = np.all(want == col, axis=-1)
            assert m.any() and np.all(np.all(got[m] == col, axis=-1)), (i, col)
    viz.save_gif([f.quantize(method=Image.Quantize.FASTOCTREE) for f in frames], tmp_path / "q.gif", 10)  # P frames


def test_story_frame(fve):
    r = fve.records[len(fve.records) // 2]
    img = viz.story_frame(r, fve.cabin, caption="a caption")
    assert img.size == viz.STORY_SIZE
    assert _count(img, viz.FACE_COLOR) > 200 and _count(img, viz.EYES_COLOR) > 200
    line1, line2 = viz.story_lines(r)
    assert line1.startswith("face ") and "eyes relative to face" in line1 and line2.startswith("model error: face ")
    assert viz.story_lines(r, show_model=False)[1] is None
    # degenerate records: dropped model output, no model face, no truth face
    for edit in (lambda x: x.__setitem__("pred", {"dropped": True}),
                 lambda x: x["pred"].__setitem__("face_dir", None),
                 lambda x: x["truth"].pop("face_dir")):
        x = copy.deepcopy(r); edit(x)
        viz.story_frame(x, fve.cabin)
    x = copy.deepcopy(r); x["pred"] = {"dropped": True}
    assert viz.story_lines(x)[1] == "model: no output on this frame"
    assert viz.dir_words(31, 10) == "31° left, 10° up" and viz.dir_words(-0.2, 1) == "ahead"
    assert viz.dir_words(-31, -1, short=True) == "31° R"


def test_story_moments_and_scenario(fve):
    """face_vs_eyes: captions from the segments, the eyes-only mirror glance reaches the mirror, no blinks, and the
    key moments are the segments where face and eyes differ most."""
    segs = fve.behavior.segments
    caps = viz.segment_captions(segs, fve.records)
    assert caps[1] == "Eyes only: glance at the rear-view mirror"
    mid = (segs[1]["start_frame"] + segs[1]["end_frame"]) // 2
    assert fve.records[mid]["truth"]["gaze"]["zone"] == "rearview_mirror"
    assert not any(r["truth"]["blink"] for r in fve.records)
    m = viz.story_moments(fve.records, segs, k=4)
    assert len(m) == 4 and [i for i, _c in m] == sorted(i for i, _c in m)
    assert m[0][1] == caps[1] and all(c in caps for _i, c in m)
    assert viz.segment_of_frames(segs, len(fve.records))[mid] == 1
    tg = viz.story_targets(fve.cabin, segs, fve.records)
    assert tg[:2] == ["road_ahead", "rearview_mirror"] and "left_mirror" not in tg


def test_render_story_cli(tmp_path, capsys):
    from dms_model_simulator import cli
    from PIL import Image
    cli.main(["render", "--scenario", "face_vs_eyes", "--story", "--view-fps", "4", "--scale", "0.2",
              "--out", str(tmp_path)])
    assert "story_sheet.png (4 moments)" in capsys.readouterr().out
    d = next(tmp_path.glob("face_vs_eyes__*"))
    gif = Image.open(d / "story.gif")
    assert gif.info["duration"] == 250 and gif.size == (384, 288)
    sheet = Image.open(d / "story_sheet.png")
    assert sheet.width > 2 * 384 and sheet.height > 2 * 288
    with pytest.raises(SystemExit):
        cli.main(["render", "--scenario", "face_vs_eyes", "--story", "--camera", "steering_column", "dash_right",
                  "--out", str(tmp_path)])

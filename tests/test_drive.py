"""Right-hand drive: a run with drive="right" is the exact mirror image of the left-hand-drive run.

The truth of every frame is compared field by field with the mirrored left-hand-drive truth: pixels u -> W - u,
vehicle / camera y -> -y, yaw and roll change sign, the driver's left and right eye (and every left_ / right_
landmark) swap, left / right gaze targets swap. Only the truth is compared: the model noise is not mirrored.
"""
import json
from pathlib import Path
import numpy as np
import pytest
from dms_model_simulator import drive as D
from dms_model_simulator.catalog import build_scenario, level_specs, load_entries
from dms_model_simulator.config import load_configs, load_scenario
from dms_model_simulator.simulate import Simulation

CAMERAS = ["steering_column", "cluster", "rearview_mirror", "a_pillar_left", "center_stack", "overhead_console",
           "dash_right"]
FPS = 5.0          # a few frames per second are enough: the mirror holds frame by frame


def _swap_lr(name: str) -> str:
    """Anatomical names: left_eye <-> right_eye, left_eye_outer <-> right_eye_outer, mouth_left <-> mouth_right."""
    return D.mirror_target_name(name)         # the same left / right token swap


def _px(p, W):
    return [W - p[0], p[1]] if p is not None and np.all(np.isfinite(p)) else p


def _box(b, W):
    return None if b is None else [W - b[2], b[1], W - b[0], b[3]]


def _vy(v):                       # vehicle frame vector: y -> -y
    return [v[0], -v[1], v[2]]


def _cx(v):                       # camera frame (OpenCV, x right) vector: x -> -x
    return [-v[0], v[1], v[2]]


def _ang(d):                      # {yaw, pitch, roll}
    return {"yaw": -d["yaw"], "pitch": d["pitch"], "roll": -d["roll"]}


def _ray(g, W):
    out = dict(g, origin_vehicle=_vy(g["origin_vehicle"]), dir_vehicle=_vy(g["dir_vehicle"]),
               yaw_vehicle=-g["yaw_vehicle"], dir_camera=_cx(g["dir_camera"]), yaw_camera=-g["yaw_camera"],
               zone=D.mirror_target_name(g["zone"]), origin_px=_px(g["origin_px"], W), tip_px=_px(g["tip_px"], W))
    if "eye_in_head_yaw" in g:
        out["eye_in_head_yaw"] = -g["eye_in_head_yaw"]
    if "target" in g:
        out["target"] = D.mirror_target_name(g["target"])
    return out


def mirrored_truth(t: dict, W: int) -> dict:
    """The truth a right-hand-drive run should give, from the left-hand-drive truth t."""
    dets = []
    for d in t["detections"]:
        dets.append(dict(d, **{"class": _swap_lr(d["class"]), "bbox": _box(d["bbox"], W)}))
    return dict(
        t,
        head_pose_vehicle=_ang(t["head_pose_vehicle"]), head_pose_camera=_ang(t["head_pose_camera"]),
        head_center_vehicle=_vy(t["head_center_vehicle"]), head_center_camera=_cx(t["head_center_camera"]),
        gaze=_ray(t["gaze"], W), face_dir=_ray(t["face_dir"], W),
        eye_openness={"left": t["eye_openness"]["right"], "right": t["eye_openness"]["left"]},
        eye_occluded={_swap_lr(k): v for k, v in t["eye_occluded"].items()},
        phone=D.mirror_target_name(t["phone"]),
        held_bbox=_box(t["held_bbox"], W), seatbelt_strap_bbox=_box(t["seatbelt_strap_bbox"], W),
        detections=dets,
        landmarks_px={_swap_lr(k): _px(v, W) for k, v in t["landmarks_px"].items()},
        landmarks_visible={_swap_lr(k): v for k, v in t["landmarks_visible"].items()},
        occluded_by={_swap_lr(k): v for k, v in t["occluded_by"].items()},
    )


def assert_same(a, b, path="", tol=1e-6):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), f"{path}: keys {sorted(a)} vs {sorted(b)}"
        for k in a:
            assert_same(a[k], b[k], f"{path}.{k}", tol)
    elif isinstance(a, (list, tuple)):
        assert isinstance(b, (list, tuple)) and len(a) == len(b), f"{path}: {a} vs {b}"
        for i, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{path}[{i}]", tol)
    elif isinstance(a, (float, int, np.floating, np.integer)) and not isinstance(a, (bool, np.bool_)):
        if np.isnan(a):
            assert np.isnan(b), f"{path}: nan vs {b}"
        else:
            assert abs(float(a) - float(b)) <= tol, f"{path}: {a} vs {b}"
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


def _by_class(dets):
    return sorted(dets, key=lambda d: d["class"])


def check_mirror(scenario, camera: str, variant=None, seed: int = 0):
    lhd = Simulation(scenario, camera, seed=seed, fps=FPS, variant=variant)
    rhd = Simulation(scenario, camera, seed=seed, fps=FPS, variant=variant, drive="right")
    W = lhd.camera.width
    for i, (tl, tr) in enumerate(zip(lhd.truths(), rhd.truths())):
        exp = mirrored_truth(tl, W)
        exp["detections"], got = _by_class(exp["detections"]), dict(tr, detections=_by_class(tr["detections"]))
        assert_same(exp, got, f"{scenario if isinstance(scenario, str) else scenario['name']}/{camera}/frame {i}")
    assert len(lhd.truths()) == len(rhd.truths())


def _entry_scenario(entry_id: str, level: str = "large") -> dict:
    from dms_model_simulator.catalog import expand_params
    e = next(e for e in load_entries() if e["id"] == entry_id)
    params = expand_params(e, "boundary", np.random.default_rng(0), 1, levels=[level])[0]
    return build_scenario(e, params, f"{entry_id}_{level}")


@pytest.mark.parametrize("camera", CAMERAS)
def test_every_camera_mirrors(camera):
    check_mirror("hand_to_face", camera)


@pytest.mark.parametrize("scenario", ["mirror_check", "phone_distraction", "smoking", "drowsy"])
def test_scenarios_mirror(scenario):
    for camera in ("steering_column", "dash_right", "overhead_console"):
        check_mirror(scenario, camera)


# entries with hands across the face / chest / wheel, scripted head poses, extra targets and objects at the lips
ENTRIES = ["NEG-SBT-HAND-ACROSS-CHEST", "NEG-ALL-HAND-OVER-HAND", "NEG-DRW-HAND-AT-EYE", "POS-DIS-NCAP-OWL-DOWN",
           "POS-DIS-NCAP-OWL-WINDOW", "NEG-ALL-HABITS-COMBO"]


@pytest.mark.parametrize("entry", ENTRIES)
def test_catalog_entries_mirror(entry):
    sc = _entry_scenario(entry)
    for camera in ("steering_column", "dash_right", "overhead_console", "rearview_mirror"):
        check_mirror(sc, camera)


@pytest.mark.parametrize("camera", ["dash_right", "a_pillar_left"])
def test_variant_mirror(camera):
    """Driver offset and camera mounting error mirror too (variant N in right-hand drive = mirrored variant N)."""
    check_mirror("phone_distraction", camera, variant=3)


def test_left_is_default_and_unchanged(tmp_path):
    a = Simulation("mirror_check", "dash_right", fps=FPS)
    b = Simulation("mirror_check", "dash_right", fps=FPS, drive="left")
    assert a.scenario["name"] == b.scenario["name"] == "mirror_check"
    assert json.dumps(a.run(), default=str) == json.dumps(b.run(), default=str)
    a.write(tmp_path)
    assert "drive" not in json.loads((tmp_path / "mirror_check__dash_right__camera.json").read_text())


def test_right_names_and_camera_json(tmp_path):
    sim = Simulation("mirror_check", "dash_right", fps=FPS, drive="right")
    assert sim.scenario["name"] == "mirror_check_rhd"
    assert sim.camera.position[1] > 0                      # beside the wheel, on the passenger side (left)
    assert sim.cabin.targets["right_mirror"][1] < 0 and sim.cabin.targets["passenger"][1] > 0
    sim.run()
    paths = sim.write(tmp_path)
    assert all("mirror_check_rhd__dash_right" in p.name for p in paths)
    desc = json.loads((tmp_path / "mirror_check_rhd__dash_right__camera.json").read_text())
    assert desc["drive"] == "right"
    # the driver-side (here: right) mirror is looked at under its physical name
    zones = {r["truth"]["gaze"]["zone"] for r in sim.records}
    assert {"right_mirror", "left_mirror"} <= zones


def test_bad_drive():
    with pytest.raises(ValueError, match="drive"):
        Simulation("mirror_check", "dash_right", fps=FPS, drive="rhd")


def test_mirror_is_an_involution():
    cfgs = load_configs()
    for f, cfg in ((D.mirror_cabin, cfgs["cabin"]), (D.mirror_cameras, cfgs["cameras"]),
                   (D.mirror_behavior, cfgs["behavior"]), (D.mirror_scenario, load_scenario("hand_to_face"))):
        assert_same(json.loads(json.dumps(f(f(cfg)))), json.loads(json.dumps(cfg)), f.__name__)


# ---- the car body: built as configs/body.yaml says, then mirrored ----------------------------------------------
def _flip_y(a):
    a = np.array(a, float)
    a[..., 1] *= -1.0
    return a


def _body_rays(occ, cam_pos, step: int = 7):
    """Part names (or None) the body gives for rays from cam_pos to points 2 cm behind every step-th triangle's
    centre, and those points."""
    from dms_model_simulator.world import Cabin
    pts = []
    for t in occ["tris"][::step]:
        c = t.mean(0)
        d = c - cam_pos
        pts.append(c + 0.02 * d / np.linalg.norm(d))
    return [Cabin.ray_blocked(cam_pos, p, [occ]) for p in pts], pts


@pytest.mark.parametrize("variant", [None, 3])
def test_body_occluder_mirrors(variant):
    """The body of a right-hand-drive cabin is the left-hand-drive body mirrored triangle by triangle (front sides
    still the front), with the same part names; a driver variant moves torso and seat the mirrored way. A ray to a
    mirrored point meets the part of the same name."""
    from dms_model_simulator.cli import _cameras
    assert "drive" not in load_configs()["cabin"] and D.mirror_cabin(load_configs()["cabin"])["drive"] == "right"
    lc = Simulation("mirror_check", "dash_right", fps=FPS, variant=variant).cabin
    rc = Simulation("mirror_check", "dash_right", fps=FPS, variant=variant, drive="right").cabin
    assert (lc.drive, rc.drive) == ("left", "right")
    if variant is not None:
        assert np.any(lc.eye_point != 0)                 # the variant moves the torso and seat: checked too
    lo, ro = lc.body_occluder, rc.body_occluder
    assert list(ro["part"]) == list(lo["part"]) and list(ro["two_sided"]) == list(lo["two_sided"])
    m = _flip_y(lo["tris"])
    np.testing.assert_array_equal(ro["tris"][:, 0], m[:, 0])
    np.testing.assert_array_equal(ro["tris"][:, 1], m[:, 2])         # two vertices swap: the front stays the front
    np.testing.assert_array_equal(ro["tris"][:, 2], m[:, 1])
    _, lcams = _cameras(load_configs(), None)
    _, rcams = _cameras(load_configs(), None, "right")
    hits = 0
    for cl, cr in zip(lcams, rcams):
        assert cl.name == cr.name
        names_l, pts = _body_rays(lo, cl.position)
        names_r = [type(rc).ray_blocked(cr.position, _flip_y(p), [ro]) for p in pts]
        assert names_r == names_l, cl.name
        hits += sum(n is not None for n in names_l)
    assert hits > 0


def test_body_parts_mirror():
    """body_parts (what dms-sim view3d draws) is the same mirrored geometry as the occluder."""
    from dms_model_simulator.cli import _cameras
    (lc, _), (rc, _) = _cameras(load_configs(), None), _cameras(load_configs(), None, "right")
    lp = D.body_parts(lc.body, lc.targets, lc.eye_point, lc.drive)
    rp = D.body_parts(rc.body, rc.targets, rc.eye_point, rc.drive)
    assert [p["name"] for p in rp] == [p["name"] for p in lp]
    for a, b in zip(lp, rp):
        assert {k: v for k, v in a.items() if k != "tris"} == {k: v for k, v in b.items() if k != "tris"}
        if len(a["tris"]):
            np.testing.assert_array_equal(np.array(b["tris"]), _flip_y(a["tris"])[:, [0, 2, 1]])


# ---- wiring: export, catalog, command line -------------------------------------------------------------
def test_export_scenario_drive(tmp_path):
    from dms_model_simulator.export import export_scenario
    sim, st, truth = export_scenario("mirror_check", "dash_right", "frame_yaml", tmp_path, fps=FPS, drive="right")
    assert sim.drive == "right" and truth.name == "mirror_check_rhd__dash_right.truth.jsonl"


def _one_entry():
    return {"id": "NEG-DIS-TEST", "number": 1, "title": "邊講話邊看左鏡", "title_en": "Talking at the left mirror",
            "summary": "看 left_mirror 0.8 s，dash_right 拍得到", "status": "ready", "behaviors": ["N01"], "params": {},
            "template": {"segments": [{"target": "road_ahead", "duration": 1.0},
                                      {"target": "left_mirror", "duration": 0.8},
                                      {"target": "road_ahead", "duration": 1.0}]},
            "variation": {"cameras": ["dash_right"], "emulator_seeds": [0]}, "fps": 10}


def test_catalog_expand_and_run_drive(tmp_path):
    from dms_model_simulator import catalog as C
    from dms_model_simulator.repro import compare_fingerprints
    left = C.expand([_one_entry()], tmp_path / "l", profile_names=["exact"])
    right = C.expand([_one_entry()], tmp_path / "r", profile_names=["exact"], drive="right")
    (rl,), (rr,) = left["rows"], right["rows"]
    assert "drive" not in rl and "drive" not in left["fingerprint"]          # left-hand drive as before
    assert rr["drive"] == "right" and rr["row_id"] == rl["row_id"].replace("__dash_right", "_rhd__dash_right", 1)
    assert right["fingerprint"]["drive"] == "right" and right["fingerprint_id"] != left["fingerprint_id"]
    assert any(d.startswith("drive:") for d in compare_fingerprints(left["fingerprint"], right["fingerprint"])[0])
    (info,) = C.run(tmp_path / "r", log=lambda *_: None)
    truth = [json.loads(x) for x in open(info["truth"], encoding="utf-8")]
    assert "right_mirror" in {t["gaze"]["eye"]["target"] for t in truth}   # the LHD script's left mirror, mirrored
    card = json.loads((Path(info["card"]).parent / "card.json").read_text(encoding="utf-8"))
    assert card["drive"] == "right"
    # the entry's words as they read in right-hand drive: in the manifest row, the card and the gallery
    assert rl["title"] == "邊講話邊看左鏡" and rr["title"] == "邊講話邊看右鏡" == card["title"]
    assert rr["summary"] == "看 right_mirror 0.8 s，dash_right 拍得到"
    from dms_model_simulator.gallery import gallery_html
    zh = gallery_html(tmp_path / "r", entries=[_one_entry()], lang="zh")
    en = gallery_html(tmp_path / "r", entries=[_one_entry()], lang="en")
    assert "邊講話邊看右鏡" in zh and "邊講話邊看左鏡" not in zh and "（右駕" in zh
    assert "Talking at the right mirror" in en and "right-hand drive" in en
    with pytest.raises(ValueError, match="drive"):
        C.expand([_one_entry()], tmp_path / "x", drive="rhd")


def test_cli_drive(tmp_path):
    from dms_model_simulator import cli
    cli.main(["run", "--scenario", "mirror_check", "--camera", "dash_right", "--fps", "5", "--drive", "right",
              "--format", "full", "--out", str(tmp_path)])
    assert (tmp_path / "mirror_check_rhd__dash_right.jsonl").exists()
    assert json.loads((tmp_path / "mirror_check_rhd__dash_right__camera.json").read_text())["drive"] == "right"
    cli.main(["cabin", "--drive", "right", "--out", str(tmp_path)])
    assert (tmp_path / "cabin_rhd.png").exists()
    with pytest.raises(SystemExit):
        cli.main(["run", "--scenario", "mirror_check", "--drive", "rhd", "--out", str(tmp_path)])


def test_view3d_drive(tmp_path):
    """dms-sim view3d --drive right: the mirrored cabin, body and cameras on a page saying right-hand drive; the
    left-hand-drive page is as before (no drive key, template as written)."""
    from dms_model_simulator import cli, view3d
    (lc, lcams), (rc, rcams) = cli._cameras(load_configs(), None), cli._cameras(load_configs(), None, "right")
    sl, sr = view3d.scene(lc, lcams), view3d.scene(rc, rcams)
    assert "drive" not in sl and sr["drive"] == "right"
    assert [c["p"] for c in sr["cameras"]] == [view3d._r(_flip_y(c["p"])) for c in sl["cameras"]]

    def body_points(cabin, flip=False):
        M = view3d._Meshes({})
        view3d.body_meshes(cabin.body, cabin, M)
        p = np.concatenate([np.array(t, float).reshape(-1, 3) for t in M.groups.values()])
        p = np.round(_flip_y(p) if flip else p, 9)
        return p[np.lexsort(p.T[::-1])]
    np.testing.assert_array_equal(body_points(rc), body_points(lc, flip=True))
    hl, hr = view3d.page(sl), view3d.page(sr)
    assert hl == view3d.page(sl) and "Generic left-hand-drive sedan." in hl and "orbitL" not in hl
    assert "Generic right-hand-drive sedan" in hr and "Right side, cut away" in hr and "orbitL" in hr
    cli.main(["view3d", "--drive", "right", "--out", str(tmp_path)])
    assert [p.name for p in tmp_path.iterdir()] == [view3d.FILE_RHD]


@pytest.mark.parametrize("text, expected", [
    ("頭向右轉 60° 看乘客，右手揉右眼", "頭向左轉 60° 看乘客，左手揉左眼"),
    ("路口左右張望：看左、右、左", "路口左右張望：看右、左、右"),          # 左右 ("left and right", "about") stays
    ("Head turned right 60°; left-hand knuckles, Left mirror", "Head turned left 60°; right-hand knuckles, Right mirror"),
    ("看left_mirror 與 road_left", "看right_mirror 與 road_right"),          # gaze targets swap, also next to Chinese
    ("dash_right, chest_left, wheel_cross_l, left_s", "dash_right, chest_left, wheel_cross_l, left_s"),   # IDs kept
    ("cigarette left in the mouth; left out; right next to it; right at the start; the look down is left;",
     "cigarette left in the mouth; left out; right next to it; right at the start; the look down is left;"),
    ("straight, bright, leftover", "straight, bright, leftover"),
])
def test_mirror_text(text, expected):
    assert D.mirror_text(text, {"left_mirror", "right_mirror", "road_left", "road_right"}) == expected
    assert D.mirror_text(D.mirror_text(text, {"left_mirror", "road_left"}), {"right_mirror", "road_right"}) == text


def test_scenario_description_reads_right_hand_drive():
    sc = dict(load_scenario("mirror_check"), description="Looks at left_mirror, then turns right")
    sim = Simulation(sc, "dash_right", fps=FPS, drive="right")
    assert sim.scenario["description"] == "Looks at right_mirror, then turns left"

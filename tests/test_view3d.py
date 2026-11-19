import json
import re
import numpy as np
import pytest
import yaml
from dms_model_simulator import geometry as G
from dms_model_simulator import view3d
from dms_model_simulator.cli import _cameras, main
from dms_model_simulator.config import CONFIG_DIR, load_configs, load_yaml


@pytest.fixture(scope="module")
def setup():
    cabin, cams = _cameras(load_configs(), None)
    return cabin, cams, view3d.scene(cabin, cams)


def _data(html: str) -> dict:
    m = re.search(r"const DATA = (\{.*?\});\n", html)
    assert m, "inlined scene not found"
    return json.loads(m.group(1))


def _tris(pos) -> np.ndarray:
    return np.asarray(pos, float).reshape(-1, 3, 3)


def test_scene_has_every_target_and_camera(setup):
    cabin, cams, sc = setup
    assert [t["name"] for t in sc["targets"]] == list(cabin.targets)
    assert [c["name"] for c in sc["cameras"]] == [c.name for c in cams]
    for t in sc["targets"]:
        np.testing.assert_allclose(t["p"], cabin.targets[t["name"]], atol=1e-4)


def test_far_targets_are_clipped_rays_in_their_direction(setup):
    cabin, _, sc = setup
    far = [t for t in sc["targets"] if t["far"]]
    assert {t["name"] for t in far} >= {"road_ahead", "road_left", "road_right"}
    for t in far:
        d = np.subtract(t["shown"], cabin.eye_point)
        assert np.linalg.norm(d) == pytest.approx(view3d.RAY_M, abs=1e-3)
        assert G.angle_between(d, cabin.targets[t["name"]] - cabin.eye_point) < 0.01
        assert t["ring"] is None
    for t in (t for t in sc["targets"] if not t["far"]):    # ring = the zone tolerance cone at the target's distance
        d = np.linalg.norm(np.subtract(t["shown"], cabin.eye_point))
        r = np.linalg.norm(np.array(t["ring"]) - np.array(t["shown"]), axis=1)
        np.testing.assert_allclose(r, d * np.tan(np.radians(cabin.zone_tol)), atol=2e-4)


def test_frustum_matches_the_camera(setup):
    _, cams, sc = setup
    for cam, c in zip(cams, sc["cameras"]):
        k = np.array(c["corners"])
        np.testing.assert_allclose(c["p"], cam.position, atol=1e-4)
        mid = k.mean(axis=0) - cam.position                  # centre of the far rectangle: on the optical axis
        assert G.angle_between(mid, cam.optical_axis) < 0.05
        assert np.linalg.norm(mid) == pytest.approx(view3d.FRUSTUM_M, abs=1e-3)
        for q in k:                                          # every corner on the image border (undistorted pinhole)
            pc = cam.to_cam(q)
            u, v = cam.fx * pc[0] / pc[2] + cam.cx, cam.fy * pc[1] / pc[2] + cam.cy
            assert min(abs(u), abs(u - cam.width)) < 0.5 and min(abs(v), abs(v - cam.height)) < 0.5
        body = _tris(c["mesh"]).reshape(-1, 3)               # the little camera box sits on the camera
        np.testing.assert_allclose(body.mean(axis=0), cam.position, atol=1e-3)


def test_meshes_are_triangles_with_outward_or_inward_faces(setup):
    cabin, _, sc = setup
    layers = {m["layer"] for m in sc["meshes"]}
    assert layers == {"body", "driver", "belt"}
    for m in sc["meshes"]:
        assert len(m["pos"]) % 9 == 0 and m["pos"]
        assert 0 < m["alpha"] <= 1 and all(0 <= c <= 1 for c in m["color"])
    # shell quads face the cabin: the floor up, the roof down
    body = load_yaml(view3d.BODY)
    M = view3d._Meshes({})
    floor = next(p for p in body["parts"] if p["name"] == "floor")
    roof = next(p for p in body["parts"] if p["name"] == "roof")
    for part, sign in ((floor, 1), (roof, -1)):
        for t in view3d.quad(part["points"], body["toward"]):
            assert sign * np.cross(t[1] - t[0], t[2] - t[0])[2] > 0
    # a closed box faces outwards
    for t in view3d.box([0, 0, 0], [1, 2, 3], view3d.rot_y(12)):
        n, c = np.cross(t[1] - t[0], t[2] - t[0]), sum(t) / 3
        assert np.dot(n, c) > 0
    del M


def test_car_view_and_simulator_view_split_the_meshes(setup):
    cabin, _, sc = setup
    assert {m["show"] for m in sc["meshes"]} == {"car", "both"}
    # every body.yaml part is car-only: the simulator view draws nothing the simulation does not read
    body = load_yaml(view3d.BODY)
    M = view3d._Meshes(body["colors"])
    view3d.body_meshes(body, cabin, M)
    assert M.out() and {m["show"] for m in M.out()} == {"car"}
    assert {m["layer"] for m in sc["meshes"] if m["show"] == "both"} == {"body", "driver", "belt"}  # wheel, head, belt
    # the head is the simulation's own: every landmark is listed, and the face fan touches every outline landmark
    head = cabin.head_state(0.0, 0.0, 0.0)
    assert {q["name"] for q in sc["landmarks"]} == set(head.landmarks)
    for q in sc["landmarks"]:
        np.testing.assert_allclose(q["p"], head.landmarks[q["name"]], atol=1e-4)
    face = [m for m in sc["meshes"] if m["color"] == view3d._rgb(view3d.FACE_COLOR)]
    pts = _tris(face[0]["pos"]).reshape(-1, 3)
    for k in cabin.face_outline:
        assert np.min(np.linalg.norm(pts - head.landmarks[k], axis=1)) < 1e-3, k
    html = view3d.page(sc)
    assert 'id="w_sim"' in html and "draw=sim" in html


def test_body_parts_follow_the_targets(setup):
    cabin, _, sc = setup
    body = load_yaml(view3d.BODY)
    M = view3d._Meshes(body["colors"])
    view3d.body_meshes({"parts": [p for p in body["parts"] if p["type"] in ("panel", "housing")]}, cabin, M)
    tris = np.concatenate([_tris(m["pos"]) for m in M.out()]).reshape(-1, 3)
    for name in ("cluster", "center_display", "left_mirror", "right_mirror", "rearview_mirror"):
        assert np.min(np.linalg.norm(tris - cabin.targets[name], axis=1)) < 0.2, name
    moved = yaml.safe_load(yaml.safe_dump(load_configs()["cabin"]))
    moved["targets"]["left_mirror"] = [0.5, 1.2, 0.0]          # move a mirror: its housing moves with it
    from dms_model_simulator.world import Cabin
    c2 = Cabin(moved)
    M2 = view3d._Meshes(body["colors"])
    view3d.body_meshes({"parts": [p for p in body["parts"] if p["name"] == "mirror_left"]}, c2, M2)
    pts = np.concatenate([_tris(m["pos"]) for m in M2.out()]).reshape(-1, 3)
    assert np.min(np.linalg.norm(pts - [0.5, 1.2, 0.0], axis=1)) < 0.2
    with pytest.raises(ValueError):
        view3d.body_meshes({"parts": [{"name": "x", "type": "cone"}]}, cabin, M2)


def test_passenger_sits_mirrored_about_the_centreline(setup):
    cabin, _, sc = setup
    t = cabin.targets
    centre = 0.5 * (t["left_mirror"][1] + t["right_mirror"][1])
    assert sc["passenger_eye"][1] == pytest.approx(2 * centre - cabin.eye_point[1], abs=1e-3)


def test_page_is_self_contained_and_reproducible(setup):
    _, _, sc = setup
    html = view3d.page(sc)
    assert html == view3d.page(sc)
    for bad in ("http://", "https://", "src=", "fetch(", "import ", "{{"):
        assert bad not in html, bad
    assert '<html lang="en">' in html
    assert _data(html)["scene"]["targets"][0]["name"] == sc["targets"][0]["name"]


def test_cli_writes_one_english_page(tmp_path):
    main(["view3d", "--out", str(tmp_path), "--camera", "steering_column", "dash_right"])
    assert [p.name for p in tmp_path.iterdir()] == [view3d.FILE]
    first = (tmp_path / view3d.FILE).read_text(encoding="utf-8")
    assert [c["name"] for c in _data(first)["scene"]["cameras"]] == ["steering_column", "dash_right"]
    main(["view3d", "--out", str(tmp_path), "--camera", "steering_column", "dash_right"])
    assert (tmp_path / view3d.FILE).read_text(encoding="utf-8") == first


def test_cli_takes_other_config_files(tmp_path):
    cams = yaml.safe_load((CONFIG_DIR / "cameras.yaml").read_text(encoding="utf-8"))
    cams["cameras"] = {"only": {"position": [0.6, -0.1, 0.0]}}
    cab = yaml.safe_load((CONFIG_DIR / "cabin.yaml").read_text(encoding="utf-8"))
    cab["targets"]["hud"] = [1.0, 0.0, -0.05]
    body = {"colors": {}, "parts": [{"name": "floor", "type": "quad", "color": "#808080",
                                     "points": [[-1, 1, -1], [1, 1, -1], [1, -1, -1], [-1, -1, -1]]}]}
    for name, d in (("cams.yaml", cams), ("cab.yaml", cab), ("body.yaml", body)):
        (tmp_path / name).write_text(yaml.safe_dump(d), encoding="utf-8")
    main(["view3d", "--out", str(tmp_path), "--cabin", str(tmp_path / "cab.yaml"),
          "--cameras", str(tmp_path / "cams.yaml"), "--body", str(tmp_path / "body.yaml")])
    html = (tmp_path / view3d.FILE).read_text(encoding="utf-8")
    d = _data(html)
    assert [c["name"] for c in d["scene"]["cameras"]] == ["only"] and d["scene"]["cameras"][0]["p"] == [0.6, -0.1, 0.0]
    assert "hud" in [t["name"] for t in d["scene"]["targets"]]
    assert d["sources"] == ["cab.yaml", "cams.yaml", "body.yaml"]
    body_meshes = [m for m in d["scene"]["meshes"] if m["color"] == [0.502, 0.502, 0.502]]
    assert len(body_meshes) == 1 and len(body_meshes[0]["pos"]) == 18

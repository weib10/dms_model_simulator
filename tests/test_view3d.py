import json
import re
import numpy as np
import pytest
from dms_model_simulator import geometry as G
from dms_model_simulator import view3d
from dms_model_simulator.cli import _cameras, main
from dms_model_simulator.config import load_configs


@pytest.fixture(scope="module")
def setup():
    cabin, cams = _cameras(load_configs(), None)
    return cabin, cams, view3d.scene(cabin, cams)


def _data(html: str) -> dict:
    m = re.search(r"const DATA = (\{.*?\});\n", html)
    assert m, "inlined scene not found"
    return json.loads(m.group(1))


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
    near = [t for t in sc["targets"] if not t["far"]]
    for t in near:                                 # ring = the zone tolerance cone at the target's distance
        d = np.linalg.norm(np.subtract(t["shown"], cabin.eye_point))
        ring = np.array(t["ring"])
        r = np.linalg.norm(ring - np.array(t["shown"]), axis=1)
        np.testing.assert_allclose(r, d * np.tan(np.radians(cabin.zone_tol)), atol=2e-4)


def test_frustum_matches_the_camera(setup):
    _, cams, sc = setup
    for cam, c in zip(cams, sc["cameras"]):
        k = np.array(c["corners"])
        np.testing.assert_allclose(c["p"], cam.position, atol=1e-4)
        mid = k.mean(axis=0) - cam.position                  # centre of the far rectangle: on the optical axis
        assert G.angle_between(mid, cam.optical_axis) < 0.05
        assert np.linalg.norm(mid) == pytest.approx(view3d.FRUSTUM_M, abs=1e-3)
        # left and right edges of the image span the horizontal field of view
        assert G.angle_between(k[0] - cam.position, k[1] - cam.position) == pytest.approx(
            2 * np.degrees(np.arctan(np.tan(np.radians(cam.hfov) / 2) / np.hypot(1, np.tan(np.radians(cam.vfov) / 2)))),
            abs=0.05)
        # every corner projects onto the image border with the camera's own (undistorted) pinhole
        for q in k:
            pc = cam.to_cam(q)
            u, v = cam.fx * pc[0] / pc[2] + cam.cx, cam.fy * pc[1] / pc[2] + cam.cy
            assert min(abs(u), abs(u - cam.width)) < 0.5 and min(abs(v), abs(v - cam.height)) < 0.5


def test_wheel_without_hands_and_head_faces_forward(setup):
    cabin, _, sc = setup
    assert len(sc["wheel"]["rim"]) == 1 and len(sc["wheel"]["hub"]) == 1 and len(sc["wheel"]["spokes"]) == 3
    d = np.subtract(sc["head"]["face_tip"], sc["head"]["c"])
    assert G.angle_between(d, [1.0, 0.0, 0.0]) < 0.01
    lo, hi = np.array(sc["bounds"]["lo"]), np.array(sc["bounds"]["hi"])
    assert np.all(hi - lo < 2 * view3d.NEAR_M)               # far targets do not stretch the fitted box


def test_page_is_self_contained_and_reproducible(setup):
    _, _, sc = setup
    for lg in ("zh", "en"):
        html = view3d.page(sc, lg)
        assert html == view3d.page(sc, lg)
        for bad in ("http://", "https://", "src=", "fetch(", "import ", "{{"):
            assert bad not in html, bad
        d = _data(html)
        assert d["scene"]["targets"][0]["name"] == sc["targets"][0]["name"]
        assert set(d["text"]) == set(view3d.TEXT["en"])
    assert set(view3d.TEXT["zh"]) == set(view3d.TEXT["en"])
    with pytest.raises(ValueError):
        view3d.page(sc, "fr")


def test_cli_writes_both_languages(tmp_path):
    main(["view3d", "--out", str(tmp_path), "--camera", "steering_column", "dash_right"])
    zh, en = (tmp_path / view3d.FILES["zh"]).read_text(encoding="utf-8"), (tmp_path / view3d.FILES["en"]).read_text(encoding="utf-8")
    assert '<html lang="zh-Hant">' in zh and '<html lang="en">' in en
    assert [c["name"] for c in _data(en)["scene"]["cameras"]] == ["steering_column", "dash_right"]
    first = en
    main(["view3d", "--out", str(tmp_path), "--camera", "steering_column", "dash_right", "--lang", "en"])
    assert (tmp_path / view3d.FILES["en"]).read_text(encoding="utf-8") == first


def test_cli_takes_other_config_files(tmp_path):
    import yaml
    from dms_model_simulator.config import CONFIG_DIR
    cams = yaml.safe_load((CONFIG_DIR / "cameras.yaml").read_text(encoding="utf-8"))
    cams["cameras"] = {"only": {"position": [0.6, -0.1, 0.0]}}
    cab = yaml.safe_load((CONFIG_DIR / "cabin.yaml").read_text(encoding="utf-8"))
    cab["targets"]["hud"] = [1.0, 0.0, -0.05]
    (tmp_path / "cams.yaml").write_text(yaml.safe_dump(cams), encoding="utf-8")
    (tmp_path / "cab.yaml").write_text(yaml.safe_dump(cab), encoding="utf-8")
    main(["view3d", "--out", str(tmp_path), "--lang", "en",
          "--cabin", str(tmp_path / "cab.yaml"), "--cameras", str(tmp_path / "cams.yaml")])
    html = (tmp_path / view3d.FILES["en"]).read_text(encoding="utf-8")
    d = _data(html)["scene"]
    assert [c["name"] for c in d["cameras"]] == ["only"] and d["cameras"][0]["p"] == [0.6, -0.1, 0.0]
    assert "hud" in [t["name"] for t in d["targets"]]
    assert "cab.yaml" in html and "cams.yaml" in html

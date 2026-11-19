"""The car body (configs/body.yaml) as an occluder of the simulation."""
import copy
import numpy as np
import pytest
from dms_model_simulator import body as B
from dms_model_simulator import geometry as G
from dms_model_simulator.cli import _cameras
from dms_model_simulator.config import CONFIG_DIR, load_configs, load_yaml
from dms_model_simulator.simulate import Simulation
from dms_model_simulator.world import Cabin



@pytest.fixture(scope="module")
def setup():
    return _cameras(load_configs(), None)


def _dist_to_triangles(p, tris) -> float:
    """Distance from p to the nearest triangle (exact: plane inside the triangle, else nearest edge)."""
    best = np.inf
    for a, b, c in tris:
        n = G.normalize(np.cross(b - a, c - a))
        q = p - np.dot(p - a, n) * n
        inside = all(np.dot(np.cross(v1 - v0, q - v0), n) >= 0 for v0, v1 in ((a, b), (b, c), (c, a)))
        if inside:
            best = min(best, abs(float(np.dot(p - a, n))))
        else:
            best = min(best, *(G.seg_point_distance(v0, v1, p) for v0, v1 in ((a, b), (b, c), (c, a))))
    return best


def test_the_cabin_config_carries_the_body(setup):
    cabin, _ = setup
    assert cabin.body["parts"] and cabin.body_occluder["type"] == "mesh"
    names = {p["name"] for p in cabin.body["parts"]}
    assert set(cabin.body_occluder["part"]) == names
    assert all(float(p.get("alpha", 1.0)) == 1.0 for p in cabin.body["parts"])          # no glass: it hides nothing
    # the body is always on, last (a hand or the wheel in front of it is named first)
    for hand in ("wheel", "none", "phone"):
        assert cabin.occluder_geometry(hand, cabin.head_state(0, 0, 0))[-1]["name"] == "car_body"
    # a cabin config without a body has no body occluder
    cfg = copy.deepcopy(load_configs()["cabin"]); cfg.pop("body")
    c2 = Cabin(cfg)
    assert c2.body_occluder is None and all(o["type"] != "mesh" for o in c2.occluder_geometry("wheel", c2.head_state(0, 0, 0)))


def test_every_camera_is_clear_of_the_body_and_sees_the_whole_nominal_face(setup):
    """Every camera stays more than 5 mm from every body surface over its mount-offset box of configs/variation.yaml
    (a 3 x 3 x 3 grid of it), so no mount variant puts a lens into the body."""
    import itertools
    cabin, cams = setup
    mo = load_yaml(CONFIG_DIR / "variation.yaml")["camera"]["mount_offset"]
    grid = [np.array(v) for v in itertools.product(*(np.linspace(*mo[k], 3) for k in ("x", "y", "z")))]
    tris = cabin.body_occluder["tris"]
    head = cabin.head_state(0.0, 0.0, 0.0)
    occl = cabin.occluder_geometry("none", head)
    for cam in cams:
        assert min(_dist_to_triangles(cam.position + g, tris) for g in grid) > 0.005, cam.name
        # outside every closed part (box, cylinder: 12+ outward triangles): a ray from the camera to far away
        # crosses each of them an even number of times
        far = cam.position + np.array([0.31, 0.17, 9.0])
        for part in B.parts(cabin.body, cabin.targets, cabin.eye_point):
            if not part["closed"]:
                continue
            n = sum(G.ray_hit_triangles(cam.position, far, np.array([t]), 0.0) >= 0 for t in part["tris"])
            assert n % 2 == 0, (cam.name, part["name"])
        hidden = {k: cabin.ray_blocked(cam.position, p, occl, cabin.occlusion_eps) for k, p in head.landmarks.items()}
        assert not {k: v for k, v in hidden.items() if v}, cam.name


def test_cached_ray_test_matches_the_plain_one(setup):
    cabin, cams = setup
    o, rng = cabin.body_occluder, np.random.default_rng(0)
    for cam in cams:
        pre = G.triangles_from(cam.position, o["tris"], o["two_sided"])
        for _ in range(300):
            p = rng.uniform([-0.5, -0.8, -0.8], [0.7, 0.45, 0.4])
            a = G.ray_hit_triangles(cam.position, p, o["tris"], 0.003)
            b = G.ray_hit_triangles_from(cam.position, p, pre, 0.003)
            assert (a >= 0) == (b >= 0), (cam.name, p)
            if a >= 0:
                assert o["part"][a] == o["part"][b]


def test_ray_hit_triangles():
    tri = np.array([[[1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, 0.0, 1.0]]])
    o = np.zeros(3)
    assert G.ray_hit_triangles(o, [2.0, 0.0, 0.0], tri) == 0
    assert G.ray_hit_triangles(o, [0.9, 0.0, 0.0], tri) == -1            # short of the triangle
    assert G.ray_hit_triangles(o, [1.0, 0.0, 0.0], tri, eps=0.003) == -1  # a point on the surface is not hidden
    assert G.ray_hit_triangles(o, [2.0, 0.0, 3.0], tri) == -1            # passes above it


def test_the_dash_hides_what_is_below_its_edge(setup):
    """A point under the dash, seen from the camera on top of it, is hidden by the dash and named after the part."""
    cabin, cams = setup
    cam = next(c for c in cams if c.name == "dash_right")
    occl = cabin.occluder_geometry("none", cabin.head_state(0, 0, 0))
    assert cabin.ray_blocked(cam.position, np.array([0.55, -0.34, -0.40]), occl, cabin.occlusion_eps) \
        in {"dash_top", "dash_top_right", "dash_underside", "dash_underside_right", "dash_face"}


def test_a_scenario_cabin_override_reaches_the_body():
    """A scenario's cabin: block can change the body like any other cabin key (here: no parts at all)."""
    a = Simulation("hand_to_face", "steering_column", seed=0)
    b = Simulation("hand_to_face", "steering_column", seed=0, cfgs=load_configs())
    cfg = load_configs(); cfg["cabin"]["body"] = {"parts": []}
    c = Simulation("hand_to_face", "steering_column", seed=0, cfgs=cfg)
    assert a.cabin.body_occluder is not None and b.cabin.body_occluder is not None and c.cabin.body_occluder is None


def test_the_belt_lies_in_front_of_the_torso(setup):
    """No camera loses any of the seat belt to the torso it lies on (or to the seat behind it)."""
    cabin, cams = setup
    occl = cabin.occluder_geometry("none", cabin.head_state(0, 0, 0))
    pts = [np.asarray(a) + t * (np.asarray(b) - np.asarray(a))
           for a, b in cabin.seatbelt_labelled_pieces() for t in np.linspace(0.02, 0.98, 25)]
    for cam in cams:
        names = {cabin.ray_blocked(cam.position, p, occl, cabin.occlusion_eps) for p in pts}
        assert not names & {"torso", "seatback_driver", "seat_driver"}, cam.name

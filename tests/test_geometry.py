import numpy as np
from dms_model_simulator import geometry as G


def test_euler_roundtrip():
    for y, p, r in [(0, 0, 0), (30, -10, 5), (-45, 20, -12), (80, 30, 0)]:
        R = G.euler_to_R(y, p, r)
        assert np.allclose(G.R_to_euler(R), (y, p, r), atol=1e-6)


def test_yaw_left_pitch_up_signs():
    f = G.euler_to_R(30, 0, 0)[:, 0]
    assert f[1] > 0            # yaw>0 turns towards +Y (left)
    f = G.euler_to_R(0, 20, 0)[:, 0]
    assert f[2] > 0            # pitch>0 looks up
    assert np.allclose(G.dir_to_yaw_pitch(G.yaw_pitch_to_dir(25, -15)), (25, -15))


def test_look_at_opencv_convention():
    cam = np.array([1.0, 0, 0]); R = G.look_at(cam, np.zeros(3))
    pc = R @ (np.zeros(3) - cam)
    assert np.allclose(pc, [0, 0, 1])               # target straight ahead on +z
    left = R @ (np.array([0, 0.2, 0]) - cam)        # a point to the vehicle's left
    assert left[0] > 0                               # ... appears on the camera's right (facing backwards)
    up = R @ (np.array([0, 0, 0.2]) - cam)
    assert up[1] < 0                                 # up is -y in OpenCV


def test_face_frame_zero_when_facing_camera():
    cam = np.array([0.6, 0, 0]); R = G.look_at(cam, np.zeros(3))
    R_head = G.euler_to_R(0, 0, 0)                   # face looks +X, straight at the camera
    rel = G.R_FACE_FROM_CV @ R @ R_head
    assert np.allclose(G.R_to_euler(rel), (0, 0, 0), atol=1e-6)
    rel = G.R_FACE_FROM_CV @ R @ G.euler_to_R(20, 10, 0)
    y, p, _ = G.R_to_euler(rel)
    assert y > 0 and p > 0                           # own-left and up stay positive


def test_seg_distances():
    assert np.isclose(G.seg_seg_distance([0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 2, 0]), 1.0)
    assert np.isclose(G.seg_point_distance([0, 0, 0], [1, 0, 0], [2, 0, 0]), 1.0)


def test_look_at_straight_up_or_down_is_a_rotation():
    for target in ([0, 0, -1.0], [0, 0, 1.0]):
        R = G.look_at(np.zeros(3), np.array(target))
        assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
        assert np.allclose(R[2], target)                     # optical axis = view direction
        assert np.isclose(R[1] @ [1.0, 0, 0], -1.0)          # fallback: image up = vehicle forward


def test_distortion_fold_over_is_rejected_and_sizes_follow_distortion():
    from dms_model_simulator.camera import Camera
    spec = {"position": [0.6, 0, 0], "width": 1280, "height": 800, "hfov_deg": 60, "distortion": [-0.3, 0.0]}
    c = Camera("t", spec, np.zeros(3))
    assert np.isclose(c.r2_max, 1 / 0.9)                     # 1 + 3 k1 r^2 = 0
    for xn in (1.2, 1.5, 2.0, 3.0):                          # used to fold back into the image
        p = c.position + c.R.T @ np.array([xn, 0, 1.0])
        uv, ok = c.project(p)
        assert not ok and np.isnan(uv).all() and c.project_size(p, 0.1) == 0.0
    p = c.position + c.R.T @ np.array([0.5, 0, 1.0])
    assert c.project(p)[1]
    assert np.isclose(c.project_size(p, 0.1), c.fx * 0.1 * (1 - 0.3 * 0.25))
    assert np.isinf(Camera("t", {**spec, "distortion": [0.1, 0.0]}, np.zeros(3)).r2_max)   # never folds


def test_ray_primitives_block_only_strictly_before_the_point():
    o, c = np.zeros(3), np.array([1.0, 0, 0])
    assert G.ray_hits_sphere(o, [2.0, 0, 0], c, 0.1)                 # behind the sphere
    assert G.ray_hits_sphere(o, [1.0, 0, 0], c, 0.1)                 # inside the sphere
    assert not G.ray_hits_sphere(o, [0.9, 0, 0], c, 0.1, eps=1e-3)   # on the surface facing the camera
    assert not G.ray_hits_sphere(o, [0.5, 0, 0], c, 0.1)             # in front of it
    assert not G.ray_hits_sphere(o, [2.0, 0.5, 0], c, 0.1)           # ray misses
    n = np.array([1.0, 0, 0])
    assert G.ray_hits_disc(o, [2.0, 0, 0], c, n, 0.1) and not G.ray_hits_disc(o, [2.0, 0.5, 0], c, n, 0.1)
    assert not G.ray_hits_disc(o, [1.0, 0, 0], c, n, 0.1, eps=1e-3)  # point on the disc itself
    a, b = [1.0, -1.0, 0], [1.0, 1.0, 0]
    assert G.ray_hits_capsule(o, [2.0, 0, 0], a, b, 0.05)
    assert not G.ray_hits_capsule(o, [0.955, 0, 0], a, b, 0.05, eps=0.01)  # within eps of the rod surface


def test_polyline_ray_test_matches_piecewise_capsules():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        o, p = rng.normal(size=(2, 3))
        pts = 0.5 * rng.normal(size=(6, 3))
        r, eps = rng.uniform(0.05, 0.5), rng.choice([0.0, 0.01])
        piecewise = any(G.ray_hits_capsule(o, p, pts[i], pts[i + 1], r, eps) for i in range(5))
        assert G.ray_hits_polyline(o, p, pts, r, eps) == piecewise


def test_slerp_constant_angular_speed():
    a, b = G.yaw_pitch_to_dir(0, 0), G.yaw_pitch_to_dir(80, 20)
    ang = [G.angle_between(a, G.slerp(a, b, t)) for t in np.linspace(0, 1, 11)]
    assert np.allclose(np.diff(ang), G.angle_between(a, b) / 10)

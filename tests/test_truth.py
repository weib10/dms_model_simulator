import numpy as np
import pytest
from dms_model_simulator.behavior import DriverFrame
from dms_model_simulator.camera import Camera
from dms_model_simulator.config import load_configs
from dms_model_simulator.simulate import Simulation
from dms_model_simulator.truth import compute_truth
from dms_model_simulator.world import Cabin
from dms_model_simulator import geometry as G


def _setup(cam="steering_column"):
    cfgs = load_configs(); cabin = Cabin(cfgs["cabin"])
    return cabin, Camera.from_config(cam, cfgs["cameras"], cabin.nominal_head_center())


def _frame(target, cabin, yaw=0, pitch=0, hand="wheel"):
    return DriverFrame(0, yaw, pitch, 0, cabin.targets[target], target, 1, 1, False, False, hand, "none")


def test_gaze_points_exactly_at_target_and_zone_matches():
    cabin, cam = _setup()
    for tgt in cabin.targets:
        tr = compute_truth(cabin, cam, _frame(tgt, cabin))
        o, d = np.array(tr["gaze"]["origin_vehicle"]), np.array(tr["gaze"]["dir_vehicle"])
        assert G.angle_between(d, cabin.targets[tgt] - o) < 1e-6
        assert tr["gaze"]["zone"] == tgt


def test_gaze_into_camera_is_zero_in_camera_frame():
    cabin, cam = _setup()
    f = DriverFrame(0, 0, 0, 0, cam.position, "camera", 1, 1, False, False, "wheel", "none")
    tr = compute_truth(cabin, cam, f)
    assert tr["gaze"]["gaze_to_camera_deg"] < 1e-6
    # direction in camera frame is anti-parallel to the eye position in the camera frame
    eye_c = np.array(tr["head_center_camera"])
    assert G.angle_between(np.array(tr["gaze"]["dir_camera"]), -eye_c) < 1e-6


def test_face_dir_is_the_head_forward_axis():
    """face_dir points where the head points: the vehicle-frame direction of the head yaw / pitch (the same
    definition as the head-off-road check), from the eye midpoint, whatever the gaze does."""
    cabin, cam = _setup()
    for tgt, yaw, pitch in (("road_ahead", 0, 0), ("left_mirror", 25, 10), ("phone_lap", -40, -30), ("road_ahead", 70, 5)):
        tr = compute_truth(cabin, cam, _frame(tgt, cabin, yaw=yaw, pitch=pitch))
        fd = tr["face_dir"]
        assert G.angle_between(np.array(fd["dir_vehicle"]), G.yaw_pitch_to_dir(yaw, pitch)) < 1e-9
        assert fd["yaw_vehicle"] == pytest.approx(yaw, abs=1e-9) and fd["pitch_vehicle"] == pytest.approx(pitch, abs=1e-9)
        assert fd["origin_vehicle"] == tr["gaze"]["origin_vehicle"] and fd["origin_px"] == tr["gaze"]["origin_px"]
        assert G.angle_between(np.array(fd["dir_camera"]), cam.dir_to_cam(np.array(fd["dir_vehicle"]))) < 1e-9
        # the camera angles of the face direction equal the head pose yaw / pitch in the camera face frame
        assert fd["yaw_camera"] == pytest.approx(tr["head_pose_camera"]["yaw"], abs=1e-9)
        assert fd["pitch_camera"] == pytest.approx(tr["head_pose_camera"]["pitch"], abs=1e-9)
        assert fd["zone"] == cabin.classify_zone(np.array(fd["origin_vehicle"]), np.array(fd["dir_vehicle"]))[0]
    tr = compute_truth(cabin, cam, _frame("phone_lap", cabin))       # eyes down, head straight
    assert tr["face_dir"]["zone"] == "road_ahead" and tr["gaze"]["zone"] == "phone_lap"


def test_face_dir_along_the_reversed_optical_axis_is_zero_in_camera_frame():
    cabin, cam = _setup()
    yaw, pitch = G.dir_to_yaw_pitch(-np.asarray(cam.optical_axis, float))
    fd = compute_truth(cabin, cam, _frame("road_ahead", cabin, yaw=yaw, pitch=pitch))["face_dir"]
    assert abs(fd["yaw_camera"]) < 1e-9 and abs(fd["pitch_camera"]) < 1e-9


def test_face_frame_pose_consistent_with_vehicle_pose():
    cabin, cam = _setup()
    tr = compute_truth(cabin, cam, _frame("road_ahead", cabin, yaw=25, pitch=10))
    # camera is nearly on-axis, so camera-frame pose ~ vehicle-frame pose (small offset from camera being below)
    assert abs(tr["head_pose_camera"]["yaw"] - 25) < 6
    assert tr["head_pose_camera"]["pitch"] > 10        # camera sits below the eyes, so the face looks 'up' relative to it


def test_projection_face_size_scales_with_distance_and_landmarks_inside():
    cabin, cam = _setup()
    tr = compute_truth(cabin, cam, _frame("road_ahead", cabin))
    face = next(d for d in tr["detections"] if d["class"] == "face")
    dist = np.linalg.norm(np.array(tr["head_center_vehicle"]) - cam.position)
    expected = cam.fx * 0.15 / dist          # ~ temple to temple (0.144) + padding
    assert 0.8 * expected < face["width_px"] < 1.4 * expected
    assert all(tr["landmarks_visible"][k] for k in ("left_eye_center", "right_eye_center", "nose_tip"))


def test_hand_on_call_occludes_right_side():
    """The call hand is at the right ear: it hides the right cheek from a camera on the right with the head turned
    left, contains only the right ear and the face contour in front of it, and leaves the mouth to a camera on
    the left."""
    cabin, _ = _setup()
    cam = Camera("right", {"position": [0.4, -0.6, -0.1], "width": 1280, "height": 800, "hfov_deg": 70}, np.zeros(3))
    tr = compute_truth(cabin, cam, _frame("road_ahead", cabin, yaw=30, hand="call"))
    assert tr["occluded_by"].get("right_cheek") == "hand_on_call"
    head = cabin.head_state(0, 0, 0)
    sph = next(o for o in cabin.occluder_geometry("call", head) if o["name"] == "hand_on_call")
    assert all(np.linalg.norm(p - sph["center_world"]) > sph["radius"]          # ear and jaw are under the hand
               for n, p in head.landmarks.items() if n not in ("right_ear", "right_jaw"))
    _, left = _setup("a_pillar_left")
    tr = compute_truth(cabin, left, _frame("road_ahead", cabin, hand="call"))
    mouth = next(d for d in tr["detections"] if d["class"] == "mouth")
    assert mouth["occlusion"] == 0.0
    assert {n for n, o in tr["occluded_by"].items() if o == "hand_on_call"} <= {"right_ear", "right_jaw"}   # hand over the ear


def test_simulation_runs_and_is_reproducible():
    a = Simulation("mirror_check", "cluster", seed=3); a.run()
    b = Simulation("mirror_check", "cluster", seed=3); b.run()
    assert len(a.records) == len(b.records) > 0
    assert a.records[50]["pred"] == b.records[50]["pred"]
    valid = [r["pred"]["gaze"]["angular_error_deg"] for r in a.records if r["pred"]["gaze"] and r["pred"]["gaze"].get("valid")]
    assert 0 < np.mean(valid) < 10


def test_emulator_overrides_zero_noise_gives_exact_output():
    ov = {"pipeline": {"drop_frame_prob": 0}, "gaze": {"base_std_deg": 0, "per_deg_gaze_to_camera": 0, "per_deg_rel_head_yaw": 0, "per_px_inv_eye_width": 0},
          "detection": {"face": {"p_max": 1, "min_width_px": 0}, "eye": {"p_max": 1, "min_width_px": 0}, "false_positives": {"rate_per_frame": 0}}}
    s = Simulation("mirror_check", "steering_column", seed=1, emulator_overrides=ov); s.run()
    errs = [r["pred"]["gaze"]["angular_error_deg"] for r in s.records if r["pred"]["gaze"].get("valid")]
    assert errs and max(errs) < 1e-6


ZERO_GAZE_NOISE = {"gaze": {"base_std_deg": 0, "per_deg_gaze_to_camera": 0, "per_deg_rel_head_yaw": 0, "per_px_inv_eye_width": 0},
                   "detection": {"face": {"p_max": 1, "min_width_px": 0}, "eye": {"p_max": 1, "min_width_px": 0},
                                 "false_positives": {"rate_per_frame": 0}}}


def test_latency_constant_across_dropped_frames():
    """Regression: a dropped frame used to bypass the latency queue, adding one frame of latency per drop."""
    lat = 3
    ov = {**ZERO_GAZE_NOISE, "pipeline": {"drop_frame_prob": 0.2, "latency_frames": lat}}
    s = Simulation("mirror_check", "steering_column", seed=4, emulator_overrides=ov); s.run()
    assert all(r["pred"].get("latency_fill") for r in s.records[:lat])
    n_checked = 0
    for i, r in enumerate(s.records[lat:], start=lat):
        g = r["pred"]["gaze"]
        if r["pred"]["dropped"] or not g or not g.get("valid"):
            continue
        assert G.angle_between(np.array(g["dir_vehicle"]), np.array(s.records[i - lat]["truth"]["gaze"]["dir_vehicle"])) < 1e-6
        n_checked += 1
    assert n_checked > 100 and sum(r["pred"]["dropped"] for r in s.records) > 30


def test_predicted_boxes_inside_image():
    s = Simulation("smoking", "a_pillar_left", seed=0); s.run()
    W, H = s.camera.width, s.camera.height
    boxes = [d["bbox"] for r in s.records for d in r["pred"]["detections"]]
    assert boxes and all(0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H for x1, y1, x2, y2 in boxes)


def test_cabin_geometry_from_yaml():
    cfgs = load_configs()
    cabin = Cabin(cfgs["cabin"])
    head = cabin.head_state(0, 0, 0)
    assert np.allclose(cabin.phone_position("call", head), head.head_to_vehicle(cfgs["cabin"]["phone_call_offset"]))
    assert np.allclose(cabin.nominal_head_center(),
                       np.add(cfgs["cabin"]["driver"]["head_pivot"], cfgs["cabin"]["driver"]["head_center_offset"]))
    cfgs["cabin"]["landmark_visibility"]["min_facing_cos"] = 2.0          # nothing can face the camera
    cabin2 = Cabin(cfgs["cabin"])
    cam = Camera.from_config("steering_column", cfgs["cameras"], cabin2.nominal_head_center())
    assert not any(compute_truth(cabin2, cam, _frame("road_ahead", cabin2))["landmarks_visible"].values())


def _cabin_with(**driver):
    cfgs = load_configs()
    cfgs["cabin"]["driver"].update(driver)
    return Cabin(cfgs["cabin"])


def test_presets_aim_at_the_nominal_eye_point_and_read_zero_into_the_lens():
    """Camera-frame angles are axis based; with the presets the nominal eye point is on the optical axis, so a
    face / gaze at the nominal eye point looking into the lens reads exactly 0."""
    cfgs = load_configs(); cabin = Cabin(cfgs["cabin"])
    eye = cabin.head_state(0, 0, 0).origin
    for name, spec in cfgs["cameras"]["cameras"].items():
        if isinstance(spec.get("orientation"), dict):              # fixed mount aimed by yaw / pitch (dash_right)
            continue
        cam = Camera.from_config(name, cfgs["cameras"], cabin.nominal_head_center())
        assert np.allclose(cam.project(eye)[0], [cam.cx, cam.cy])
        f = DriverFrame(0, 0, 0, 0, cam.position, "camera", 1, 1, False, False, "wheel", "none")
        g = compute_truth(cabin, cam, f)["gaze"]
        assert abs(g["yaw_camera"]) < 1e-9 and abs(g["pitch_camera"]) < 1e-9, name
        y, p = G.dir_to_yaw_pitch(cam.position - eye)                 # head axis parallel to the reversed axis
        pose = compute_truth(cabin, cam, _frame("road_ahead", cabin, yaw=y, pitch=p))["head_pose_camera"]
        assert abs(pose["yaw"]) < 1e-9 and abs(pose["pitch"]) < 1e-9, name


def test_face_occlusion_counts_external_occluders_only():
    """Turning away hides landmarks (landmarks_visible, self_occluded_fraction) but is not 'occlusion': the
    emulator's yaw term models it already."""
    cabin, cam = _setup()
    tr = compute_truth(cabin, cam, _frame("road_ahead", cabin, yaw=60, hand="lap"))
    face = next(d for d in tr["detections"] if d["class"] == "face")
    assert face["occlusion"] == 0.0 and not tr["occluded_by"]
    face_vis = [v for n, v in tr["landmarks_visible"].items() if n not in ("left_ear", "right_ear")]   # ears: not face
    hidden = sum(not v for v in face_vis)
    assert hidden > 0 and np.isclose(face["self_occluded_fraction"], hidden / len(face_vis))
    # box_style annotation: the far eye is cut at the nose, and not labelled once no more than 1/3 is visible
    assert not any(d["class"] == "left_eye" for d in tr["detections"]) and tr["eye_occluded"]["left_eye"]
    tr = compute_truth(cabin, cam, _frame("road_ahead", cabin, yaw=55, hand="lap"))
    far = next(d for d in tr["detections"] if d["class"] == "left_eye")
    assert far["occlusion"] == 0.0 and far["self_occluded_fraction"] > 0 and tr["eye_occluded"]["left_eye"]


def test_eye_and_mouth_occlusion_are_fractions():
    cabin, cam = _setup()
    tr = compute_truth(cabin, cam, _frame("road_ahead", cabin, hand="smoke"))
    mouth = next(d for d in tr["detections"] if d["class"] == "mouth")
    n_hidden = sum(tr["occluded_by"].get(n) == "hand_smoking" for n in ("mouth_left", "mouth_right", "mouth_center"))
    assert 0 < n_hidden < 3 and tr["mouth_occluded"]
    # box_style annotation: occlusion = hidden share of the labelled mouth region (head-frame sample grid)
    assert 0 < mouth["occlusion"] < 1
    cfgs = load_configs(); cfgs["cabin"]["box_style"] = "geometric"
    geo = Cabin(cfgs["cabin"])
    mouth = next(d for d in compute_truth(geo, cam, _frame("road_ahead", geo, hand="smoke"))["detections"]
                 if d["class"] == "mouth")
    assert np.isclose(mouth["occlusion"], n_hidden / 3)


def test_steering_wheel_hides_tall_short_and_nodding_drivers_from_the_column_camera():
    cfgs = load_configs(); nominal = Cabin(cfgs["cabin"])
    cam = Camera.from_config("steering_column", cfgs["cameras"], nominal.nominal_head_center())   # fixed mount
    wheel = {"steering_wheel", "steering_wheel_hub", "steering_wheel_spoke_left", "steering_wheel_spoke_right",
             "steering_wheel_spoke_bottom"}
    tr = compute_truth(nominal, cam, _frame("road_ahead", nominal))
    assert all(tr["landmarks_visible"].values()) and not tr["occluded_by"]
    for dz in (-0.08, 0.08):
        cab = _cabin_with(eye_point_offset=[0.0, 0.0, dz])
        hit = set(compute_truth(cab, cam, _frame("road_ahead", cab))["occluded_by"].values())
        assert hit and hit <= wheel, dz
    hit = set(compute_truth(nominal, cam, _frame("road_ahead", nominal, pitch=-30))["occluded_by"].values())
    assert hit and hit <= wheel
    # a mount lower on the column looks straight through the hub
    old = Camera("old", {**cfgs["cameras"]["defaults"], "position": [0.55, 0.0, -0.25]}, nominal.nominal_head_center())
    hit = set(compute_truth(nominal, old, _frame("road_ahead", nominal))["occluded_by"].values())
    assert hit == {"steering_wheel_hub"}


def test_hand_on_wheel_occluder_follows_the_hand_state():
    cabin = Cabin(load_configs()["cabin"]); head = cabin.head_state(0, 0, 0)
    names = lambda hand: {o["name"] for o in cabin.occluder_geometry(hand, head)}
    assert "hand_on_wheel" in names("wheel") and "hand_on_wheel" not in names("lap")
    sph = next(o for o in cabin.occluder_geometry("wheel", head) if o["name"] == "hand_on_wheel")
    assert np.allclose(sph["center_world"], cabin.hand_position("wheel", head))


def test_far_eye_inner_corner_hides_first_behind_the_nose():
    cabin = Cabin(load_configs()["cabin"])
    cam = Camera("front", {"position": [0.6, 0, 0], "width": 1280, "height": 800, "hfov_deg": 90,
                           "orientation": {"yaw": 180, "pitch": 0}}, cabin.nominal_head_center())
    first = {}
    for yaw in range(0, 121):
        tr = compute_truth(cabin, cam, _frame("road_ahead", cabin, yaw=yaw, hand="lap"))
        for n, v in tr["landmarks_visible"].items():
            if not v:
                first.setdefault(n, yaw)
    assert 50 <= first["left_eye_inner"] <= 62
    assert first["left_eye_inner"] < first["left_eye_center"] < first["left_cheek"]
    assert first["left_eye_inner"] < first["left_temple"]
    assert min(first[n] for n in ("left_cheek", "left_temple")) >= 65


def test_held_cup_or_bottle_at_the_mouth_hides_the_mouth():
    import dataclasses
    cabin, cam = _setup()
    bare = compute_truth(cabin, cam, _frame("road_ahead", cabin, hand="at_mouth"))
    assert bare["held"] == "none" and not any(str(o).startswith("held_") for o in bare["occluded_by"].values())
    for obj in ("cup", "bottle"):
        f = dataclasses.replace(_frame("road_ahead", cabin, hand="at_mouth"), held=obj)
        tr = compute_truth(cabin, cam, f)
        hidden = {n for n, o in tr["occluded_by"].items() if o == f"held_{obj}"}
        assert tr["held"] == obj and tr["mouth_occluded"]
        assert "mouth_center" in hidden, (obj, hidden)                    # the thin bottle may leave a corner
        assert not {"left_eye_center", "right_eye_center"} <= hidden        # never both eyes (a low camera may lose one)


def test_held_object_does_not_reach_into_the_face():
    """The held occluder is a capsule (ray_hits_capsule): its rounded rear end must stop at the lips, so no face
    landmark lies inside it (a landmark inside would be hidden from every camera, side ones included)."""
    cabin, _ = _setup()
    for obj, spec in cabin.held_objects.items():
        for pose in ((0, 0, 0), (0, 15, 0), (30, 0, 0)):
            head = cabin.head_state(*pose)
            (oc,) = [o for o in cabin.occluder_geometry("at_mouth", head, obj) if o["name"] == f"held_{obj}"]
            for name, p in head.landmarks.items():
                d = G.seg_seg_distance(p, p, oc["a_world"], oc["b_world"])
                assert d >= oc["radius"], (obj, pose, name, d)
            # the capsule spans the configured length from the lips (rear cap to front cap)
            span = np.linalg.norm(oc["b_world"] - oc["a_world"]) + 2 * oc["radius"]
            assert np.isclose(span, spec["length"])

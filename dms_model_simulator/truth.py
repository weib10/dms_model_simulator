"""Ground truth per frame: exact geometry of what a camera at this position sees."""
from __future__ import annotations
import numpy as np
from . import geometry as G
from .behavior import DriverFrame
from .camera import Camera
from .world import Cabin

EYE_LMS = {
    "left_eye": ["left_eye_outer", "left_eye_inner", "left_eye_center"],
    "right_eye": ["right_eye_outer", "right_eye_inner", "right_eye_center"],
}
MOUTH_LMS = ["mouth_left", "mouth_right", "mouth_center"]
EAR_LMS = {"left_ear", "right_ear"}       # never in the face box (ears excluded)
JAW_LMS = {"left_jaw", "right_jaw"}       # face contour in front of the ears: face box side edges, annotation only
NEAR_M = 0.06            # near plane for polygon clipping (Camera.project rejects z <= 0.05)


def _bbox(points: list[np.ndarray], pad: float = 0.0) -> list[float] | None:
    pts = np.array([p for p in points if np.all(np.isfinite(p))])
    if len(pts) == 0:
        return None
    lo, hi = pts.min(0), pts.max(0)
    ext = hi - lo
    lo, hi = lo - pad * ext, hi + pad * ext
    return [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]


def _clip_bbox(b, w, h):
    if b is None:
        return None
    x1, y1, x2, y2 = b
    x1, x2 = np.clip([x1, x2], 0, w)
    y1, y2 = np.clip([y1, y2], 0, h)
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return [float(x1), float(y1), float(x2), float(y2)]


def _pose_in_camera(cam: Camera, R_vehicle_from_head: np.ndarray) -> dict:
    """Head pose in the camera face frame (axis based, independent of where the head is): (0,0,0) = face
    direction parallel to the camera's reversed optical axis. The presets aim at the nominal eye point, so a
    face at the nominal eye point looking into the lens reads 0."""
    R_rel = G.R_FACE_FROM_CV @ cam.R @ R_vehicle_from_head
    y, p, r = G.R_to_euler(R_rel)
    return {"yaw": y, "pitch": p, "roll": r}


def _box_around(cam: Camera, center: np.ndarray, w_m: float, h_m: float):
    uv, _ = cam.project(center)
    if not np.all(np.isfinite(uv)):
        return None
    pw, ph = cam.project_size(center, w_m), cam.project_size(center, h_m)
    return [uv[0] - pw / 2, uv[1] - ph / 2, uv[0] + pw / 2, uv[1] + ph / 2]


def _segment_box(cam: Camera, a: np.ndarray, b: np.ndarray, diameter_m: float):
    """bbox of a thin cylinder a->b: projected end points (and midpoint) padded by half the projected diameter."""
    pts = [a, 0.5 * (a + b), b]
    uvs = [cam.project(p)[0] for p in pts]
    if not all(np.all(np.isfinite(uv)) for uv in uvs):
        return None
    pad = 0.5 * max(cam.project_size(p, diameter_m) for p in pts)
    uv = np.array(uvs)
    lo, hi = uv.min(0) - pad, uv.max(0) + pad
    return [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]


def _clip_poly(poly: list[np.ndarray], inside, cross) -> list[np.ndarray]:
    """Sutherland-Hodgman against one half-space: inside(p) -> bool, cross(p, q) -> intersection point."""
    out = []
    for i, cur in enumerate(poly):
        prev = poly[i - 1]
        if inside(cur):
            if not inside(prev):
                out.append(cross(prev, cur))
            out.append(cur)
        elif inside(prev):
            out.append(cross(prev, cur))
    return out


def _lerp_at(k: int, val: float):
    """Intersection of segment p->q with the plane coordinate[k] == val."""
    def f(p, q):
        t = (val - p[k]) / (q[k] - p[k])
        return p + t * (q - p)
    return f


def _seatbelt_box(cabin: Cabin, cam: Camera) -> list[float] | None:
    """bbox of the labelled belt pieces (cabin.seatbelt_labelled_pieces); None if none is in view."""
    boxes = [b for a, c in cabin.seatbelt_labelled_pieces() if (b := _belt_piece_box(cabin, cam, a, c))]
    if not boxes:
        return None
    b = np.array(boxes)
    return [float(b[:, 0].min()), float(b[:, 1].min()), float(b[:, 2].max()), float(b[:, 3].max())]


def _belt_piece_box(cabin: Cabin, cam: Camera, a: np.ndarray, b: np.ndarray) -> list[float] | None:
    """bbox of one projected belt band a -> b, clipped to the near plane and the image; None if not in view."""
    q = cabin.seatbelt_quad(a, b)                           # TL, BL, BR, TR
    t = np.linspace(0.0, 1.0, cabin.seatbelt_edge_samples)[:, None]
    poly3 = list(q[0] + t * (q[1] - q[0])) + list(q[2] + t * (q[3] - q[2]))
    pc = _clip_poly([cam.to_cam(p) for p in poly3], lambda p: p[2] >= NEAR_M, _lerp_at(2, NEAR_M))
    if len(pc) < 3:
        return None
    uv = [cam.project(cam.R.T @ p + cam.position)[0] for p in pc]
    for k, lim in ((0, cam.width), (1, cam.height)):
        uv = _clip_poly(uv, lambda p, k=k: p[k] >= 0.0, _lerp_at(k, 0.0))
        uv = _clip_poly(uv, lambda p, k=k, lim=lim: p[k] <= lim, _lerp_at(k, float(lim)))
    if len(uv) < 3:
        return None
    return _clip_bbox(_bbox(uv), cam.width, cam.height)


def _seatbelt_occlusion(cabin: Cabin, cam: Camera, occl: list[dict]) -> float:
    """Fraction of in-image strap centreline samples whose ray to the camera hits an occluder."""
    t = np.linspace(0.0, 1.0, cabin.seatbelt_occlusion_samples)[:, None]
    pts = cabin.seatbelt_top + t * (cabin.seatbelt_bottom - cabin.seatbelt_top)
    vis = [p for p in pts if cam.project(p)[1]]
    if not vis:
        return 0.0
    return sum(cabin.ray_blocked(cam.position, p, occl, cabin.occlusion_eps) is not None for p in vis) / len(vis)


def _bounding_sphere(o: dict) -> tuple[np.ndarray, float]:
    t = o["type"]
    if t == "torus":
        c = o["polyline"].mean(0)
        return c, float(np.linalg.norm(o["polyline"] - c, axis=1).max()) + o["tube_radius"]
    if t == "cylinder":
        a, b = o["a_world"], o["b_world"]
        return 0.5 * (a + b), 0.5 * float(np.linalg.norm(b - a)) + o["radius"]
    return o["center_world"], o["radius"]                              # sphere, disc


def _near_occluders(cam_pos: np.ndarray, center: np.ndarray, reach: float, occl: list[dict]) -> list[dict]:
    """Occluders that may cross a ray from the camera to any point within `reach` of `center` (such a ray stays
    within `reach` of the segment camera -> center); the rest are skipped for speed."""
    return [o for o in occl
            if G.seg_point_distance(cam_pos, center, (bs := _bounding_sphere(o))[0]) <= bs[1] + reach]


def _region(cabin: Cabin, cam: Camera, head, occl: list[dict], self_occl: list[dict], names: list[str],
            ys: tuple[float, float], zs: tuple[float, float], x_of_y) -> dict:
    """Labelled box of a face region (box_style annotation): a head-frame grid over y in ys, z in zs (x = x_of_y(y),
    on the face surface). Points the head itself hides (facing away, or behind a self-occluder that applies to
    `names`) are cut off the box, as a labeller does for the far eye of a turned head. Returns the box (None if no
    point faces the camera), occlusion (share of the self-visible points behind an external occluder), self_frac
    (share hidden by the head) and visible (share neither)."""
    nu, nv = cabin.annotation_boxes.get("samples", [5, 3])
    # stricter than the landmark test: only two thin nose cylinders model the head, so a grid point facing away
    # must count as hidden or the far eye of a profile head reappears through the gaps
    min_cos = float(cabin.annotation_boxes.get("min_facing_cos", 0.0))
    so = [o for o in self_occl if o["applies_to"] & set(names)]
    grid = [np.array([x_of_y(y), y, z]) for y in np.linspace(*ys, int(nu)) for z in np.linspace(*zs, int(nv))]
    world = [head.head_to_vehicle(ph) for ph in grid]
    center = np.mean(world, axis=0)
    occl = _near_occluders(cam.position, center, max(np.linalg.norm(p - center) for p in world), occl)
    uv_keep, n_occ, n_self, n = [], 0, 0, 0
    for ph, p in zip(grid, world):
        n_head = ph.copy(); n_head[0] += cabin.lm_normal_forward_bias
        facing = float(np.dot(head.R @ G.normalize(n_head), G.normalize(cam.position - p))) > min_cos
        n += 1
        if not facing or cabin.ray_blocked(cam.position, p, so, cabin.occlusion_eps) is not None:
            n_self += 1
            continue
        n_occ += cabin.ray_blocked(cam.position, p, occl, cabin.occlusion_eps) is not None
        uv_keep.append(cam.project(p)[0])
    box = _clip_bbox(_bbox(uv_keep), cam.width, cam.height) if uv_keep else None
    n_vis = n - n_self
    return {"box": box, "occlusion": n_occ / n_vis if n_vis else 0.0, "self_frac": n_self / n,
            "visible": (n_vis - n_occ) / n}


def compute_truth(cabin: Cabin, cam: Camera, f: DriverFrame) -> dict:
    head = cabin.head_state(f.head_yaw, f.head_pitch, f.head_roll)
    held = getattr(f, "held", "none")
    occl = cabin.occluder_geometry(f.hand, head, held)
    hs = cabin.held_segment(held, head)          # object at the lips: no OD class, its box anchors context rules
    held_bbox = _clip_bbox(_segment_box(cam, hs[0], hs[1], 2 * hs[2]), cam.width, cam.height) if hs else None
    self_occl = cabin.self_occluder_geometry(head)
    W, H = cam.width, cam.height

    # landmarks -> image. lm_occ = external occluder (cabin / hands), lm_self = hidden by the head itself
    lm_img, lm_vis, lm_occ, lm_self = {}, {}, {}, {}
    for name, p in head.landmarks.items():
        uv, inside = cam.project(p)
        # self-occlusion: landmark normal (its head-frame direction) facing away from the camera ...
        n_head = cabin.template[name].copy(); n_head[0] += cabin.lm_normal_forward_bias
        n_world = head.R @ G.normalize(n_head)
        facing = float(np.dot(n_world, G.normalize(cam.position - p))) > cabin.lm_min_facing_cos
        # ... or behind a part of the face (nose ridge for the eye landmarks)
        behind = cabin.ray_blocked(cam.position, p, [o for o in self_occl if name in o["applies_to"]],
                                   cabin.occlusion_eps)
        blocked = cabin.ray_blocked(cam.position, p, occl, cabin.occlusion_eps)
        lm_img[name] = uv
        lm_occ[name] = blocked
        lm_self[name] = not facing or behind is not None
        lm_vis[name] = bool(inside and not lm_self[name] and blocked is None)

    face_lms = [n for n in lm_img if n not in EAR_LMS and (cabin.box_style == "annotation" or n not in JAW_LMS)]

    def occ_frac(names) -> float:
        """Fraction of the landmarks hidden by external occluders (detection 'occlusion'; the emulator's yaw
        term already models turning away, so self-occlusion is not counted twice)."""
        return sum(lm_occ[n] is not None for n in names) / len(names)

    def self_frac(names) -> float:
        return sum(lm_self[n] for n in names) / len(names)

    # eyes, openness-aware lid points
    annotated = cabin.box_style == "annotation"
    ab = cabin.annotation_boxes
    min_vis = float(ab.get("min_visible_fraction", 0.0))
    region = {}                                                   # annotation: label region stats per class
    eye_boxes, eye_occ, eye_w = {}, {}, {}
    for eye, names in EYE_LMS.items():
        c_head = cabin.template[f"{eye}_center"]
        if annotated:
            pin, pout = cabin.template[names[1]], cabin.template[names[0]]      # inner, outer corner
            x_of_y = lambda y, pin=pin, pout=pout: float(pin[0] + (pout[0] - pin[0]) * (y - pin[1]) / (pout[1] - pin[1]))
            r = _region(cabin, cam, head, occl, self_occl, names, (pin[1], pout[1]),
                        (c_head[2] - ab["eye_bottom"], c_head[2] + ab["eye_top"]), x_of_y)
            region[eye] = r
            eye_boxes[eye] = r["box"] if r["visible"] > min_vis else None
        else:
            op = f.openness_left if eye == "left_eye" else f.openness_right
            hh = cabin.eye_half_height * max(op, cabin.eye_min_openness_frac)
            lids = [head.head_to_vehicle(c_head + [0, 0, hh]), head.head_to_vehicle(c_head - [0, 0, hh])]
            pts = [lm_img[n] for n in names] + [cam.project(p)[0] for p in lids]
            eye_boxes[eye] = _clip_bbox(_bbox(pts), W, H)
        eye_occ[eye] = not all(lm_vis[n] for n in names)          # any landmark not visible (gaze validity)
        eye_w[eye] = (eye_boxes[eye][2] - eye_boxes[eye][0]) if eye_boxes[eye] else 0.0

    # mouth, openness-aware lip points (DriverFrame.mouth_openness may not exist yet -> closed)
    mouth_open = float(getattr(f, "mouth_openness", 0.0))
    c_head = cabin.template["mouth_center"]
    mh = cabin.mouth_half_height * max(mouth_open, cabin.mouth_min_openness_frac)
    if annotated:
        corner = cabin.template["mouth_left"]
        half_w = abs(corner[1] - c_head[1])
        x_of_y = lambda y: float(c_head[0] + (corner[0] - c_head[0]) * min(abs(y - c_head[1]) / half_w, 1.0))
        side = half_w + ab["mouth_side_extend"]
        r = _region(cabin, cam, head, occl, self_occl, MOUTH_LMS, (c_head[1] - side, c_head[1] + side),
                    (c_head[2] - mh - ab["mouth_below_lower_lip"], c_head[2] + ab["mouth_top"]), x_of_y)
        region["mouth"] = r
        mouth_box = r["box"] if r["visible"] > min_vis else None
    else:
        lips = [head.head_to_vehicle(c_head + [0, 0, mh]), head.head_to_vehicle(c_head - [0, 0, mh])]
        mouth_pts = [lm_img[n] for n in MOUTH_LMS] + [cam.project(p)[0] for p in lips]
        mouth_box = _clip_bbox(_bbox(mouth_pts), W, H)
    mouth_occ = not all(lm_vis[n] for n in MOUTH_LMS)
    mouth_w = (mouth_box[2] - mouth_box[0]) if mouth_box else 0.0

    if annotated:          # hairline (forehead) .. chin, jaw .. jaw (no ears); parts behind the head are not boxed
        face_pts = [lm_img[n] for n in face_lms if not lm_self[n] and np.all(np.isfinite(lm_img[n]))]
        face_box = _clip_bbox(_bbox(face_pts), W, H) if len(face_pts) >= 3 else None
    else:
        face_pts = [lm_img[n] for n in face_lms if lm_vis[n] or lm_occ[n] is not None]
        face_box = _clip_bbox(_bbox(face_pts, cabin.face_bbox_pad), W, H) if len(face_pts) >= 3 else None
    face_w = (face_box[2] - face_box[0]) if face_box else 0.0

    # head pose
    pose_cam = _pose_in_camera(cam, head.R)
    head_center = head.origin
    to_cam = cam.position - head_center
    face_yaw_rel = G.angle_between(head.R[:, 0], to_cam)   # 0 = looking straight into camera

    # gaze (exact: from eye midpoint to the gaze point)
    gaze_dir = G.normalize(f.gaze_point - head.origin)
    g_yaw_v, g_pitch_v = G.dir_to_yaw_pitch(gaze_dir)
    gaze_dir_cam = cam.dir_to_cam(gaze_dir)
    g_yaw_c, g_pitch_c = G.dir_to_yaw_pitch(G.cv_to_face(gaze_dir_cam))
    gaze_dir_head = head.R.T @ gaze_dir
    e_yaw, e_pitch = G.dir_to_yaw_pitch(gaze_dir_head)       # eye-in-head
    gaze_to_cam = G.angle_between(gaze_dir, to_cam)
    zone, zone_err = cabin.classify_zone(head.origin, gaze_dir)
    origin_uv, _ = cam.project(head.origin)
    tip_uv, _ = cam.project(head.origin + 0.15 * gaze_dir)

    dets = []
    if face_box:
        dets.append({"class": "face", "bbox": face_box, "occlusion": occ_frac(face_lms), "width_px": face_w,
                     "self_occluded_fraction": self_frac(face_lms)})
    for eye, names in EYE_LMS.items():
        if eye_boxes[eye]:
            r = region.get(eye)
            dets.append({"class": eye, "bbox": eye_boxes[eye], "width_px": eye_w[eye],
                         "occlusion": r["occlusion"] if r else occ_frac(names),
                         "self_occluded_fraction": r["self_frac"] if r else self_frac(names)})
    if mouth_box:
        r = region.get("mouth")
        dets.append({"class": "mouth", "bbox": mouth_box, "width_px": mouth_w,
                     "occlusion": r["occlusion"] if r else occ_frac(MOUTH_LMS),
                     "self_occluded_fraction": r["self_frac"] if r else self_frac(MOUTH_LMS)})
    hand_p = cabin.hand_position(f.hand, head)
    if hand_p is not None:
        b = _clip_bbox(_box_around(cam, hand_p, cabin.hand_size, cabin.hand_size), W, H)
        if b:
            dets.append({"class": "hand", "bbox": b, "occlusion": 0.0, "width_px": b[2] - b[0], "state": f.hand})
    phone_p = cabin.phone_position(f.phone, head)
    if phone_p is not None:
        b = _clip_bbox(_box_around(cam, phone_p, *cabin.phone_size), W, H)
        if b:
            dets.append({"class": "phone", "bbox": b, "occlusion": 0.0, "width_px": b[2] - b[0], "state": f.phone})
    cigarette = getattr(f, "cigarette", "none")
    seg = cabin.cigarette_segment(cigarette, f.hand, head)
    if seg is not None:
        b = _clip_bbox(_segment_box(cam, *seg, cabin.cigarette_diameter), W, H)
        if b:
            dets.append({"class": "cigarette", "bbox": b, "occlusion": 0.0, "width_px": b[2] - b[0], "state": cigarette})
    # seatbelt: the strap box is computed even when off, so the emulator can place false positives there
    seatbelt = getattr(f, "seatbelt", "on")
    strap_box = _seatbelt_box(cabin, cam)
    if strap_box and seatbelt == "on":
        dets.append({"class": "seatbelt", "bbox": strap_box, "occlusion": _seatbelt_occlusion(cabin, cam, occl),
                     "width_px": strap_box[2] - strap_box[0], "state": seatbelt})

    return {
        "head_pose_vehicle": {"yaw": f.head_yaw, "pitch": f.head_pitch, "roll": f.head_roll},
        "head_pose_camera": pose_cam,
        "head_center_vehicle": head.origin.tolist(),
        "head_center_camera": cam.to_cam(head.origin).tolist(),
        "face_yaw_rel_camera_deg": face_yaw_rel,
        "gaze": {
            "origin_vehicle": head.origin.tolist(), "dir_vehicle": gaze_dir.tolist(),
            "yaw_vehicle": g_yaw_v, "pitch_vehicle": g_pitch_v,
            "dir_camera": gaze_dir_cam.tolist(), "yaw_camera": g_yaw_c, "pitch_camera": g_pitch_c,
            "eye_in_head_yaw": e_yaw, "eye_in_head_pitch": e_pitch,
            "target": f.target_name, "zone": zone, "zone_angle_err_deg": zone_err,
            "gaze_to_camera_deg": gaze_to_cam,
            "origin_px": origin_uv.tolist(), "tip_px": tip_uv.tolist(),
        },
        "eye_openness": {"left": f.openness_left, "right": f.openness_right},
        "mouth_openness": mouth_open,
        "blink": f.blink, "closure": f.closure,
        "eye_occluded": eye_occ, "mouth_occluded": mouth_occ,
        "hand": f.hand, "phone": f.phone, "cigarette": cigarette, "seatbelt": seatbelt, "held": held,
        "held_bbox": held_bbox,
        "seatbelt_strap_bbox": strap_box,
        "detections": dets,
        "landmarks_px": {k: v.tolist() for k, v in lm_img.items()},
        "landmarks_visible": lm_vis,
        "occluded_by": {k: v for k, v in lm_occ.items() if v},
    }

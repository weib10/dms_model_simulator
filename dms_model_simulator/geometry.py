"""Rotation / projection helpers. Angles in degrees at the API boundary."""
from __future__ import annotations
import numpy as np

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


def rot_x(a_deg: float) -> np.ndarray:
    a = a_deg * D2R
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a_deg: float) -> np.ndarray:
    a = a_deg * D2R
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a_deg: float) -> np.ndarray:
    a = a_deg * D2R
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_to_R(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Forward-left-up frame. yaw>0 turns left, pitch>0 looks up, roll>0 tilts head
    towards the right shoulder (right-hand rule about +X)."""
    return rot_z(yaw) @ rot_y(-pitch) @ rot_x(roll)


def R_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of euler_to_R. Returns (yaw, pitch, roll) in degrees."""
    f = R[:, 0]
    yaw = np.arctan2(f[1], f[0]) * R2D
    pitch = np.arcsin(np.clip(f[2], -1, 1)) * R2D
    Rp = (rot_z(yaw) @ rot_y(-pitch)).T @ R
    roll = np.arctan2(Rp[2, 1], Rp[1, 1]) * R2D
    return float(yaw), float(pitch), float(roll)


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def dir_to_yaw_pitch(d: np.ndarray) -> tuple[float, float]:
    """Direction in a forward-left-up frame -> (yaw, pitch) degrees."""
    yaw = np.arctan2(d[1], d[0]) * R2D
    pitch = np.arctan2(d[2], np.hypot(d[0], d[1])) * R2D
    return float(yaw), float(pitch)


def yaw_pitch_to_dir(yaw: float, pitch: float) -> np.ndarray:
    y, p = yaw * D2R, pitch * D2R
    return np.array([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)])


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in degrees. atan2(|a x b|, a . b) instead of arccos(a . b): arccos turns a last-digit difference of a
    near-1 dot product into ~1e-6 deg, so the same input gave different angles on different maths libraries."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)) * R2D)


def cv_to_flu(v: np.ndarray) -> np.ndarray:
    """OpenCV camera vector (x right, y down, z fwd) -> forward-left-up."""
    return np.array([v[2], -v[0], -v[1]])


def flu_to_cv(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], -v[2], v[0]])


# Rotation that maps forward-left-up coordinates to OpenCV coordinates.
R_CV_FROM_FLU = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=float)

# "Face frame" of a camera: forward = towards the camera (-z), left = +x, up = -y.
# A face looking straight into the lens has yaw = pitch = roll = 0 in this frame;
# yaw > 0 means the subject turns to their own left, pitch > 0 looks up.
R_FACE_FROM_CV = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]], dtype=float)


def cv_to_face(v: np.ndarray) -> np.ndarray:
    return R_FACE_FROM_CV @ np.asarray(v, float)


def face_to_cv(v: np.ndarray) -> np.ndarray:
    return R_FACE_FROM_CV.T @ np.asarray(v, float)


# up hints tried in order by look_at when the view direction is (anti)parallel to the requested up vector
_LOOK_AT_FALLBACK_UP = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
_LOOK_AT_EPS = 1e-6        # |f x up| below this = view direction parallel to up


def look_at(cam_pos: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Return R (3x3) such that p_cam = R @ (p_world - cam_pos), OpenCV convention.
    When the view direction is parallel to `up` (camera looking straight up / down) the image "up" falls back
    to vehicle forward (+X), then left (+Y), so R stays a proper rotation."""
    f = normalize(np.asarray(target, float) - np.asarray(cam_pos, float))
    right = np.cross(f, np.asarray(up, float))
    for alt in _LOOK_AT_FALLBACK_UP:
        if np.linalg.norm(right) >= _LOOK_AT_EPS:
            break
        right = np.cross(f, np.asarray(alt, float))
    right = normalize(right)
    down = np.cross(f, right)
    return np.vstack([right, down, f])


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3 - 2 * x)


def seg_seg_distance(p1, p2, q1, q2) -> float:
    """Minimum distance between segments p1-p2 and q1-q2."""
    p1, p2, q1, q2 = (np.asarray(v, float) for v in (p1, p2, q1, q2))
    u, v, w = p2 - p1, q2 - q1, p1 - q1
    a, b, c, d, e = u @ u, u @ v, v @ v, u @ w, v @ w
    den = a * c - b * b
    sN, sD, tN, tD = 0.0, den, 0.0, den
    if den < 1e-12:
        sN, sD, tN, tD = 0.0, 1.0, e, c
    else:
        sN, tN = b * e - c * d, a * e - b * d
        if sN < 0:
            sN, tN, tD = 0.0, e, c
        elif sN > sD:
            sN, tN, tD = sD, e + b, c
    if tN < 0:
        tN = 0.0
        if -d < 0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN, sD = -d, a
    elif tN > tD:
        tN = tD
        if -d + b < 0:
            sN = 0.0
        elif -d + b > a:
            sN = sD
        else:
            sN, sD = -d + b, a
    sc = 0.0 if abs(sN) < 1e-12 else sN / sD
    tc = 0.0 if abs(tN) < 1e-12 else tN / tD
    return float(np.linalg.norm(w + sc * u - tc * v))


def seg_point_distance(p1, p2, q) -> float:
    p1, p2, q = (np.asarray(v, float) for v in (p1, p2, q))
    u = p2 - p1
    t = np.clip(((q - p1) @ u) / max(u @ u, 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(p1 + t * u - q))


# ---- ray tests (camera -> point). A primitive blocks the point only if the ray meets it strictly before the
# point (more than eps short of it), so a landmark lying on an occluder's surface is not hidden by it.
def _ray(o, p):
    o, p = np.asarray(o, float), np.asarray(p, float)
    d = p - o
    L = float(np.linalg.norm(d))
    return o, p, (d / L if L > 0 else d), L


def ray_hits_sphere(o, p, center, radius: float, eps: float = 0.0) -> bool:
    """Ray o->p enters the sphere at distance t_enter with 0 < t_enter < |p - o| - eps."""
    o, p, u, L = _ray(o, p)
    oc = o - np.asarray(center, float)
    b = float(u @ oc)
    disc = b * b - (float(oc @ oc) - radius * radius)
    if disc <= 0:
        return False
    t_enter = -b - np.sqrt(disc)
    return 0.0 < t_enter < L - eps


def ray_hits_capsule(o, p, a, b, radius: float, eps: float = 0.0) -> bool:
    """Ray o->p (stopped eps short of p) passes within radius of segment a-b (thin cylinder / polyline piece)."""
    o, p, u, L = _ray(o, p)
    if L <= eps:
        return False
    return seg_seg_distance(o, p - eps * u, a, b) < radius


def ray_hits_disc(o, p, center, normal, radius: float, eps: float = 0.0) -> bool:
    """Ray o->p crosses the disc (centre, unit normal, radius) at 0 < t < |p - o| - eps."""
    o, p, u, L = _ray(o, p)
    c, n = np.asarray(center, float), np.asarray(normal, float)
    den = float(u @ n)
    if abs(den) < 1e-12:
        return False
    t = float((c - o) @ n) / den
    if not 0.0 < t < L - eps:
        return False
    return float(np.linalg.norm(o + t * u - c)) < radius


def ray_triangles(o, d, tris) -> tuple[np.ndarray, np.ndarray]:
    """Moller-Trumbore for the ray o + t d against every triangle (tris: N x 3 x 3), vectorised: (inside, t) with
    inside[i] = the ray's line crosses triangle i, t[i] = where (in units of d)."""
    o, d = np.asarray(o, float), np.asarray(d, float)
    v0 = tris[:, 0]
    e1, e2 = tris[:, 1] - v0, tris[:, 2] - v0
    h = np.cross(d, e2)
    a = np.einsum("ij,ij->i", e1, h)
    ok = np.abs(a) > 1e-15
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    s = o - v0
    uu = f * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    vv = f * (q @ d)
    return ok & (uu >= 0) & (vv >= 0) & (uu + vv <= 1), f * np.einsum("ij,ij->i", e2, q)


def ray_hit_triangles(o, p, tris, eps: float = 0.0) -> int:
    """Index of the triangle (tris: N x 3 x 3) the ray o->p meets first, more than eps short of p (a point on a
    surface is not hidden by it), or -1."""
    o, p, u, L = _ray(o, p)
    if L <= eps:
        return -1
    inside, t = ray_triangles(o, p - o, tris)              # t: 0 at o, 1 at p
    m = inside & (t > 1e-9) & (t < 1.0 - eps / L)
    if not m.any():
        return -1
    idx = np.flatnonzero(m)
    return int(idx[np.argmin(t[idx])])


def triangles_from(o, tris, two_sided, group=None) -> dict:
    """Per-origin precomputation for ray_hit_triangles_from: the triangles (N x 3 x 3) a ray from o can enter
    through (front side towards o, or two_sided), with the Moller-Trumbore terms that depend on o only. group: an
    integer per triangle (the part it belongs to); a ray then tests only the triangles of the groups whose bounding
    box it crosses."""
    o = np.asarray(o, float)
    v0 = tris[:, 0]
    e1, e2 = tris[:, 1] - v0, tris[:, 2] - v0
    n = np.cross(e1, e2)
    sel = np.flatnonzero(np.asarray(two_sided, bool) | (np.einsum("ij,ij->i", n, o - v0) > 0))
    e1, e2, s = e1[sel], e2[sel], o - v0[sel]
    q = np.cross(s, e1)
    # Moller-Trumbore with direction d: a = e1.(d x e2) = d.(e2 x e1), u*a = s.(d x e2) = d.(e2 x s), v*a = q.d,
    # t*a = e2.q -- three rows per triangle, so a ray costs one matrix-vector product
    M = np.stack([np.cross(e2, e1), np.cross(e2, s), q], axis=1)          # N x 3 x 3
    g = np.zeros(len(tris), int) if group is None else np.asarray(group, int)
    gs = g[sel]
    groups = np.unique(gs)
    pts = tris[sel]
    lo = np.array([pts[gs == k].reshape(-1, 3).min(0) for k in groups]).reshape(-1, 3)
    hi = np.array([pts[gs == k].reshape(-1, 3).max(0) for k in groups]).reshape(-1, 3)
    pos = np.searchsorted(groups, gs)                    # each selected triangle's row in lo / hi
    return {"idx": sel, "M": M, "e2q": np.einsum("ij,ij->i", e2, q), "lo": lo, "hi": hi, "pos": pos}


def ray_hit_triangles_from(o, p, pre: dict, eps: float = 0.0) -> int:
    """ray_hit_triangles with the terms of triangles_from(o, ...): index into the original triangles, or -1.
    Only the triangles facing o are tested: a ray from outside a closed part enters it through one of those (so o
    must not lie inside a part: tests/test_body.py keeps every camera clear of the body)."""
    o, p, _, L = _ray(o, p)
    if L <= eps or not len(pre["idx"]):
        return -1
    d = p - o
    # slab test: which groups' bounding boxes the segment o -> p crosses
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1, t2 = (pre["lo"] - o) * inv, (pre["hi"] - o) * inv
    t1 = np.where(np.isnan(t1), -np.inf, t1)
    t2 = np.where(np.isnan(t2), np.inf, t2)
    near, far = np.minimum(t1, t2).max(1), np.maximum(t1, t2).min(1)
    crossed = (near <= far) & (far >= 0.0) & (near <= 1.0)
    if not crossed.any():
        return -1
    k = np.flatnonzero(crossed[pre["pos"]])
    if not len(k):
        return -1
    a, ua, va = (pre["M"][k] @ d).T
    ok = np.abs(a) > 1e-15
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    uu, vv, t = f * ua, f * va, f * pre["e2q"][k]
    m = ok & (uu >= 0) & (vv >= 0) & (uu + vv <= 1) & (t > 1e-9) & (t < 1.0 - eps / L)
    if not m.any():
        return -1
    j = np.flatnonzero(m)
    return int(pre["idx"][k[j[np.argmin(t[j])]]])


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation between unit vectors a (t = 0) and b (t = 1): constant angular speed in t."""
    a, b = normalize(np.asarray(a, float)), normalize(np.asarray(b, float))
    om = float(np.arccos(np.clip(a @ b, -1.0, 1.0)))
    if om < 1e-9:
        return normalize((1 - t) * a + t * b)
    return (np.sin((1 - t) * om) * a + np.sin(t * om) * b) / np.sin(om)


def ray_hits_polyline(o, p, pts, radius: float, eps: float = 0.0) -> bool:
    """ray_hits_capsule against every piece of the polyline pts (N x 3) at once (closest points of two
    segments, Ericson's clamped solution, vectorised over the pieces)."""
    o, p, u, L = _ray(o, p)
    if L <= eps:
        return False
    pts = np.asarray(pts, float)
    d1 = (p - eps * u) - o                       # ray piece
    q, d2 = pts[:-1], pts[1:] - pts[:-1]         # polyline pieces
    r = o - q
    a = float(d1 @ d1)
    e = np.einsum("ij,ij->i", d2, d2)
    f = np.einsum("ij,ij->i", d2, r)
    c = r @ d1
    b = d2 @ d1
    den = a * e - b * b
    s = np.where(den > 1e-12, np.clip((b * f - c * e) / np.where(den > 1e-12, den, 1.0), 0.0, 1.0), 0.0)
    t = (b * s + f) / np.maximum(e, 1e-12)
    s = np.where(t < 0, np.clip(-c / a, 0.0, 1.0), np.where(t > 1, np.clip((b - c) / a, 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)
    diff = (o + s[:, None] * d1) - (q + t[:, None] * d2)
    return bool((np.einsum("ij,ij->i", diff, diff) < radius * radius).any())

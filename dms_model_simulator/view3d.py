"""Cabin layout as one self-contained HTML page with a 3D view of a generic car interior (`dms-sim view3d`).

Solid surfaces (configs/body.yaml: shell, dashboard, seats, a seated driver) plus what configs/cabin.yaml and
configs/cameras.yaml put where: gaze targets with the zone tolerance cone as a ring, cameras with their field of
view, steering wheel, seat belt and the head at rest. The body file is display only: nothing in the simulation
reads it. No model output, no time axis.

The page is one HTML file: the scene is inlined as JSON and drawn with raw WebGL plus a 2D canvas for labels (no
libraries, no network, no fetch), so it opens straight from disk. Same input -> byte-identical file.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from . import geometry as G
from .camera import Camera
from .config import CONFIG_DIR, load_yaml
from .world import Cabin

TEMPLATE = Path(__file__).with_name("view3d_template.html")
BODY = CONFIG_DIR / "body.yaml"
FILE = "cabin3d.html"
NEAR_M = 3.0            # targets further than this from the eye point are drawn as clipped direction rays
RAY_M = 2.0             # length of such a ray
FRUSTUM_M = 0.30        # depth of the drawn field-of-view pyramid
HEAD_R_M = 0.09         # head sphere radius (same as the 2D cabin views)
CAM_BODY_M = 0.035      # size of the little camera box
WHEEL_TUBE_M = 0.016    # rim tube radius when the cabin gives none
WHEEL_COLOR = "#24282e"
HUB_COLOR = "#2f343b"
BELT_COLOR = "#4f7f5a"
HEAD_COLOR = "skin"     # body.yaml colour name (this default when the body file has none)
SKIN_DEFAULT = "#c9ced6"
EYE_COLOR = "#1f2937"
RING_N = 40             # points per zone-tolerance ring
SEG = 16                # sides of a cylinder / tube
DIGITS = 4


def _r(v):
    """Rounded plain floats (lists nest)."""
    return (np.round(np.asarray(v, float), DIGITS) + 0.0).tolist()          # + 0.0: no "-0.0"


def _rgb(c: str) -> list[float]:
    c = c.lstrip("#")
    return [round(int(c[i:i + 2], 16) / 255.0, 4) for i in (0, 2, 4)]


# ---- triangle builders: each returns a list of (p0, p1, p2), counter-clockwise seen from the front ----
def _tri_facing(p0, p1, p2, toward):
    n = np.cross(p1 - p0, p2 - p0)
    return (p0, p1, p2) if np.dot(n, toward - (p0 + p1 + p2) / 3) >= 0 else (p0, p2, p1)


def quad(points, toward) -> list:
    """Two triangles of a planar quad, front side towards the point `toward` (None: as given)."""
    p = [np.asarray(q, float) for q in points]
    tris = [(p[0], p[1], p[2]), (p[0], p[2], p[3])]
    return tris if toward is None else [_tri_facing(*t, np.asarray(toward, float)) for t in tris]


def rot_y(deg: float) -> np.ndarray:
    """About Y; deg > 0 tips +Z towards -X (a seat back leaning backwards)."""
    t = np.radians(deg)
    return np.array([[np.cos(t), 0, -np.sin(t)], [0, 1, 0], [np.sin(t), 0, np.cos(t)]])


def box(center, size, R=None) -> list:
    """Closed box, faces outwards. R: rotation (columns = box axes in the vehicle frame)."""
    c, h = np.asarray(center, float), 0.5 * np.asarray(size, float)
    R = np.eye(3) if R is None else np.asarray(R, float)
    tris = []
    for ax in range(3):
        for s in (-1, 1):
            u, v = [i for i in range(3) if i != ax]
            corners = []
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                q = np.zeros(3); q[ax] = s * h[ax]; q[u] = a * h[u]; q[v] = b * h[v]
                corners.append(c + R @ q)
            fc = c + R @ (np.eye(3)[ax] * s * h[ax])
            tris += quad(corners, 2 * fc - c)
    return tris


def _frame(axis):
    a = G.normalize(np.asarray(axis, float))
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = G.normalize(np.cross(a, ref))
    return a, u, np.cross(a, u)


def cylinder(a, b, radius, caps=True) -> list:
    a, b = np.asarray(a, float), np.asarray(b, float)
    _, u, v = _frame(b - a)
    ang = np.linspace(0, 2 * np.pi, SEG + 1)
    ring = [np.cos(t) * u + np.sin(t) * v for t in ang]
    tris = []
    for i in range(SEG):
        tris += quad([a + radius * ring[i], a + radius * ring[i + 1], b + radius * ring[i + 1], b + radius * ring[i]],
                     (a + b) / 2 + 3 * radius * (ring[i] + ring[i + 1]))
    if caps:
        for c, out in ((a, 2 * a - b), (b, 2 * b - a)):
            tris += [_tri_facing(c, c + radius * ring[i], c + radius * ring[i + 1], out) for i in range(SEG)]
    return tris


def sphere(c, r, nlat=12, nlon=20) -> list:
    c = np.asarray(c, float)

    def P(i, j):
        th, ph = np.pi * i / nlat, 2 * np.pi * j / nlon
        return c + r * np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    tris = []
    for i in range(nlat):
        for j in range(nlon):
            q = [P(i, j), P(i + 1, j), P(i + 1, j + 1), P(i, j + 1)]
            mid = sum(q) / 4
            tris += [t for t in quad(q, 2 * mid - c) if np.linalg.norm(np.cross(t[1] - t[0], t[2] - t[0])) > 1e-12]
    return tris


def tube(polyline, radius) -> list:
    """Tube along a polyline (the steering-wheel rim)."""
    pts = [np.asarray(p, float) for p in polyline]
    tris = []
    for p, q in zip(pts[:-1], pts[1:]):
        tris += cylinder(p, q, radius, caps=False)
    return tris


def panel(center, eye, w, h, lift=0.0) -> list:
    """Flat w x h rectangle at `center` facing `eye` (width horizontal), moved `lift` towards the eye."""
    n = G.normalize(np.asarray(eye, float) - np.asarray(center, float))
    wa = G.normalize(np.cross([0.0, 0.0, 1.0], n)) if abs(n[2]) < 0.99 else np.array([0.0, 1.0, 0.0])
    ha = np.cross(n, wa)
    c = np.asarray(center, float) + lift * n
    return quad([c + sw * w / 2 * wa + sh * h / 2 * ha for sw, sh in ((-1, -1), (1, -1), (1, 1), (-1, 1))], c + n)


def _ring(center: np.ndarray, normal: np.ndarray, radius: float) -> list:
    _, a, b = _frame(normal)
    ang = np.linspace(0, 2 * np.pi, RING_N + 1)
    return _r(center + radius * (np.outer(np.cos(ang), a) + np.outer(np.sin(ang), b)))


def frustum_corners(cam: Camera, depth: float = FRUSTUM_M) -> np.ndarray:
    """Corners (4x3, vehicle frame) of the camera's pinhole field of view `depth` metres along the optical axis:
    image top-left, top-right, bottom-right, bottom-left. Rows of cam.R are the camera axes (x right, y down, z
    optical axis) in the vehicle frame."""
    tx, ty = np.tan(np.radians(cam.hfov) / 2), np.tan(np.radians(cam.vfov) / 2)
    x, y, z = cam.R
    return np.array([cam.position + depth * (z + sx * tx * x + sy * ty * y)
                     for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


# ---- scene ----
class _Meshes:
    """Triangles grouped by (layer, colour, alpha, two-sided)."""

    def __init__(self, colors: dict):
        self.colors, self.groups = colors, {}

    def add(self, tris, layer: str, color: str, alpha: float = 1.0, two_sided: bool = False):
        col = self.colors.get(color, color)
        self.groups.setdefault((layer, col, float(alpha), bool(two_sided)), []).extend(tris)

    def out(self) -> list:
        return [{"layer": layer, "color": _rgb(col), "alpha": alpha, "two_sided": two,
                 "pos": _r(np.array(tris).reshape(-1))}
                for (layer, col, alpha, two), tris in self.groups.items() if tris]


def body_meshes(body: dict, cabin: Cabin, M: _Meshes) -> None:
    """configs/body.yaml parts as triangles."""
    toward = np.asarray(body.get("toward", [0.0, 0.0, 0.0]), float)
    eye = cabin.eye_point
    for p in body.get("parts", []):
        t, layer, color = p["type"], p.get("layer", "body"), p.get("color", "trim")
        alpha = float(p.get("alpha", 1.0))
        if t == "quad":
            shell = p.get("shell", True)
            M.add(quad(p["points"], toward if shell else None), layer, color, alpha, two_sided=not shell)
        elif t == "box":
            M.add(box(p["center"], p["size"], rot_y(float(p.get("pitch_deg", 0.0)))), layer, color, alpha)
        elif t == "cylinder":
            M.add(cylinder(p["a"], p["b"], float(p["radius"])), layer, color, alpha)
        elif t == "panel":
            w, h = p["size"]
            tg = cabin.targets[p["target"]]
            M.add(panel(tg, eye, w, h, lift=0.003), layer, color, alpha, two_sided=True)
            if p.get("back_x") is not None and float(p["back_x"]) > tg[0]:
                # housing behind the screen, tilted with it, reaching back to the dash face (x = back_x)
                n = G.normalize(eye - tg)
                wa = G.normalize(np.cross([0.0, 0.0, 1.0], n))
                depth = (float(p["back_x"]) - tg[0]) / max(-n[0], 0.2)
                M.add(box(tg - n * depth / 2, [depth, w, h], np.column_stack([n, wa, np.cross(n, wa)])),
                      layer, body.get("housing_color", "dash"))
        elif t == "housing":
            tg = cabin.targets[p["target"]]
            c = tg + np.asarray(p.get("offset", [0.0, 0.0, 0.0]), float)
            sz = p["size"]
            M.add(box(c, sz), layer, color, alpha)
            if p.get("glass"):
                M.add(panel(tg, eye, 0.85 * sz[1], 0.75 * sz[2], lift=0.004), layer, p["glass"], 1.0, two_sided=True)
            if p.get("arm_to") is not None:
                M.add(cylinder(c, p["arm_to"], 0.012), layer, color, alpha)
        else:
            raise ValueError(f"body part {p.get('name')!r}: unknown type {t!r} (quad | box | cylinder | panel | housing)")


def scene(cabin: Cabin, cameras: list[Camera], body: dict | None = None) -> dict:
    """Everything the page draws, as plain lists (vehicle frame, metres)."""
    body = load_yaml(BODY) if body is None else body
    M = _Meshes({HEAD_COLOR: SKIN_DEFAULT, **body.get("colors", {})})
    body_meshes(body, cabin, M)
    eye, tol = cabin.eye_point, cabin.zone_tol
    head = cabin.head_state(0.0, 0.0, 0.0)
    for oc in cabin.occluder_geometry("none", head):           # steering wheel (hand occluders are not drawn)
        if "active_when_hand" in oc:
            continue
        if oc["type"] == "torus":
            M.add(tube(oc["polyline"], float(oc.get("tube_radius", WHEEL_TUBE_M))), "body", WHEEL_COLOR)
        elif oc["type"] == "disc":
            c, r = oc["center_world"], float(oc["radius"])
            _, u, v = _frame(oc["normal_world"])
            ang = np.linspace(0, 2 * np.pi, 2 * SEG + 1)
            M.add([(c, c + r * (np.cos(a0) * u + np.sin(a0) * v), c + r * (np.cos(a1) * u + np.sin(a1) * v))
                   for a0, a1 in zip(ang[:-1], ang[1:])], "body", HUB_COLOR, two_sided=True)
        elif oc["type"] == "cylinder":
            M.add(cylinder(oc["a_world"], oc["b_world"], float(oc["radius"])), "body", WHEEL_COLOR)
    hc = cabin.nominal_head_center()
    M.add(sphere(hc, HEAD_R_M), "driver", HEAD_COLOR)
    if "nose_tip" in head.landmarks:
        M.add(sphere(head.landmarks["nose_tip"], 0.018, 6, 10), "driver", HEAD_COLOR)
    for k in ("left_eye_center", "right_eye_center"):
        if k in head.landmarks:
            M.add(sphere(head.landmarks[k] + np.array([0.004, 0.0, 0.0]), 0.009, 6, 10), "driver", EYE_COLOR)
    for a, b in cabin.seatbelt_labelled_pieces():
        M.add(quad(cabin.seatbelt_quad(a, b), None), "belt", BELT_COLOR, two_sided=True)

    targets = []
    for name, p in cabin.targets.items():
        d = p - eye
        far = float(np.linalg.norm(p)) > NEAR_M
        u = G.normalize(d)
        shown = eye + RAY_M * u if far else p
        r = float(np.linalg.norm(shown - eye))
        targets.append({"name": name, "p": _r(p), "shown": _r(shown), "far": far,
                        "dist": round(float(np.linalg.norm(d)), 2),
                        "ring": None if far else _ring(shown, u, r * np.tan(np.radians(tol)))})
    cams = []
    for cam in cameras:
        yaw, pitch = G.dir_to_yaw_pitch(cam.optical_axis)
        cams.append({"name": cam.name, "p": _r(cam.position), "corners": _r(frustum_corners(cam)),
                     "R": _r(cam.R), "hfov": round(cam.hfov, 3), "vfov": round(float(cam.vfov), 3),
                     "w": cam.width, "h": cam.height, "yaw": round(float(yaw), 1), "pitch": round(float(pitch), 1),
                     "mesh": _r(np.array(box(cam.position, [CAM_BODY_M] * 3, cam.R.T)).reshape(-1))})
    # the passenger sits mirrored about the car's centreline (midway between the side mirrors, else 0.36 m right)
    t = cabin.targets
    centre_y = 0.5 * (t["left_mirror"][1] + t["right_mirror"][1]) if {"left_mirror", "right_mirror"} <= set(t) else -0.36
    return {
        "meshes": M.out(), "targets": targets, "cameras": cams, "zone_tol_deg": tol,
        "eye_point": _r(eye), "head": {"c": _r(hc), "r": HEAD_R_M},
        "passenger_eye": _r(eye + np.array([0.0, 2 * (centre_y - eye[1]), 0.0])),
    }


def page(scene_data: dict, sources: tuple[str, ...] = ("configs/cabin.yaml", "configs/cameras.yaml",
                                                       "configs/body.yaml")) -> str:
    """The HTML document. `sources`: names of the config files, printed in the page footer."""
    blob = json.dumps({"scene": scene_data, "sources": list(sources)}, sort_keys=True, separators=(",", ":"))
    html = TEMPLATE.read_text(encoding="utf-8")
    return html.replace("/*{{DATA}}*/null", blob.replace("</", "<\\/"))


def write(out_dir, cabin: Cabin, cameras: list[Camera], body: dict | None = None,
          sources: tuple[str, ...] = ("configs/cabin.yaml", "configs/cameras.yaml", "configs/body.yaml")) -> Path:
    """Write FILE into out_dir. Returns its path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / FILE
    p.write_text(page(scene(cabin, cameras, body), sources), encoding="utf-8", newline="\n")
    return p

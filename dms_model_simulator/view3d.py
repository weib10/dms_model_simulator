"""Cabin layout as one self-contained HTML page with a 3D view of a generic car interior (`dms-sim view3d`).

Only what the simulation has: the car body (configs/body.yaml, which hides what is behind it from a camera,
body.py), gaze targets with the zone tolerance cone as a ring, cameras with their field of view, the steering
wheel, the resting hand, the seat belt and the head at rest. The head is drawn as the simulation has it: its
landmark points, with the face outline, eyes, nose and mouth filled in as the 2D camera render draws them. No model
output, no time axis.

The page is one HTML file: the scene is inlined as JSON and drawn with raw WebGL plus a 2D canvas for labels (no
libraries, no network, no fetch), so it opens straight from disk. Same input -> byte-identical file.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from . import geometry as G
from .body import SEG, box, cylinder, quad, rot_y  # noqa: F401  (rot_y: re-exported for the tests)
from .body import frame as _frame
from .body import parts as body_parts
from .camera import Camera
from .config import CONFIG_DIR, load_yaml
from .world import Cabin

TEMPLATE = Path(__file__).with_name("view3d_template.html")
BODY = CONFIG_DIR / "body.yaml"
FILE = "cabin3d.html"
NEAR_M = 3.0            # targets further than this from the eye point are drawn as clipped direction rays
RAY_M = 2.0             # length of such a ray
FRUSTUM_M = 0.30        # depth of the drawn field-of-view pyramid
CAM_BODY_M = 0.035      # size of the little camera box
WHEEL_TUBE_M = 0.016    # rim tube radius when the cabin gives none
WHEEL_COLOR = "#24282e"
HUB_COLOR = "#2f343b"
BELT_COLOR = "#4f7f5a"
HAND_COLOR = "#b08d7a"  # the hand occluders of the resting hand state (configs/behavior.yaml hand)
FACE_COLOR = "#3c373c"   # the face in the 2D camera render's colours (viz.py face fill, eyes, nose / mouth)
FACE_EYE_COLOR = "#f0f0f0"
FACE_LINE_COLOR = "#aaaab4"
FACE_LINE_M = 0.003      # nose / mouth line radius
FACE_EYE_R_M = 0.008
RING_N = 40             # points per zone-tolerance ring
DIGITS = 4


def _r(v):
    """Rounded plain floats (lists nest)."""
    return (np.round(np.asarray(v, float), DIGITS) + 0.0).tolist()          # + 0.0: no "-0.0"


def _rgb(c: str) -> list[float]:
    c = c.lstrip("#")
    return [round(int(c[i:i + 2], 16) / 255.0, 4) for i in (0, 2, 4)]


def body_meshes(body: dict, cabin: Cabin, M: _Meshes) -> None:
    """configs/body.yaml parts (body.parts) as page meshes."""
    for p in body_parts(body, cabin.targets, cabin.eye_point):
        M.add(p["tris"], p["layer"], p["color"], p["alpha"], p["two_sided"])


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


def _face_forward(cabin: Cabin, head) -> np.ndarray:
    """Unit vector out of the face: face_outline centre -> nose tip (+X when either is missing)."""
    lm = head.landmarks
    ring = [np.asarray(lm[k], float) for k in cabin.face_outline if k in lm]
    if "nose_tip" not in lm or not ring:
        return np.array([1.0, 0.0, 0.0])
    return G.normalize(np.asarray(lm["nose_tip"], float) - np.mean(ring, axis=0))


def face_meshes(cabin: Cabin, head, M: _Meshes) -> None:
    """The face the simulation projects into a camera, as the 2D render draws it: the face_outline landmarks as a
    filled outline, the eye centres, nose bridge -> tip and the mouth corners through its centre."""
    lm = head.landmarks
    ring = [np.asarray(lm[k], float) for k in cabin.face_outline if k in lm]
    if len(ring) >= 3:
        c = np.mean(ring, axis=0)
        M.add([(c, ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring))], "driver", FACE_COLOR,
              two_sided=True)
    fwd = _face_forward(cabin, head)
    for k in ("left_eye_center", "right_eye_center"):
        if k in lm:
            M.add(sphere(np.asarray(lm[k], float) + 0.006 * fwd, FACE_EYE_R_M, 6, 10), "driver", FACE_EYE_COLOR)
    for line in (("nose_bridge", "nose_tip"), ("mouth_left", "mouth_center", "mouth_right")):
        if all(k in lm for k in line):
            M.add(tube([np.asarray(lm[k], float) + 0.004 * fwd for k in line], FACE_LINE_M), "driver",
                  FACE_LINE_COLOR)


def scene(cabin: Cabin, cameras: list[Camera], body: dict | None = None, hand: str | None = None) -> dict:
    """Everything the page draws, as plain lists (vehicle frame, metres). body: the cabin's own (what the
    simulation uses) unless given. hand: the hand state drawn (default the resting one, configs/behavior.yaml
    hand), with its hand / forearm occluders."""
    body = cabin.body if body is None else body
    hand = load_yaml(CONFIG_DIR / "behavior.yaml")["hand"] if hand is None else hand
    M = _Meshes(body.get("colors", {}))
    body_meshes(body, cabin, M)
    eye, tol = cabin.eye_point, cabin.zone_tol
    head = cabin.head_state(0.0, 0.0, 0.0)
    for oc in cabin.occluder_geometry(hand, head):            # steering wheel, and the hand at rest
        col = HAND_COLOR if "active_when_hand" in oc else WHEEL_COLOR
        if oc["type"] == "sphere":
            M.add(sphere(oc["center_world"], float(oc["radius"])), "driver", col)
        elif oc["type"] == "torus":
            M.add(tube(oc["polyline"], float(oc.get("tube_radius", WHEEL_TUBE_M))), "body", WHEEL_COLOR)
        elif oc["type"] == "disc":
            c, r = oc["center_world"], float(oc["radius"])
            _, u, v = _frame(oc["normal_world"])
            ang = np.linspace(0, 2 * np.pi, 2 * SEG + 1)
            M.add([(c, c + r * (np.cos(a0) * u + np.sin(a0) * v), c + r * (np.cos(a1) * u + np.sin(a1) * v))
                   for a0, a1 in zip(ang[:-1], ang[1:])], "body", HUB_COLOR, two_sided=True)
        elif oc["type"] == "cylinder":
            M.add(cylinder(oc["a_world"], oc["b_world"], float(oc["radius"])), "driver" if "active_when_hand" in oc
                  else "body", col)
    hc = cabin.nominal_head_center()
    face_meshes(cabin, head, M)
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
        "eye_point": _r(eye), "head": {"c": _r(hc), "fwd": _r(_face_forward(cabin, head))},
        "landmarks": [{"name": k, "p": _r(v)} for k, v in sorted(head.landmarks.items())],
        "passenger_eye": _r(eye + np.array([0.0, 2 * (centre_y - eye[1]), 0.0])),
    }


def page(scene_data: dict, sources: tuple[str, ...] = ("configs/cabin.yaml", "configs/cameras.yaml",
                                                       "configs/body.yaml")) -> str:
    """The HTML document. `sources`: names of the config files, printed in the page footer."""
    blob = json.dumps({"scene": scene_data, "sources": list(sources)}, sort_keys=True, separators=(",", ":"))
    html = TEMPLATE.read_text(encoding="utf-8")
    return html.replace("/*{{DATA}}*/null", blob.replace("</", "<\\/"))


def write(out_dir, cabin: Cabin, cameras: list[Camera],
          sources: tuple[str, ...] = ("configs/cabin.yaml", "configs/cameras.yaml", "configs/body.yaml")) -> Path:
    """Write FILE into out_dir. Returns its path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / FILE
    p.write_text(page(scene(cabin, cameras), sources), encoding="utf-8", newline="\n")
    return p

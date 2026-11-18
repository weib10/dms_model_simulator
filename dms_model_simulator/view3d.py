"""Cabin layout as one self-contained HTML page with a 3D view the reader can turn (`dms-sim view3d`).

Shows what configs/cabin.yaml and configs/cameras.yaml put where: gaze targets (with the zone tolerance cone as a
ring), cameras (position, field-of-view pyramid, optical axis), steering wheel, seat belt and the driver's head at
rest. No model output, no time axis: layout only.

The page is one HTML file with the scene inlined as JSON and a small hand-written canvas renderer (no libraries, no
network, no fetch), so it opens straight from disk. Same input -> byte-identical file (sorted keys, rounded floats).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from . import geometry as G
from .camera import Camera
from .world import Cabin

TEMPLATE = Path(__file__).with_name("view3d_template.html")
FILES = {"zh": "cabin3d.html", "en": "cabin3d.en.html"}
NEAR_M = 3.0            # targets further than this from the eye point are drawn as clipped direction rays
RAY_M = 2.5             # length of such a ray
FRUSTUM_M = 0.35        # depth of the drawn field-of-view pyramid
HEAD_R_M = 0.09         # head sphere radius (same as the 2D cabin views)
FACE_LEN_M = 0.22       # face direction arrow from the head centre
RING_N = 32             # points per zone-tolerance ring
DIGITS = 4

TEXT = {
    "zh": {
        "lang": "zh-Hant", "title": "車內佈置 3D 檢視",
        "hint": "拖曳：旋轉　滾輪／雙指：縮放　Shift＋拖曳或右鍵拖曳：平移　游標停在點上：看座標",
        "views": "視角", "v_iso": "斜後方", "v_top": "俯視", "v_side": "右側", "v_front": "前方往後看",
        "v_driver": "駕駛眼睛", "from_cam": "從相機看",
        "layers": "圖層", "l_targets": "視線目標", "l_rings": "區域容差（±{tol}°）", "l_cams": "相機",
        "l_fov": "全部相機的視野", "l_wheel": "方向盤", "l_belt": "安全帶", "l_head": "頭（靜止姿勢）",
        "l_grid": "地板格線與方位", "cameras": "相機（點名稱：標出它的視野）",
        "frame": "座標：原點＝駕駛眼點，X 向前、Y 向左、Z 向上，單位公尺。",
        "front": "前", "rear": "後", "left": "左", "right": "右",
        "far": "{name}（{dist:.0f} m 外，只畫方向）",
        "cam_info": "{name}：位置 ({x}, {y}, {z})，水平視角 {hfov}°，垂直 {vfov}°，{w}×{h}",
        "note": "視野錐只畫針孔模型，沒畫鏡頭畸變。遠處目標（道路）只畫方向，長度不是真的距離。",
        "src": "來源：{cabin}、{cameras}",
    },
    "en": {
        "lang": "en", "title": "Cabin layout, 3D view",
        "hint": "Drag: turn   Wheel / pinch: zoom   Shift-drag or right-drag: pan   Hover a point: coordinates",
        "views": "View", "v_iso": "Rear 3/4", "v_top": "Top", "v_side": "Right side", "v_front": "Front, looking back",
        "v_driver": "Driver's eyes", "from_cam": "From camera",
        "layers": "Layers", "l_targets": "Gaze targets", "l_rings": "Zone tolerance (±{tol}°)", "l_cams": "Cameras",
        "l_fov": "Field of view of every camera", "l_wheel": "Steering wheel", "l_belt": "Seat belt",
        "l_head": "Head (rest pose)", "l_grid": "Floor grid and directions",
        "cameras": "Cameras (click a name to show its field of view)",
        "frame": "Frame: origin = driver eye point, X forward, Y left, Z up, metres.",
        "front": "FRONT", "rear": "REAR", "left": "LEFT", "right": "RIGHT",
        "far": "{name} ({dist:.0f} m away, direction only)",
        "cam_info": "{name}: position ({x}, {y}, {z}), horizontal FOV {hfov}°, vertical {vfov}°, {w}×{h}",
        "note": "Field-of-view pyramids are pinhole only (no lens distortion). Far targets (road) show the direction "
                "only; their length is not the distance.",
        "src": "Source: {cabin}, {cameras}",
    },
}


def _r(v):
    """Rounded plain floats (lists nest)."""
    a = np.round(np.asarray(v, float), DIGITS) + 0.0          # + 0.0: no "-0.0"
    return a.tolist()


def _ring(center: np.ndarray, normal: np.ndarray, radius: float) -> list:
    n = G.normalize(normal)
    a = G.normalize(np.cross(n, [0.0, 0.0, 1.0])) if abs(n[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
    b = np.cross(n, a)
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


def scene(cabin: Cabin, cameras: list[Camera]) -> dict:
    """Everything the page draws, as plain lists (vehicle frame, metres)."""
    eye = cabin.eye_point
    tol = cabin.zone_tol
    targets = []
    for name, p in cabin.targets.items():
        d = p - eye
        dist = float(np.linalg.norm(d))
        far = float(np.linalg.norm(p)) > NEAR_M
        u = G.normalize(d)
        shown = eye + RAY_M * u if far else p
        r = float(np.linalg.norm(shown - eye))
        targets.append({"name": name, "p": _r(p), "shown": _r(shown), "far": far, "dist": round(dist, 2),
                        # the tolerance cone as a ring around a near target (around a far one's clipped ray end
                        # it would be a large circle that says nothing about the road)
                        "ring": None if far else _ring(shown, u, r * np.tan(np.radians(tol)))})
    cams = []
    for cam in cameras:
        yaw, pitch = G.dir_to_yaw_pitch(cam.optical_axis)
        cams.append({"name": cam.name, "p": _r(cam.position), "corners": _r(frustum_corners(cam)),
                     "axis": _r(cam.position + FRUSTUM_M * 1.4 * cam.optical_axis),
                     "R": _r(cam.R), "hfov": round(cam.hfov, 3), "vfov": round(float(cam.vfov), 3),
                     "w": cam.width, "h": cam.height, "yaw": round(float(yaw), 1), "pitch": round(float(pitch), 1)})
    head = cabin.head_state(0.0, 0.0, 0.0)
    wheel = {"rim": [], "hub": [], "spokes": []}
    for oc in cabin.occluder_geometry("none", head):
        if "active_when_hand" in oc:
            continue
        if oc["type"] == "torus":
            wheel["rim"].append(_r(oc["polyline"]))
        elif oc["type"] == "disc":
            wheel["hub"].append(_ring(oc["center_world"], oc["normal_world"], float(oc["radius"])))
        elif oc["type"] == "cylinder":
            wheel["spokes"].append([_r(oc["a_world"]), _r(oc["b_world"])])
    hc = cabin.nominal_head_center()
    face = [head.landmarks[k] for k in cabin.face_outline] + [head.landmarks[cabin.face_outline[0]]]
    eyes = [head.landmarks[k] for k in ("left_eye_center", "right_eye_center") if k in head.landmarks]
    belt = cabin.seatbelt_pillar + [cabin.seatbelt_top, cabin.seatbelt_bottom]
    pts = [t["shown"] for t in targets if not t["far"]] + [c["p"] for c in cams] + [_r(hc)]
    pts += [q for poly in wheel["rim"] for q in poly] + _r(belt)
    lo, hi = np.min(pts, axis=0), np.max(pts, axis=0)
    return {
        "targets": targets, "cameras": cams, "wheel": wheel, "zone_tol_deg": tol,
        "head": {"c": _r(hc), "r": HEAD_R_M, "face_tip": _r(hc + FACE_LEN_M * head.R[:, 0]),
                 "outline": _r(face), "eyes": _r(eyes)},
        "eye_point": _r(eye), "belt": _r(belt),
        "bounds": {"lo": _r(lo), "hi": _r(hi)},
    }


def page(scene_data: dict, lang: str = "zh", sources: tuple[str, str] = ("configs/cabin.yaml",
                                                                        "configs/cameras.yaml")) -> str:
    """The HTML document. `sources`: names of the two config files, printed in the page footer."""
    if lang not in TEXT:
        raise ValueError(f"lang must be one of {sorted(TEXT)}, got {lang!r}")
    t = dict(TEXT[lang])
    t["l_rings"] = t["l_rings"].format(tol=f"{scene_data['zone_tol_deg']:g}")
    t["src"] = t["src"].format(cabin=sources[0], cameras=sources[1])
    blob = json.dumps({"scene": scene_data, "text": t}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    blob = blob.replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8")
    for k in ("lang", "title"):
        html = html.replace("{{" + k + "}}", t[k])
    return html.replace("/*{{DATA}}*/null", blob)


def write(out_dir, cabin: Cabin, cameras: list[Camera], langs=("zh", "en"),
          sources: tuple[str, str] = ("configs/cabin.yaml", "configs/cameras.yaml")) -> list[Path]:
    """Write one page per language into out_dir (FILES names). Returns the paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = scene(cabin, cameras)
    paths = []
    for lg in langs:
        p = out / FILES[lg]
        p.write_text(page(data, lg, sources), encoding="utf-8", newline="\n")
        paths.append(p)
    return paths

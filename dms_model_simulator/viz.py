"""Visualisation: synthetic camera view (what the model 'sees' + outputs) and cabin layout plots.

The composite view (camera view + cabin top/side plots) is rendered fast: the static cabin figure
(targets, cameras, the highlighted camera's field of view, wheel) is drawn once with matplotlib into a cached PIL
image together with a data->pixel transform per subplot; each frame only overlays the head, the face "nose" wedge
and the eye sight line with PIL. The story view (story_frame) is one large top-down picture with plain words, for
people who do not read the per-frame output.
"""
from __future__ import annotations
import functools
import io
import shutil
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from .camera import Camera
from .world import Cabin

# ---- one visual coding for every view (camera view, cabin views, story view) ----------------------------------
# Hue = quantity: eyes (gaze) orange, face direction blue (a pair that stays apart under colour blindness).
# Line style = source: truth solid and filled, model dashed and hollow. Shape = quantity again: the face is a short
# "nose" wedge, the eyes a long sight line with an arrowhead.
EYES_COLOR = (240, 150, 0)          # eye gaze: where the eyes look
FACE_COLOR = (45, 125, 225)         # face direction: where the nose points
GT_BOX_COLOR = (80, 220, 120)       # truth boxes: solid
PRED_BOX_COLOR = (255, 90, 90)      # model boxes: dashed
EVENT_COLOR = (220, 30, 30)
HEAD_FILL = (205, 208, 214)         # head disc in the cabin and story views (neutral: blue is the face)
HEAD_EDGE = (120, 124, 134)
CAM_BG = (28, 28, 34)               # camera-view background
CAM_CASING = (12, 12, 16)           # thin border under model (dashed) strokes in the camera view
PAPER = (255, 255, 255)             # cabin / story background, also the border under model strokes there
INK = (40, 40, 40)
INK_DIM = (115, 115, 120)
HUD_TEXT = (235, 235, 235)
HUD_TRUTH = (200, 255, 200)
HUD_MODEL = (255, 175, 160)
HUD_DIM = (150, 150, 160)
TARGET_COLOR = (165, 165, 165)
# every colour a legend or marker uses: save_gif keeps these exact in every frame
LEGEND_COLORS = (EYES_COLOR, FACE_COLOR, GT_BOX_COLOR, PRED_BOX_COLOR, EVENT_COLOR, HEAD_FILL, HEAD_EDGE, CAM_BG,
                 CAM_CASING, PAPER, INK, INK_DIM, HUD_TEXT, HUD_TRUTH, HUD_MODEL, HUD_DIM, TARGET_COLOR,
                 (0, 0, 0), (220, 220, 220))

FACE_LEN_M, FACE_HALF_W_M, FACE_BASE_M = 0.30, 0.07, 0.06    # cabin / story nose wedge (m)
EYES_LEN_M = 1.0                                               # cabin / story sight line (m)
HEAD_R_M = 0.09                                                # head disc radius (m)
STUB_RATIO = 0.2       # a direction whose 2D projection is shorter than this part of its length: in/out-of-plane mark
TOWARD_CAM_PX = 15     # camera view: an arrow shorter than this (px at a 400 px high image) becomes a "toward" mark

# cabin subplots: (horizontal axis idx, vertical axis idx, xlabel, ylabel, sign h, sign v, xlim, ylim)
_CABIN_VIEWS = (
    (1, 0, "left (+Y)  <-  Y  ->  right (-Y)", "X forward", -1, 1, (-1.3, 1.3), (-0.6, 1.3)),
    (0, 2, "X forward", "Z up", 1, 1, (-0.6, 1.3), (-0.9, 0.7)),
)
_CABIN_FIGSIZE = (13, 6)
_SIDE_LABEL_MAX_Y = 0.3    # side view: targets further than this from the y = 0 plane get no label (they pile up)


@functools.lru_cache(maxsize=64)
def _font(size=14, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for cand in (name, str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name)):
        try:
            return ImageFont.truetype(cand, max(1, int(size)))
        except Exception:
            continue
    return ImageFont.load_default()


def _stroke(height: float) -> int:
    """Stroke width for a picture `height` px high: 3 px at 400 px, never under 2."""
    return max(2, round(height / 133))


def _label_px(height: float) -> float:
    """Cabin-view label size (px) for a picture `height` px high: 11 px at 400 px, between 8 and 15."""
    return float(np.clip(height / 36.0, 8.0, 15.0))


def _out_of_screen(ia: int, ib: int, sa: float, sb: float) -> np.ndarray:
    """Vehicle-frame unit vector pointing out of a subplot towards the viewer (screen x right, y up)."""
    ex, ey = np.zeros(3), np.zeros(3)
    ex[ia], ey[ib] = sa, sb
    return np.cross(ex, ey)


# ---- drawing primitives (PIL, no antialiasing: colours stay exact) -----------------------------------------------
def _dashes(p0, p1, dash: float, gap: float):
    """Segments of a dashed line that starts inked at p0."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    v = p1 - p0
    L = float(np.linalg.norm(v))
    if not np.isfinite(L) or L < 1e-6:
        return []
    u, out, s = v / L, [], 0.0
    while s < L:
        e = min(L, s + dash)
        out.append((tuple(p0 + u * s), tuple(p0 + u * e)))
        s = e + gap
    return out


def _line(d, p0, p1, color, width, dashed=False, casing=None):
    """Solid line, or a dashed one (inked at p0) with an optional `casing` border of the background colour."""
    if not dashed:
        d.line([tuple(map(float, p0)), tuple(map(float, p1))], fill=color, width=int(width))
        return
    segs = _dashes(p0, p1, max(4.0, 3.0 * width), max(3.0, 2.0 * width))
    if casing is not None:
        for a, b in segs:
            d.line([a, b], fill=casing, width=int(width) + 2)
    for a, b in segs:
        d.line([a, b], fill=color, width=int(width))


def _outline(d, pts, color, width, dashed=False, casing=None):
    """Closed outline; dashed edges start inked at their first point (so a wedge's tip is always drawn)."""
    pts = [tuple(map(float, p)) for p in pts]
    if not dashed:
        if casing is not None:
            d.line(pts + [pts[0]], fill=casing, width=int(width) + 2, joint="curve")
        d.line(pts + [pts[0]], fill=color, width=int(width), joint="curve")
        return
    for a, b in zip(pts, pts[1:] + pts[:1]):
        _line(d, a, b, color, width, dashed=True, casing=casing)


def _arrow(d: ImageDraw.ImageDraw, p0, p1, color, width=2, dashed=False, casing=None):
    """Line p0 -> p1 with an arrowhead at p1: truth solid with a filled head, model (dashed=True) dashed with a hollow
    head. `casing`: background colour drawn under the model strokes so they stay visible on a truth arrow."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    v = p1 - p0
    L = float(np.linalg.norm(v))
    if not np.isfinite(L) or L < 1:
        return
    u = v / L
    n = np.array([-u[1], u[0]])
    hl = min(5 + 3 * width, L * 0.5)
    base = p1 - u * hl
    head = [tuple(p1), tuple(base + n * hl * 0.5), tuple(base - n * hl * 0.5)]
    if not dashed:
        d.line([tuple(p0), tuple(base)], fill=color, width=int(width))
        d.polygon(head, fill=color, outline=color)
        return
    _line(d, base, p0, color, width, dashed=True, casing=casing)        # dashes start at the head
    _outline(d, head, color, max(2, int(width) - 1), casing=casing)


def _wedge_pts(base, tip, half_w):
    base, tip = np.asarray(base, float), np.asarray(tip, float)
    v = tip - base
    L = float(np.linalg.norm(v))
    if not np.isfinite(L) or L < 1e-6:
        return None
    n = np.array([-v[1], v[0]]) / L
    return [tuple(tip), tuple(base + n * half_w), tuple(base - n * half_w)]


def _wedge(d, base, tip, half_w, color, width=2, model=False, casing=None):
    """Face "nose" wedge from base to tip: truth filled, model hollow with a dashed outline."""
    pts = _wedge_pts(base, tip, half_w)
    if pts is None:
        return
    if model:
        _outline(d, pts, color, max(2, int(width) - 1), dashed=True, casing=casing)
    else:
        d.polygon(pts, fill=color, outline=color)


def _plane_marker(d, c, r, color, width, toward: bool, model=False, casing=None):
    """A direction along the line of sight: a ring with a dot (towards the viewer) or a cross (away from it).
    Model: dashed ring."""
    x, y = map(float, c)
    box = [x - r, y - r, x + r, y + r]
    w = int(width)
    if model:
        for k in range(8):
            if casing is not None:
                d.arc([box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1], k * 45, k * 45 + 26, fill=casing, width=w + 2)
            d.arc(box, k * 45, k * 45 + 26, fill=color, width=w)
    else:
        d.ellipse(box, outline=color, width=w)
    if toward:
        rd = max(2.0, 0.3 * r)
        if model:
            d.ellipse([x - rd, y - rd, x + rd, y + rd], outline=color, width=max(1, w - 1))
        else:
            d.ellipse([x - rd, y - rd, x + rd, y + rd], fill=color)
    else:
        k = 0.55 * r
        d.line([(x - k, y - k), (x + k, y + k)], fill=color, width=max(2, w - 1))
        d.line([(x - k, y + k), (x + k, y - k)], fill=color, width=max(2, w - 1))


# ---- plain words for directions (vehicle frame: yaw + = left, pitch + = up) --------------------------------------
def _ang(a: float) -> int:
    return int(round(abs(float(a))))


def dir_words(yaw: float, pitch: float, short: bool = False, min_pitch: float = 3.0) -> str:
    """'31° L 10° up' (short) or '31° left, 10° up'; small pitches are left out; '0°' / 'ahead' when straight."""
    parts = []
    if _ang(yaw):
        parts.append(f"{_ang(yaw)}° {('L' if yaw > 0 else 'R') if short else ('left' if yaw > 0 else 'right')}")
    if abs(pitch) >= min_pitch and _ang(pitch):
        parts.append(f"{_ang(pitch)}° {'up' if pitch > 0 else 'down'}")
    if not parts:
        return "0°" if short else "ahead"
    return " ".join(parts) if short else ", ".join(parts)


def _wrap180(a: float) -> float:
    return (float(a) + 180.0) % 360.0 - 180.0


def _rel_words(eyes: dict, face: dict, short: bool = False) -> str:
    """Eyes relative to the face, in words."""
    return dir_words(_wrap180(eyes["yaw_vehicle"] - face["yaw_vehicle"]),
                     eyes["pitch_vehicle"] - face["pitch_vehicle"], short=short)


def _valid_dir(x) -> bool:
    return bool(x) and x.get("valid", True) and x.get("dir_vehicle") is not None


def _finite(*pts) -> bool:
    return all(p is not None and np.all(np.isfinite(np.asarray(p, float))) for p in pts)


def _active_events(tr: dict) -> list[str]:
    """Normalise `truth["events"]` (list of names, or dict name -> bool / {"active": bool}) to active names."""
    ev = tr.get("events")
    if not ev:
        return []
    if isinstance(ev, str):
        return [ev]
    if isinstance(ev, dict):
        out = []
        for k, v in ev.items():
            on = v.get("active", v.get("value", True)) if isinstance(v, dict) else v
            if isinstance(on, (bool, np.bool_)) and on:      # numeric fields (off_road_s, closed_s) are not labels
                out.append(str(k))
        return out
    return [str(e) for e in ev if e]


def _wrap(d: ImageDraw.ImageDraw, segments: list[str], font, max_w: float, sep: str = "  ") -> list[str]:
    """Greedy-pack text segments into lines no wider than max_w (segments are never split)."""
    lines, cur = [], ""
    for seg in segments:
        cand = f"{cur}{sep}{seg}" if cur else seg
        if cur and d.textlength(cand, font=font) > max_w:
            lines.append(cur); cur = seg
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


def _draw_mouth(d, lm, ok, P, openness):
    """Lips as a lens shape between the mouth corners; the opening height follows mouth_openness."""
    if not (ok("mouth_left") and ok("mouth_right") and ok("mouth_center")):
        return
    L, R, C = np.array(P(lm["mouth_left"])), np.array(P(lm["mouth_right"])), np.array(P(lm["mouth_center"]))
    w = float(np.linalg.norm(R - L))
    if w < 1:
        return
    u = (R - L) / w
    n = np.array([-u[1], u[0]])
    if n[1] < 0:
        n = -n                                   # normal points down in the image
    mo = float(np.clip(openness or 0.0, 0.0, 1.0))
    h = w * 0.75 * mo                            # opening height in px
    up_h, lo_h = 0.12 * w + 0.3 * h, 0.12 * w + 0.7 * h
    mid = (L + R) / 2
    shift = (C - mid) - np.dot(C - mid, u) * u   # follow the centre landmark off the corner line
    ts = np.linspace(0, 1, 11)
    sn = np.sin(np.pi * ts)
    upper = [L + (R - L) * t + (shift - n * up_h) * k for t, k in zip(ts, sn)]
    lower = [L + (R - L) * t + (shift + n * lo_h) * k for t, k in zip(ts[::-1], sn[::-1])]
    lip = (210, 120, 135)
    d.polygon([tuple(p) for p in upper + lower], outline=lip, fill=(30, 12, 18) if mo > 0.08 else (120, 70, 80))
    if mo <= 0.08:
        d.line([tuple(L), tuple(mid + shift), tuple(R)], fill=lip, width=2)


def draw_camera_view(record: dict, cam: Camera, cabin: Cabin, scale: float = 0.5, show_pred: bool = True,
                     clean: bool = False) -> Image.Image:
    """Render one frame as the camera would see it: face wireframe, boxes (truth solid green, model dashed red), face
    direction (blue wedge) and eye gaze (orange line), truth solid, model dashed and hollow; HUD in plain words.
    clean=True: the face wireframe only (no landmark dots, boxes, gaze or HUD), as a stand-in camera image for a
    program that draws its own boxes (export frame_yaml --image render)."""
    W, H = int(cam.width * scale), int(cam.height * scale)
    img = Image.new("RGB", (W, H), CAM_BG)
    d = ImageDraw.Draw(img)
    tr, pr = record["truth"], record.get("pred") or {"dropped": True}
    s = scale
    P = lambda uv: (uv[0] * s, uv[1] * s)
    lm = {k: np.array(v, float) for k, v in tr.get("landmarks_px", {}).items()}
    vis = tr.get("landmarks_visible", {})
    ok = lambda k: k in lm and bool(np.all(np.isfinite(lm[k])))
    fs = max(9, round(W / 58))                   # base HUD font size, scales with image width
    f_small, f_hud = _font(max(8, fs - 2)), _font(fs)

    # ---- face wireframe --------------------------------------------------------
    outline = [k for k in cabin.face_outline if ok(k)]
    if len(outline) >= 3:
        d.polygon([P(lm[k]) for k in outline], outline=(150, 150, 160), fill=(60, 55, 60))
    if ok("nose_bridge") and ok("nose_tip"):
        d.line([P(lm["nose_bridge"]), P(lm["nose_tip"])], fill=(170, 170, 180), width=2)
    _draw_mouth(d, lm, ok, P, tr.get("mouth_openness", 0))
    eo = tr.get("eye_openness", {"left": 1.0, "right": 1.0})

    def draw_eyes():
        for eye in ("left_eye", "right_eye"):     # ellipse height follows openness
            o, i_, c = f"{eye}_outer", f"{eye}_inner", f"{eye}_center"
            if not (ok(o) and ok(i_) and ok(c)):
                continue
            ew = np.linalg.norm(lm[o] - lm[i_]) * s
            opn = eo["left" if eye == "left_eye" else "right"]
            h = max(ew * 0.45 * opn, 1.0)
            cx, cy = P(lm[c])
            col = (240, 240, 240) if vis.get(c) else (110, 90, 90)
            d.ellipse([cx - ew / 2, cy - h / 2, cx + ew / 2, cy + h / 2], outline=col, width=2)
            if opn > 0.2:
                d.ellipse([cx - h / 3, cy - h / 3, cx + h / 3, cy + h / 3], fill=col)
    if clean:
        draw_eyes()
        return img

    # ---- face (wedge) and eyes (line): truth solid, model dashed and hollow. The model rays have the truth rays'
    # length (0.15 m) and projection (incl. lens distortion, truth.py), from the projected head centre. The face
    # wedges go under the eyes, landmarks, boxes and labels (a filled wedge would hide the near eye at a head
    # turn); the eye lines go on top of everything.
    w = _stroke(H)
    dropped = bool(pr.get("dropped"))
    pg = None if dropped else pr.get("gaze")
    pf = None if dropped else pr.get("face_dir")
    g = tr["gaze"]
    fg = tr.get("face_dir")
    items = []                                   # (quantity, model, p0, p1, toward the camera)
    if fg and _finite(fg.get("origin_px"), fg.get("tip_px")):
        items.append(("face", False, P(fg["origin_px"]), P(fg["tip_px"]), _toward_camera(fg)))
    if show_pred and _valid_dir(pf):
        seg = _project_ray(cam, tr["head_center_vehicle"], pf["dir_vehicle"])
        if seg:
            items.append(("face", True, P(seg[0]), P(seg[1]), _toward_camera(pf)))
    if _finite(g.get("origin_px"), g.get("tip_px")):
        items.append(("eyes", False, P(g["origin_px"]), P(g["tip_px"]), _toward_camera(g)))
    if show_pred and _valid_dir(pg):
        seg = _project_ray(cam, tr["head_center_vehicle"], pg["dir_vehicle"])
        if seg:
            items.append(("eyes", True, P(seg[0]), P(seg[1]), _toward_camera(pg)))
    _draw_camera_dirs(d, items, w, H, only="face")

    draw_eyes()
    for k, uv in lm.items():
        if ok(k):
            x, y = P(uv)
            col = (200, 230, 200) if vis.get(k) else (200, 70, 70)
            d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=col)

    # ---- boxes: truth solid, model dashed ------------------------------------------
    for det in tr.get("detections", []):
        x1, y1, x2, y2 = [v * s for v in det["bbox"]]
        d.rectangle([x1, y1, x2, y2], outline=GT_BOX_COLOR, width=2)
        d.text((x1 + 2, y1 - fs - 1), str(det.get("class", det.get("label", "?"))), fill=GT_BOX_COLOR, font=f_small)
    if show_pred and not dropped:
        for det in pr.get("detections", []):
            x1, y1, x2, y2 = [v * s for v in det["bbox"]]
            _outline(d, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], PRED_BOX_COLOR, 2, dashed=True)
            conf = det.get("conf", det.get("score", 0.0))
            d.text((x1 + 2, y2 + 1), f"{det.get('class', det.get('label', '?'))} {conf:.2f}", fill=PRED_BOX_COLOR, font=f_small)
    _draw_camera_dirs(d, items, w, H, only="eyes")

    # ---- HUD text (wrapped to the image width): plain words in the vehicle frame first ------------------------
    hp_gt = tr["head_pose_camera"]
    truth = ["truth: " + (f"face {dir_words(fg['yaw_vehicle'], fg['pitch_vehicle'], short=True)}" if fg else "face none"),
             f"eyes {dir_words(g['yaw_vehicle'], g['pitch_vehicle'], short=True)}"]
    if fg:
        truth.append(f"eyes−face {_rel_words(g, fg, short=True)}")
    state = [f"roll (cam) {hp_gt['roll']:+.0f}°", f"eye opening {eo['left']:.2f}/{eo['right']:.2f}",
             f"mouth {float(tr.get('mouth_openness', 0) or 0):.2f}", f"hand={tr.get('hand')}", f"phone={tr.get('phone')}"]
    if dropped:
        model = ["model: frame dropped" + (" (latency fill)" if pr.get("latency_fill") else "")]
    else:
        model = ["model: " + (f"face {dir_words(pf['yaw_vehicle'], pf['pitch_vehicle'], short=True)} "
                              f"(err {_err(pf)})" if _valid_dir(pf) else "face none")]
        if _valid_dir(pg):
            model += [f"eyes {dir_words(pg['yaw_vehicle'], pg['pitch_vehicle'], short=True)} (err {_err(pg)})",
                      f"eyes conf {pg.get('conf', pg.get('score', 0)):.2f}"]
        else:
            model.append(f"eyes invalid ({pg.get('reason') if pg else 'none'})")
        hp = pr.get("head_pose_camera")
        if hp:
            model.append(f"roll (cam) {hp['roll']:+.0f}°")
    camf = ["camera frame:"]
    if fg:
        camf.append(f"face yaw/pitch {fg['yaw_camera']:+.0f}/{fg['pitch_camera']:+.0f}")
    camf.append(f"eyes {g['yaw_camera']:+.0f}/{g['pitch_camera']:+.0f}")
    if g.get("gaze_to_camera_deg") is not None:
        camf.append(f"eyes→camera {g['gaze_to_camera_deg']:.0f}°")
    if _valid_dir(pf) and pf.get("yaw_camera") is not None:
        camf.append(f"model face {pf['yaw_camera']:+.0f}/{pf['pitch_camera']:+.0f}")
    if _valid_dir(pg) and pg.get("yaw_camera") is not None:
        camf.append(f"model eyes {pg['yaw_camera']:+.0f}/{pg['pitch_camera']:+.0f}")
    hud = [(HUD_TEXT, [cam.name, f"{cam.width}x{cam.height} hfov {cam.hfov:.0f}", f"t={record['t']:.2f}s"], "  ", f_hud),
           (HUD_TRUTH, truth, " | ", f_hud), (HUD_TRUTH, state, "  ", f_small), (HUD_MODEL, model, " | ", f_hud),
           (HUD_DIM, camf, "  ", f_small)]
    lines = [(col, ln, fnt) for col, segs, sep, fnt in hud for ln in _wrap(d, segs, fnt, W - 12, sep)]
    lh_of = lambda fnt: fnt.size + 3
    top_h = 4 + sum(lh_of(fnt) for _c, _l, fnt in lines) + 2

    f_big = _font(fs + 3, bold=True)
    leg_h = fs + 4
    leg_rows = camera_legend_rows(d, W, fs)
    bot_h = (fs + 7) + leg_h * len(leg_rows) + 8      # target/zone row + legend rows
    band = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bd = ImageDraw.Draw(band)
    bd.rectangle([0, 0, W, top_h], fill=(0, 0, 0, 165))
    bd.rectangle([0, H - bot_h, W, H], fill=(0, 0, 0, 165))
    img = Image.alpha_composite(img.convert("RGBA"), band).convert("RGB")
    d = ImageDraw.Draw(img)
    y = 4
    for col, ln, fnt in lines:
        d.text((6, y), ln, fill=col, font=fnt)
        y += lh_of(fnt)
    # which way round the picture is
    d.text((6, H - bot_h - f_small.size - 5), "camera view (mirror image: driver's left is on the right)",
           fill=HUD_DIM, font=f_small)

    # scripted target and zone, large
    y_big, x = H - bot_h + 3, 6
    pz = pg.get("zone") if (pg and pg.get("valid")) else None
    items = [(f"target: {g.get('target')}", (255, 255, 255)), (f"zone: {g.get('zone')}", HUD_TRUTH)]
    if pz is not None:
        items.append((f"model: {pz}", HUD_TRUTH if pz == g.get("zone") else PRED_BOX_COLOR))
    for t, col in items:
        d.text((x, y_big), t, fill=col, font=f_big)
        x += d.textlength(t, font=f_big) + fs

    # legend: each swatch drawn in the style it stands for
    sw = _legend_swatch_w(fs)
    lw = max(2, w - 1)
    for i, row in enumerate(leg_rows):
        y_leg = H - leg_h * (len(leg_rows) - i) - 4
        yc = y_leg + leg_h // 2
        for x, (kind, model, label) in row:
            _legend_swatch(d, kind, model, x, yc, sw, fs, lw, CAM_CASING)
            d.text((x + sw + 4, y_leg + 1), label, fill=(220, 220, 220), font=f_small)

    # event tags (top-right, under the HUD band)
    tags = [(e.upper(), EVENT_COLOR) for e in _active_events(tr)]
    if dropped:
        tags.append(("DROPPED", INK_DIM))
    f_tag = _font(fs + 2, bold=True)
    xr, yt = W - 6, top_h + 6
    for text, col in tags:
        tw = d.textlength(text, font=f_tag)
        if xr - tw - 10 < 6 and xr < W - 6:     # wrap to the next row
            xr, yt = W - 6, yt + fs + 12
        d.rectangle([xr - tw - 10, yt, xr, yt + fs + 8], fill=col, outline=(255, 255, 255))
        d.text((xr - tw - 5, yt + 3), text, fill=(255, 255, 255), font=f_tag)
        xr -= tw + 16
    return img


def _err(x: dict) -> str:
    e = x.get("angular_error_deg")
    return "?" if e is None or not np.isfinite(e) else f"{e:.0f}°"


def _toward_camera(x: dict) -> bool:
    """Does the direction point towards the camera (camera frame: the camera looks along +z)?"""
    dc = x.get("dir_camera")
    return True if dc is None else float(dc[2]) <= 0


def _project_ray(cam: Camera, origin_vehicle, dir_vehicle, length: float = 0.15):
    o = np.array(origin_vehicle, float)
    uv0, _ = cam.project(o)
    uv1, _ = cam.project(o + length * np.array(dir_vehicle, float))
    return (uv0, uv1) if _finite(uv0, uv1) else None


def _draw_camera_dirs(d, items, w: int, H: int, only: str | None = None):
    """Face wedges and eye lines (only: draw just "face" or "eyes"; draw_camera_view puts the wedges under the
    features and the eye lines on top). A ray that projects shorter than TOWARD_CAM_PX (scaled to the image height)
    points along the line of sight: a ring with a dot (towards the camera) or a cross (away), larger for the face
    than for the eyes. Ring or ray is decided by the truth direction of that quantity and used for the model too,
    so a small model error never swaps the symbol."""
    min_px = TOWARD_CAM_PX * H / 400.0
    r_face = max(7, round(H / 30))
    r_eyes = max(4, round(r_face * 0.5))
    length = lambda p0, p1: float(np.linalg.norm(np.subtract(p1, p0)))
    stub = {q: length(p0, p1) < min_px for q, model, p0, p1, _t in items if not model}
    for q, model, p0, p1, toward in items:
        if only is not None and q != only:
            continue
        col = FACE_COLOR if q == "face" else EYES_COLOR
        L = length(p0, p1)
        if stub.get(q, L < min_px):
            _plane_marker(d, p0, r_face if q == "face" else r_eyes, col, w if q == "face" else max(2, w - 1),
                          toward, model=model, casing=CAM_CASING)
        elif q == "face":
            _wedge(d, p0, p1, max(1.5 * w, min(0.15 * L, 3.5 * w)), col, w, model=model, casing=CAM_CASING)
        else:
            _arrow(d, p0, p1, col, w, dashed=model, casing=CAM_CASING)


_CAMERA_LEGEND = (("box", False, "truth box"), ("box", True, "model box"), ("eyes", False, "eyes: truth"),
                  ("eyes", True, "eyes: model"), ("face", False, "face: truth"), ("face", True, "face: model"))


def _legend_swatch_w(fs: int) -> int:
    return max(16, round(2.2 * fs))


def camera_legend_rows(d, W: int, fs: int):
    """Camera-view legend entries packed into rows no wider than the image: [[(x, (kind, model, label)), ...]]."""
    f_small = _font(max(8, fs - 2))
    sw = _legend_swatch_w(fs)
    rows, x = [[]], 6
    for item in _CAMERA_LEGEND:
        wd = sw + 4 + d.textlength(item[2], font=f_small)
        if rows[-1] and x + wd > W - 6:
            rows.append([]); x = 6
        rows[-1].append((x, item))
        x += wd + fs
    return rows


def _legend_swatch(d, kind, model, x, yc, sw, fs, lw, casing):
    """One legend swatch in the style it stands for (box, eye line, face wedge; model dashed / hollow)."""
    if kind == "box":
        pts = [(x, yc - fs * 0.4), (x + sw, yc - fs * 0.4), (x + sw, yc + fs * 0.4), (x, yc + fs * 0.4)]
        if model:
            _outline(d, pts, PRED_BOX_COLOR, 2, dashed=True)
        else:
            d.rectangle([pts[0], pts[2]], outline=GT_BOX_COLOR, width=2)
    elif kind == "eyes":
        _arrow(d, (x, yc), (x + sw, yc), EYES_COLOR, lw, dashed=model, casing=casing)
    elif kind == "face":
        _wedge(d, (x, yc), (x + sw, yc), max(3, fs * 0.42), FACE_COLOR, lw, model=model, casing=casing)
    elif kind == "plane":
        r = max(4, fs * 0.42)
        _plane_marker(d, (x + r, yc), r, INK, 2, True)
        _plane_marker(d, (x + sw - r, yc), r, INK, 2, False)


# ---- cabin plots ---------------------------------------------------------------
def _hex(c) -> str:
    return "#%02x%02x%02x" % tuple(c)


def _cabin_static_fig(cabin: Cabin, cameras: list[Camera], highlight: str | None, dpi: float | None = None,
                      targets=None):
    """Figure with everything that does not change per frame. Returns (fig, axes). Only the highlighted camera
    gets its field of view drawn. `targets`: names to label (the rest are small unlabelled marks); None = all.
    Rasterised (dpi given), the label size follows the picture height (_label_px)."""
    fig, axes = plt.subplots(1, 2, figsize=_CABIN_FIGSIZE, dpi=dpi)
    lab_fs = 8.0 if dpi is None else _label_px(dpi * _CABIN_FIGSIZE[1]) * 72.0 / dpi
    wheel = [oc["polyline"] for oc in cabin.occluder_geometry("wheel", cabin.head_state(0, 0, 0)) if oc["type"] == "torus"]
    used = None if targets is None else set(targets)
    for k, (ax, (ia, ib, la, lb, sa, sb, xl, yl)) in enumerate(zip(axes, _CABIN_VIEWS)):
        for name, p in cabin.targets.items():
            if np.linalg.norm(p) > 3:
                continue
            if used is not None and name not in used:
                ax.plot(sa * p[ia], sb * p[ib], "s", color="#ccc", ms=4)
                continue
            ax.plot(sa * p[ia], sb * p[ib], "s", color=_hex(TARGET_COLOR), ms=7)
            if k == 1 and abs(p[1]) > _SIDE_LABEL_MAX_Y:      # side view: far off the y = 0 plane, labels pile up
                continue
            ax.annotate(name, (sa * p[ia], sb * p[ib]), fontsize=lab_fs, color="#444", xytext=(4, 4),
                        textcoords="offset points")
        for cam in cameras:
            c = cam.position
            hi = cam.name == highlight
            col = "#e33" if hi else "#aaa"
            ax.plot(sa * c[ia], sb * c[ib], "^", color=col, ms=9 if hi else 6)
            if hi or highlight is None:
                ax.annotate(cam.name, (sa * c[ia], sb * c[ib]), fontsize=lab_fs, color="#c22" if hi else "#999",
                            xytext=(4, -10), textcoords="offset points")
            if hi:
                axis = cam.optical_axis
                ang = np.degrees(np.arctan2(sb * axis[ib], sa * axis[ia]))
                fov = cam.hfov if k == 0 else cam.vfov
                ax.add_patch(Wedge((sa * c[ia], sb * c[ib]), 0.6, ang - fov / 2, ang + fov / 2, color=col, alpha=0.18))
        for pl in wheel:
            ax.plot(sa * pl[:, ia], sb * pl[:, ib], color="#888", lw=2)
        ax.set_xlabel(la, fontsize=lab_fs + 1); ax.set_ylabel(lb, fontsize=lab_fs + 1)
        ax.tick_params(labelsize=lab_fs)
        ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlim(*xl); ax.set_ylim(*yl)
    axes[0].set_title("top-down (origin = driver eye point)", fontsize=lab_fs * 1.25)
    axes[1].set_title("side view (seen from the driver's right)", fontsize=lab_fs * 1.25)
    fig.tight_layout()
    return fig, axes


def _record_dirs(record: dict, show_model: bool = True) -> list[tuple[str, bool, np.ndarray]]:
    """(quantity, model, unit dir_vehicle) in drawing order: face wedges first, eye lines on top."""
    tr = record["truth"]
    pr = record.get("pred") or {}
    pr = {} if pr.get("dropped") or not show_model else pr
    out = []
    for q, key in (("face", "face_dir"), ("eyes", "gaze")):
        for model, src in ((False, tr.get(key)), (True, pr.get(key))):
            if _valid_dir(src):
                v = np.array(src["dir_vehicle"], float)
                n = float(np.linalg.norm(v))
                if np.isfinite(n) and n > 0:
                    out.append((q, model, v / n))
    return out


def _draw_head_dirs(d, to_px, hc, dirs, px_per_m: float, w: int, casing, out_vec, head: bool = True):
    """Head disc, face "nose" wedges (FACE_LEN_M) and eye sight lines (EYES_LEN_M) in one 2D view. A direction that
    projects shorter than STUB_RATIO of its length becomes a ring with a dot (out of the picture, towards the
    viewer) or a cross (into it); the face ring is larger than the eyes ring. Ring or ray is decided by the truth
    direction of that quantity and used for the model too, so a small model error never swaps the symbol."""
    hx, hy = to_px(hc)
    r = HEAD_R_M * px_per_m
    if head:
        d.ellipse([hx - r, hy - r, hx + r, hy + r], fill=HEAD_FILL, outline=HEAD_EDGE, width=max(1, w // 2))
    ratio = lambda dv: float(np.linalg.norm(np.subtract(to_px(hc + dv), (hx, hy)))) / px_per_m
    stub = {q: ratio(dv) < STUB_RATIO for q, model, dv in dirs if not model}
    for q, model, dv in dirs:
        col = FACE_COLOR if q == "face" else EYES_COLOR
        if stub.get(q, ratio(dv) < STUB_RATIO):
            toward = float(np.dot(dv, out_vec)) > 0
            _plane_marker(d, (hx, hy), r * (0.8 if q == "face" else 0.42), col, w if q == "face" else max(2, w - 1),
                          toward, model=model, casing=casing)
        elif q == "face":
            _wedge(d, to_px(hc + FACE_BASE_M * dv), to_px(hc + FACE_LEN_M * dv), FACE_HALF_W_M * px_per_m, col, w,
                   model=model, casing=casing)
        else:
            _arrow(d, (hx, hy), to_px(hc + EYES_LEN_M * dv), col, w, dashed=model, casing=casing)


def _mpl_head_dirs(ax, hc, dirs, ia, ib, sa, sb):
    """The same coding as _draw_head_dirs, with matplotlib artists (plot_cabin)."""
    from matplotlib.patches import Circle, Polygon
    P = lambda p: np.array([sa * p[ia], sb * p[ib]])
    out_vec = _out_of_screen(ia, ib, sa, sb)
    h = P(hc)
    ax.add_patch(Circle(h, HEAD_R_M, fc=_hex(HEAD_FILL), ec=_hex(HEAD_EDGE), zorder=3))
    stub = {q: np.linalg.norm(P(hc + dv) - h) < STUB_RATIO for q, model, dv in dirs if not model}
    for q, model, dv in dirs:
        col = _hex(FACE_COLOR if q == "face" else EYES_COLOR)
        ls = (0, (4, 3)) if model else "-"
        v2 = P(hc + dv) - h
        if stub.get(q, np.linalg.norm(v2) < STUB_RATIO):
            toward = float(np.dot(dv, out_vec)) > 0
            ax.plot(*h, marker="o", ms=16 if q == "face" else 9, mfc="none", mec=col, ls="none", zorder=5)
            ax.plot(*h, marker="." if toward else "x", ms=6, color=col, ls="none", zorder=5)
        elif np.linalg.norm(v2) < 1e-9:              # model straight along the view, truth not: nothing to draw
            continue
        elif q == "face":
            u = v2 / np.linalg.norm(v2)
            n = np.array([-u[1], u[0]]) * FACE_HALF_W_M
            base, tip = P(hc + FACE_BASE_M * dv), P(hc + FACE_LEN_M * dv)
            ax.add_patch(Polygon([tip, base + n, base - n], closed=True, fc="none" if model else col, ec=col,
                                 lw=1.5, ls=ls, zorder=4))
        else:
            ax.annotate("", xy=tuple(P(hc + EYES_LEN_M * dv)), xytext=tuple(h),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5 if not model else 2, ls=ls), zorder=4)


def plot_cabin(cabin: Cabin, cameras: list[Camera], record: dict | None = None, highlight: str | None = None):
    """Top-down (left) and side (right) views of targets, cameras (the highlighted one with its field of view), head,
    face "nose" wedge and eye sight line (matplotlib). Truth solid, model dashed."""
    fig, axes = _cabin_static_fig(cabin, cameras, highlight)
    for ax, (ia, ib, _la, _lb, sa, sb, _xl, _yl) in zip(axes, _CABIN_VIEWS):
        if record:
            _mpl_head_dirs(ax, np.array(record["truth"]["head_center_vehicle"], float), _record_dirs(record),
                           ia, ib, sa, sb)
        else:
            from matplotlib.patches import Circle
            hc = cabin.nominal_head_center()
            ax.add_patch(Circle((sa * hc[ia], sb * hc[ib]), HEAD_R_M, fc=_hex(HEAD_FILL), ec=_hex(HEAD_EDGE)))
    if record:
        from matplotlib.lines import Line2D
        e, f = _hex(EYES_COLOR), _hex(FACE_COLOR)
        axes[0].legend([Line2D([], [], color=e, lw=2.5), Line2D([], [], color=e, lw=2, ls=(0, (4, 3))),
                        Line2D([], [], marker=">", color=f, lw=0, ms=10),
                        Line2D([], [], marker=">", mfc="none", color=f, lw=0, ms=10)],
                       ["eyes: truth", "eyes: model", "face (nose): truth", "face (nose): model"],
                       loc="upper left", fontsize=8)
    return fig


class CabinBackground:
    """Static cabin figure rasterised once, plus a data->pixel affine transform for each subplot."""

    def __init__(self, cabin: Cabin, cameras: list[Camera], highlight: str | None, height: int, targets=None):
        dpi = height / _CABIN_FIGSIZE[1]
        fig, axes = _cabin_static_fig(cabin, cameras, highlight, dpi=dpi, targets=targets)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        self.image = Image.fromarray(buf[..., :3].copy())
        fh = buf.shape[0]
        self.px_per_pt = dpi / 72.0
        self.axes = []
        for ax, (ia, ib, _la, _lb, sa, sb, _xl, _yl) in zip(axes, _CABIN_VIEWS):
            t = ax.transData                                     # linear axes: affine per coordinate
            p0, p1 = t.transform((0.0, 0.0)), t.transform((1.0, 1.0))
            bb = ax.bbox.extents                                 # x0, y0, x1, y1 in display (y up)
            box = (int(np.floor(bb[0])), int(np.floor(fh - bb[3])), int(np.ceil(bb[2])), int(np.ceil(fh - bb[1])))
            self.axes.append({"ia": ia, "ib": ib, "sa": sa, "sb": sb, "box": box,
                              "ox": p0[0], "gx": p1[0] - p0[0], "oy": fh - p0[1], "gy": -(p1[1] - p0[1]),
                              "out": _out_of_screen(ia, ib, sa, sb)})
        plt.close(fig)
        if self.image.height != height:                         # guard against canvas rounding
            k = height / self.image.height
            self.image = self.image.resize((round(self.image.width * k), height), Image.BILINEAR)
            self.px_per_pt *= k
            for a in self.axes:
                a["box"] = tuple(int(round(v * k)) for v in a["box"])
                for key in ("ox", "gx", "oy", "gy"):
                    a[key] *= k
        self._legend(self.image, _stroke(height))

    def to_px(self, a: dict, p_vehicle) -> tuple[float, float]:
        """Vehicle-frame point -> pixel in the cabin image for subplot `a`."""
        p = np.asarray(p_vehicle, float)
        return (a["ox"] + a["gx"] * a["sa"] * p[a["ia"]], a["oy"] + a["gy"] * a["sb"] * p[a["ib"]])

    def render(self, record: dict | None) -> Image.Image:
        out = self.image.copy()
        if not record:
            return out
        hc = np.array(record["truth"]["head_center_vehicle"], float)
        dirs = _record_dirs(record)
        w = _stroke(out.height)
        for a in self.axes:
            x0, y0, x1, y1 = a["box"]
            sub = out.crop((x0, y0, x1, y1))                    # draw clipped to the subplot
            loc = lambda p, a=a: tuple(np.subtract(self.to_px(a, p), (x0, y0)))
            _draw_head_dirs(ImageDraw.Draw(sub), loc, hc, dirs, abs(a["gx"]), w, PAPER, a["out"])
            out.paste(sub, (x0, y0))
        return out

    def _legend(self, out: Image.Image, w: int):
        """Legend, bottom-left of the top-down view, each swatch in the style it stands for."""
        d = ImageDraw.Draw(out)
        entries = (("eyes", False, "eyes: truth"), ("eyes", True, "eyes: model"), ("face", False, "face: truth"),
                   ("face", True, "face: model"), ("plane", False, "toward / away"))
        # left-rear corner of the top-down view (behind the driver, to the left: X < 0, Y > 0.5 m), where no sight
        # line of a forward, mirror or passenger glance passes; part of the static background, so the per-frame
        # overlays draw over it rather than under it. The font shrinks until the box ends left of Y = 0.5 m.
        a = self.axes[0]
        x0, _y0, _x1, y1 = a["box"]
        x_max = self.to_px(a, [0.0, 0.5, 0.0])[0] - 2
        fs = round(_label_px(out.height))
        while True:
            f = _font(fs)
            sw = _legend_swatch_w(fs)
            cw = sw + 10 + max(d.textlength(t, font=f) for *_k, t in entries) + 6
            if x0 + 2 + cw <= x_max or fs <= 7:
                break
            fs -= 1
        lh = fs + (6 if out.height >= 300 else 3)          # tighter rows on small pictures
        rows = len(entries)
        top = y1 - 2 - (lh * rows + 6)
        d.rectangle([x0 + 2, top, x0 + 2 + cw, y1 - 2], fill=PAPER, outline=(200, 200, 200))
        for n, (kind, model, label) in enumerate(entries):
            x, y = x0 + 7, top + 3 + n * lh
            _legend_swatch(d, kind, model, x, y + lh // 2, sw, fs, max(2, w - 1), PAPER)
            d.text((x + sw + 6, y + 1), label, fill=INK, font=f)


_BG_CACHE: dict = {}


def cabin_background(cabin: Cabin, cameras: list[Camera], highlight: str | None, height: int,
                     targets=None) -> CabinBackground:
    """Cached CabinBackground for this (cabin, cameras, highlight, height, targets)."""
    tkey = None if targets is None else tuple(targets)
    key = (id(cabin), tuple(id(c) for c in cameras), highlight, int(height), tkey)
    hit = _BG_CACHE.get(key)
    if hit is not None and hit[0] is cabin and all(a is b for a, b in zip(hit[1], cameras)):
        return hit[2]
    if len(_BG_CACHE) > 16:
        _BG_CACHE.clear()
    bg = CabinBackground(cabin, cameras, highlight, int(height), targets=tkey)
    _BG_CACHE[key] = (cabin, list(cameras), bg)
    return bg


def fig_to_image(fig) -> Image.Image:
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=80); plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGB")


def composite(record: dict, cam: Camera, cabin: Cabin, cameras: list[Camera], scale=0.5, targets=None) -> Image.Image:
    """Camera view beside the cabin top-down and side views. `targets`: names to label in the cabin views."""
    view = draw_camera_view(record, cam, cabin, scale)
    cab = cabin_background(cabin, cameras, cam.name, view.height, targets).render(record)
    out = Image.new("RGB", (view.width + cab.width, view.height), (0, 0, 0))
    out.paste(view, (0, 0)); out.paste(cab, (view.width, 0))
    return out


def view_indices(n: int, sim_fps: float, view_fps: float) -> list[int]:
    """Record indices to show when n records simulated at sim_fps are played back in real time at view_fps:
    frame k of the video shows the latest record at or before k / view_fps (records repeat when view_fps is
    higher than sim_fps)."""
    if not view_fps > 0 or not sim_fps > 0:
        raise ValueError(f"view_fps and sim_fps must be > 0, got {view_fps!r} and {sim_fps!r}")
    count = max(1, int(np.floor(n / sim_fps * view_fps + 1e-9)))
    return [min(n - 1, int(np.floor(k / view_fps * sim_fps + 1e-9))) for k in range(count)]


def gif_palette(frames: list[Image.Image], n_sample: int = 8) -> Image.Image:
    """One palette for a whole GIF: LEGEND_COLORS exactly, plus colours adapted to a sample of the frames."""
    fixed = list(dict.fromkeys(tuple(c) for c in LEGEND_COLORS))
    step = max(1, len(frames) // n_sample)
    tiles = []
    for fr in frames[::step][:n_sample]:
        im = fr.convert("RGB")
        k = min(1.0, 320 / max(1, im.width))
        tiles.append(im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.NEAREST))
    mosaic = Image.new("RGB", (max(t.width for t in tiles), sum(t.height for t in tiles)))
    y = 0
    for t in tiles:
        mosaic.paste(t, (0, y)); y += t.height
    q = mosaic.quantize(256 - len(fixed), method=Image.Quantize.MEDIANCUT)
    adaptive = np.array(q.getpalette()[:3 * (256 - len(fixed))], dtype=np.uint8).reshape(-1, 3)
    # Pillow maps colours to a palette through a coarse lookup cache: an adaptive colour close to a fixed one could
    # take its pixels, so adaptive colours keep a distance from every fixed colour (and fixed ones from each other)
    fx = np.array(fixed, float)
    far = np.linalg.norm(adaptive[:, None, :].astype(float) - fx[None], axis=-1).min(axis=1) > _PALETTE_GAP
    colors = fixed + [tuple(int(v) for v in c) for c in adaptive[far]]
    colors = (colors + [(0, 0, 0)] * 256)[:256]
    pal = Image.new("P", (1, 1))
    pal.putpalette([v for c in colors for v in c])
    return pal


_PALETTE_GAP = 14.0


def save_gif(frames: list[Image.Image], path: str | Path, fps: float):
    """Write an animated GIF. Every frame is mapped (no dithering) onto one palette that holds every legend and
    marker colour exactly (gif_palette), so a frame cannot recolour the legend. Takes RGB or palette frames."""
    pal = gif_palette(frames)
    q = [(fr if fr.mode == "RGB" else fr.convert("RGB")).quantize(palette=pal, dither=Image.Dither.NONE)
         for fr in frames]
    q[0].save(path, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def save_mp4(frames: list[Image.Image], path: str | Path, fps: float) -> bool:
    """Encode frames to MP4 by piping PNGs into ffmpeg. Returns False if ffmpeg is missing or fails."""
    exe = shutil.which("ffmpeg")
    if not exe or not frames:
        return False
    for codec in (["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20"], ["-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p"]):
        cmd = [exe, "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", f"{fps:.4f}", "-c:v", "png", "-i", "-",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", *codec, str(path)]
        proc = None
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for fr in frames:
                buf = io.BytesIO(); fr.save(buf, format="PNG", compress_level=1)
                proc.stdin.write(buf.getvalue())
            proc.stdin.close()
            if proc.wait() == 0:
                return True
        except OSError:
            if proc is not None:
                proc.kill(); proc.wait()
    return False


# ---- camera grids ----------------------------------------------------------------------------


def _bar_height(width: int) -> int:
    return max(16, round(width / 58) + 12)


def grid_label_height(tile_width: int) -> int:
    """Height of the camera-name bar make_grid puts above each tile."""
    return _bar_height(tile_width)


def make_grid(images: list[Image.Image], labels: list[str], cols: int | None = None) -> Image.Image:
    """Tile images (one per camera) into a grid, each with a name bar on top. Tiles are padded to the
    largest image; cols defaults to ceil(sqrt(n)). Size: (cols * tile_w, rows * (tile_h + label_h))."""
    n = len(images)
    if n == 0:
        raise ValueError("make_grid needs at least one image")
    cols = max(1, min(int(cols or np.ceil(np.sqrt(n))), n))
    rows = int(np.ceil(n / cols))
    tw, th = max(im.width for im in images), max(im.height for im in images)
    lh = grid_label_height(tw)
    out = Image.new("RGB", (cols * tw, rows * (th + lh)), (12, 12, 16))
    d = ImageDraw.Draw(out)
    f = _font(lh - 10, bold=True)
    for i, (im, lab) in enumerate(zip(images, labels)):
        x0, y0 = (i % cols) * tw, (i // cols) * (th + lh)
        d.rectangle([x0, y0, x0 + tw - 1, y0 + lh - 1], fill=(44, 44, 52))
        d.text((x0 + 6, y0 + 4), str(lab), fill=(240, 240, 240), font=f)
        out.paste(im, (x0, y0 + lh))
        if i % cols:                                  # thin separator between columns
            d.line([x0, y0, x0, y0 + lh + th], fill=(12, 12, 16), width=2)
    return out


# ---- timeline plot ------------------------------------------------------------------------------
# reference categorical palette (fixed order) + neutral inks on a light surface
_TL_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
_TL_SURFACE, _TL_INK, _TL_INK2, _TL_MUTED, _TL_GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#a3a29c", "#e4e3df"


def _tl_frame(r: dict) -> dict:
    """Per-frame fields from a full simulator record or a truth-JSONL record."""
    if isinstance(r.get("truth"), dict):
        tr, pr = r["truth"], r.get("pred") or {}
        pg = None if pr.get("dropped") else (pr.get("gaze") or {})
        return {"frame_id": r["frame"], "t": float(r["t"]), "zone": tr["gaze"]["zone"], "eo": tr.get("eye_openness"),
                "pred_eo": None if pr.get("dropped") else pr.get("eye_openness"),
                "pred_zone": pg.get("zone") if pg and pg.get("valid") else None, "events": tr.get("events") or {}}
    g = (r.get("gaze") or {}).get("eye") or {}
    return {"frame_id": r["frame_id"], "t": r["timestamp_ms"] / 1000.0, "zone": g.get("zone"), "eo": r.get("eye_openness"),
            "pred_eo": None, "pred_zone": None, "events": r.get("events") or {}}


def _mean_eo(eo):
    if not eo:
        return np.nan
    v = [x for x in (eo.get("left"), eo.get("right")) if x is not None]
    return float(np.mean(v)) if v else np.nan


def _error_rows(records: list[dict]) -> dict[str, list[bool]]:
    """Injected model errors per frame (full simulator records only): dropped frame, missed face box, eye / mouth
    open-closed class flip, any false-positive box, any box jump, a head pose / face direction / eye gaze outlier.
    Rows that never occur are left out."""
    if not records or not isinstance(records[0].get("pred"), dict):
        return {}
    rows = {
        "dropped": [bool(r["pred"].get("dropped")) for r in records],
        "face missed": [not r["pred"].get("dropped") and any(d["class"] == "face" for d in r["truth"]["detections"])
                        and not any(d["class"] == "face" for d in r["pred"]["detections"]) for r in records],
        "eye class flip": ["eye" in (r["pred"].get("class_flip") or []) for r in records],
        "mouth class flip": ["mouth" in (r["pred"].get("class_flip") or []) for r in records],
        "false positive": [any(d.get("false_positive") for d in r["pred"]["detections"]) for r in records],
        "box jump": [any(d.get("jump") for d in r["pred"]["detections"]) for r in records],
        "pose/gaze outlier": [any((r["pred"].get(k) or {}).get("outlier")
                                  for k in ("head_pose_camera", "gaze", "face_dir")) for r in records],
    }
    return {k: v for k, v in rows.items() if any(v)}


def plot_timeline(records: list[dict], path: str | Path | None = None, pipeline: list[dict] | None = None,
                  title: str | None = None):
    """Timeline figure: truth gaze zone, eye openness (truth + model), truth behaviour labels and the injected
    model errors over time. `records`: full simulator records or truth-JSONL records (no error rows then);
    `pipeline`: optional pipeline records for the model openness / zone when `records` are truth-only.
    Saves to `path` when given; returns the figure."""
    from .metrics import segments
    fr = [_tl_frame(r) for r in records]
    if pipeline is not None:
        pm = {p["frame_id"]: p for p in pipeline}
        for f in fr:
            p = pm.get(f["frame_id"])
            if p is not None:
                f["pred_eo"] = p.get("eye_openness")
                g = (p.get("gaze") or {}).get("eye") or {}
                f["pred_zone"] = g.get("zone") if g.get("valid") else None
    t = np.array([f["t"] for f in fr])
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0 / 30
    err = _error_rows(records)
    err_names = list(err)
    ev0 = fr[0]["events"] if fr else {}
    ev_names = [k for k, v in ev0.items() if isinstance(v, (bool, np.bool_))]
    ev_color = {n: _TL_SERIES[i % len(_TL_SERIES)] for i, n in enumerate(ev_names)}
    err_color = {n: _TL_SERIES[(i + 1) % len(_TL_SERIES)] for i, n in enumerate(err_names)}

    zones = []
    for f in fr:
        for z in (f["zone"], f["pred_zone"]):
            if z is not None and z not in zones:
                zones.append(z)
    zi = {z: i for i, z in enumerate(zones)}

    hr = [max(1.6, 0.32 * len(zones)), 1.5, max(0.6, 0.3 * len(ev_names)), max(0.6, 0.3 * len(err_names))]
    with plt.rc_context({"font.size": 9}):
        fig, axes = plt.subplots(4, 1, figsize=(11, 1.0 + 1.1 * sum(hr)), sharex=True,
                                 gridspec_kw={"height_ratios": hr, "hspace": 0.18})
        fig.patch.set_facecolor(_TL_SURFACE)
        for ax in axes:
            ax.set_facecolor(_TL_SURFACE)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(_TL_MUTED)
            ax.tick_params(colors=_TL_INK2, length=3)
            ax.grid(axis="x", color=_TL_GRID, lw=0.8)
            ax.set_axisbelow(True)

        ax = axes[0]
        pz = [(f["t"], zi[f["pred_zone"]]) for f in fr if f["pred_zone"] is not None]
        if pz:
            ax.plot(*zip(*pz), "o", ms=2.5, color=_TL_MUTED, alpha=0.7, label="model zone (valid gaze)", lw=0)
        ax.step(t, [zi.get(f["zone"], np.nan) for f in fr], where="post", color=_TL_SERIES[0], lw=2, label="truth zone")
        ax.set_yticks(range(len(zones))); ax.set_yticklabels(zones)
        ax.set_ylim(-0.6, len(zones) - 0.4); ax.invert_yaxis()
        ax.set_ylabel("gaze zone", color=_TL_INK2)
        ax.grid(axis="y", color=_TL_GRID, lw=0.8)
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.18), frameon=False, fontsize=8, ncol=2, labelcolor=_TL_INK2)

        ax = axes[1]
        ax.plot(t, [_mean_eo(f["eo"]) for f in fr], color=_TL_INK2, lw=2, label="truth")
        if any(f["pred_eo"] for f in fr):
            ax.plot(t, [_mean_eo(f["pred_eo"]) for f in fr], color=_TL_SERIES[0], lw=1.5, alpha=0.9, label="model")
        ax.set_ylim(-0.05, 1.1); ax.set_ylabel("eye openness", color=_TL_INK2)
        ax.legend(loc="lower right", frameon=False, fontsize=8, ncol=2, labelcolor=_TL_INK2)

        def bars(ax, names, flags_of, colors, ylabel):
            for i, n in enumerate(names):
                spans = [(t[a], (t[b - 1] - t[a]) + dt) for a, b in segments(flags_of(n))]
                c = colors.get(n, _TL_MUTED)            # edge in the bar's colour: a blink in a 5-minute row is
                ax.broken_barh(spans, (i - 0.32, 0.64), facecolors=c, edgecolor=c, lw=0.8)   # narrower than a pixel
            ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
            ax.set_ylim(-0.6, max(len(names), 1) - 0.4); ax.invert_yaxis()
            ax.set_ylabel(ylabel, color=_TL_INK2)
            ax.tick_params(axis="y", length=0)

        bars(axes[2], ev_names, lambda n: [bool(f["events"].get(n)) for f in fr], ev_color, "truth\nbehaviour")
        bars(axes[3], err_names, lambda n: err[n], err_color, "injected\nmodel errors")
        axes[3].set_xlabel("time (s)", color=_TL_INK2)
        if len(t):
            axes[3].set_xlim(t[0], t[-1] + dt)
        for ax in axes:
            ax.yaxis.set_label_coords(-0.115, 0.5)
        if title:
            fig.suptitle(title, x=0.01, ha="left", color=_TL_INK, fontsize=11)
        fig.subplots_adjust(left=0.135, right=0.985, top=0.92 if title else 0.96, bottom=0.07)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=110, facecolor=_TL_SURFACE)
    return fig


# ---- story view: one large top-down picture with plain words, for people who do not read JSONL ---------------
STORY_SIZE = (960, 720)
_STORY_Y = (1.15, -1.15)        # vehicle Y shown from the left edge (+ = driver's left) to the right edge (m)
_STORY_X = (-0.3, 1.15)         # vehicle X shown from the bottom (behind the head) to the top (forward) (m)


def _pretty(name) -> str:
    return str(name).replace("_", " ")


class StoryBackground:
    """Everything in a story frame that does not change: targets, steering wheel, the key under the picture."""

    def __init__(self, cabin: Cabin, targets, size=STORY_SIZE):
        W, H = map(int, size)
        self.size = (W, H)
        self.title_h = round(H * 0.09)
        self.text_h = round(H * 0.19)
        pic_h = H - self.title_h - self.text_h
        span_y, span_x = _STORY_Y[0] - _STORY_Y[1], _STORY_X[1] - _STORY_X[0]
        self.ppm = min(W * 0.96 / span_y, pic_h * 0.94 / span_x)
        self.cx = W / 2 + (_STORY_Y[0] + _STORY_Y[1]) / 2 * self.ppm
        self.oy = self.title_h + (pic_h - span_x * self.ppm) / 2 + _STORY_X[1] * self.ppm
        self.w = max(3, round(H / 150))
        self.f_title = _font(max(14, round(H / 24)), bold=True)
        self.f_text = _font(max(12, round(H / 32)))
        self.f_label = _font(max(11, round(H / 42)))
        self.f_key = _font(max(10, round(H / 46)))
        img = Image.new("RGB", (W, H), PAPER)
        d = ImageDraw.Draw(img)
        d.line([(0, self.title_h), (W, self.title_h)], fill=(225, 225, 228), width=1)
        d.line([(0, H - self.text_h), (W, H - self.text_h)], fill=(225, 225, 228), width=1)
        for oc in cabin.occluder_geometry("wheel", cabin.head_state(0, 0, 0)):
            if oc["type"] == "torus":
                d.line([self.to_px(p) for p in oc["polyline"]], fill=(215, 215, 218), width=max(2, self.w - 1))
        d.text((10, self.title_h + 6), "seen from above · front of the car ↑", fill=INK_DIM, font=self.f_key)
        self.target_px = {}
        self.labels = []                             # (xy, text): drawn per frame on top of the arrows, with a halo
        boxes = []
        sq = max(5, round(H / 110))
        # the picture in metres (vehicle X up, Y left), a margin inside its edges
        m = 0.05
        x_hi, x_lo = (self.oy - self.title_h) / self.ppm - m, (self.oy - (H - self.text_h)) / self.ppm + m
        y_hi, y_lo = self.cx / self.ppm - m, (self.cx - W) / self.ppm + m
        for name in targets or []:
            p = np.asarray(cabin.targets.get(name, [np.nan] * 3), float)
            if not np.all(np.isfinite(p)):
                continue
            far = np.linalg.norm(p) > 3
            inside = x_lo <= p[0] <= x_hi and y_lo <= p[1] <= y_hi
            if not inside:                           # off the picture: a mark at its edge, in the target's direction
                u = p[:2] / max(np.linalg.norm(p[:2]), 1e-9)
                ts = [b / c for b, c in ((x_hi, u[0]), (x_lo, u[0]), (y_hi, u[1]), (y_lo, u[1])) if c and b / c > 0]
                t = min(ts) if ts else 0.0
                p = np.array([u[0] * t, u[1] * t, 0.0])
            x, y = self.to_px(p)
            if not inside:
                ux, uy = np.subtract(self.to_px(p * 1.01 + [0, 0, 0]), (x, y))
                n = np.hypot(ux, uy) or 1.0
                ux, uy = ux / n, uy / n
                d.polygon([(x + ux * sq * 1.5, y + uy * sq * 1.5), (x - uy * sq - ux * sq * 0.5, y + ux * sq - uy * sq * 0.5),
                           (x + uy * sq - ux * sq * 0.5, y - ux * sq - uy * sq * 0.5)], fill=TARGET_COLOR)
            else:
                d.rectangle([x - sq, y - sq, x + sq, y + sq], fill=TARGET_COLOR)
            self.target_px[name] = (x, y, sq)
            label = _pretty(name) + (" (far)" if far else "")
            tw = d.textlength(label, font=self.f_label)
            lx = min(max(4, x - tw / 2), W - tw - 4)
            ly = min(y + sq + 3, H - self.text_h - self.f_label.size - 4)
            if ly < y + sq:                          # no room under the mark (bottom / side edge): beside it
                lx, ly = (x - sq - 4 - tw, y - self.f_label.size / 2) if x > W / 2 else (x + sq + 4, y - self.f_label.size / 2)
            for _ in range(4):                       # step down past labels already placed
                bb = (lx, ly, lx + tw, ly + self.f_label.size + 2)
                if not any(bb[0] < b[2] and b[0] < bb[2] and bb[1] < b[3] and b[1] < bb[3] for b in boxes):
                    break
                ly += self.f_label.size + 3
            boxes.append((lx, ly, lx + tw, ly + self.f_label.size + 2))
            self.labels.append(((lx, ly), label))
        self._key(d)
        self.image = img

    def draw_labels(self, d):
        for xy, text in self.labels:
            d.text(xy, text, fill=INK_DIM, font=self.f_label, stroke_width=2, stroke_fill=PAPER)

    def to_px(self, p) -> tuple[float, float]:
        p = np.asarray(p, float)
        return (self.cx - p[1] * self.ppm, self.oy - p[0] * self.ppm)

    def _key(self, d):
        """One line at the bottom: what the shapes and line styles mean."""
        W, H = self.size
        f = self.f_key
        fs = f.size
        sw = _legend_swatch_w(fs)
        y = H - fs - 10
        yc = y + fs // 2 + 1
        entries = [("face", False, "face (nose direction)"), ("eyes", False, "eyes (where they look)"),
                   ("style", False, "solid = truth"), ("style", True, "dashed / hollow = model")]
        widths = [sw + 6 + d.textlength(t, font=f) for *_k, t in entries]
        x = max(8, (W - sum(widths) - 2 * fs * (len(entries) - 1)) / 2)
        for (kind, model, label), wd in zip(entries, widths):
            if kind == "style":
                _line(d, (x, yc), (x + sw, yc), INK_DIM, 2, dashed=model, casing=PAPER)
            else:
                _legend_swatch(d, kind, model, x, yc, sw, fs, max(2, self.w - 1), PAPER)
            d.text((x + sw + 6, y), label, fill=INK, font=f)
            x += wd + 2 * fs


_STORY_CACHE: dict = {}


def story_background(cabin: Cabin, targets=None, size=STORY_SIZE) -> StoryBackground:
    """Cached StoryBackground. targets: names to show (default: every target near the driver)."""
    if targets is None:
        targets = [n for n, p in cabin.targets.items() if np.linalg.norm(p) <= 3]
    key = (id(cabin), tuple(targets), tuple(map(int, size)))
    hit = _STORY_CACHE.get(key)
    if hit is not None and hit[0] is cabin:
        return hit[1]
    if len(_STORY_CACHE) > 8:
        _STORY_CACHE.clear()
    bg = StoryBackground(cabin, targets, size)
    _STORY_CACHE[key] = (cabin, bg)
    return bg


def auto_caption(record: dict) -> str:
    """A caption from the truth directions, e.g. 'face 30° left, eyes ahead'."""
    tr = record["truth"]
    g, f = tr["gaze"], tr.get("face_dir")
    eyes = f"eyes {dir_words(g['yaw_vehicle'], g['pitch_vehicle'])}"
    return f"face {dir_words(f['yaw_vehicle'], f['pitch_vehicle'])}, {eyes}" if f else eyes


def story_lines(record: dict, show_model: bool = True) -> tuple[str, str | None]:
    """The two lines under a story picture: the truth directions in words, and the model error."""
    tr = record["truth"]
    g, f = tr["gaze"], tr.get("face_dir")
    eyes = f"eyes {dir_words(g['yaw_vehicle'], g['pitch_vehicle'])}"
    if f:
        line1 = (f"face {dir_words(f['yaw_vehicle'], f['pitch_vehicle'])} · {eyes} · "
                 f"eyes relative to face: {_rel_words(g, f)}")
    else:
        line1 = eyes
    if not show_model:
        return line1, None
    pr = record.get("pred") or {}
    if pr.get("dropped"):
        return line1, "model: no output on this frame"
    parts = []
    pf, pg = pr.get("face_dir"), pr.get("gaze")
    if f:
        parts.append(f"face {_err(pf)}" if _valid_dir(pf) else "face: no output")
    parts.append(f"eyes {_err(pg)}" if _valid_dir(pg) else
                 f"eyes: no output ({_pretty(pg.get('reason'))})" if pg and pg.get("reason") else "eyes: no output")
    return line1, "model error: " + ", ".join(parts)


def _fit_font(d, text: str, font, max_w: float):
    """The font, or a smaller one, so the text fits max_w."""
    size = font.size
    while size > 9 and d.textlength(text, font=font) > max_w:
        size -= 1
        font = _font(size, bold="Bold" in str(getattr(font, "path", "")))
    return font


def story_frame(record: dict, cabin: Cabin, caption: str | None = None, targets=None, show_model: bool = True,
                size=STORY_SIZE, background: StoryBackground | None = None) -> Image.Image:
    """Presentation frame: a large top-down view of the head with the face "nose" wedge and the eye sight line (truth
    solid, model dashed and hollow on top), the targets near the action (`targets`, default: all near ones, the one
    the eyes are on ringed), a one-line `caption` (default auto_caption) and two lines of plain words underneath."""
    bg = background or story_background(cabin, targets, size)
    W, H = bg.size
    img = bg.image.copy()
    d = ImageDraw.Draw(img)
    tr = record["truth"]
    zone = tr["gaze"].get("zone")
    if zone in bg.target_px:                         # ring the target the eyes are on
        x, y, sq = bg.target_px[zone]
        r = sq * 2.2
        d.ellipse([x - r, y - r, x + r, y + r], outline=EYES_COLOR, width=max(2, bg.w - 1))
    hc = np.array(tr["head_center_vehicle"], float)
    _draw_head_dirs(d, bg.to_px, hc, _record_dirs(record, show_model), bg.ppm, bg.w, PAPER, np.array([0.0, 0.0, 1.0]))
    bg.draw_labels(d)
    # title, time
    t_txt = f"t = {float(record.get('t', 0.0)):.1f} s"
    tw = d.textlength(t_txt, font=bg.f_key)
    title = caption or auto_caption(record)
    f_title = _fit_font(d, title, bg.f_title, W - tw - 40)
    d.text((14, (bg.title_h - f_title.size) / 2 - 2), title, fill=INK, font=f_title)
    d.text((W - tw - 12, (bg.title_h - bg.f_key.size) / 2), t_txt, fill=INK_DIM, font=bg.f_key)
    # two lines of plain words
    line1, line2 = story_lines(record, show_model)
    y = H - bg.text_h + max(6, round(bg.text_h * 0.08))
    for text, col in ((line1, INK), (line2, INK_DIM)):
        if not text:
            continue
        f = _fit_font(d, text, bg.f_text, W - 28)
        d.text(((W - d.textlength(text, font=f)) / 2, y), text, fill=col, font=f)
        y += f.size + 8
    return img


def story_targets(cabin: Cabin, segments, records=None, min_dwell_s: float = 0.3) -> list[str]:
    """Targets near the action, for the whole run (labels do not flicker): the scripted targets and the zones the
    truth gaze stays on for at least min_dwell_s (a gaze sweeping past a zone does not count), in order of use."""
    names = [s.get("target") for s in segments]
    if records:
        run_zone, run_t0 = None, 0.0
        for r in list(records) + [None]:
            z = None if r is None else r["truth"]["gaze"].get("zone")
            t = float(records[-1]["t"]) if r is None else float(r["t"])
            if z != run_zone:
                if run_zone is not None and t - run_t0 >= min_dwell_s:
                    names.append(run_zone)
                run_zone, run_t0 = z, t
    return [n for n in dict.fromkeys(names) if n in cabin.targets]


def segment_captions(segments, records) -> list[str]:
    """Per segment: its `caption:` key, else auto_caption of its middle frame."""
    out = []
    for s in segments:
        if s.get("caption"):
            out.append(str(s["caption"]))
            continue
        f0, f1 = int(s["start_frame"]), min(int(s["end_frame"]), len(records)) - 1
        out.append(auto_caption(records[max(0, (f0 + f1) // 2)]) if f1 >= f0 else "")
    return out


def segment_of_frames(segments, n: int) -> list[int]:
    """Segment index of each record 0..n-1."""
    out, k = [], 0
    for i in range(n):
        while k < len(segments) - 1 and i >= int(segments[k]["end_frame"]):
            k += 1
        out.append(k)
    return out


def story_moments(records: list[dict], segments, k: int = 4) -> list[tuple[int, str]]:
    """Up to k key moments, in time order: (record index, caption). One candidate per segment, its middle frame,
    ranked by the angle between face and eyes, plus 15° when the eyes are on something other than the road."""
    from . import geometry as G
    caps = segment_captions(segments, records)
    cands = []
    for j, s in enumerate(segments):
        f0, f1 = int(s["start_frame"]), min(int(s["end_frame"]), len(records)) - 1
        if f1 < f0:
            continue
        i = (f0 + f1) // 2
        tr = records[i]["truth"]
        f = tr.get("face_dir")
        ang = float(G.angle_between(f["dir_vehicle"], tr["gaze"]["dir_vehicle"])) if f else 0.0
        zone = tr["gaze"].get("zone")
        cands.append((ang + (15.0 if zone not in (None, "road_ahead") else 0.0), i, caps[j]))
    top = sorted(cands, key=lambda c: -c[0])[:k]
    return [(i, c) for _s, i, c in sorted(top, key=lambda c: c[1])]


def story_sheet(images: list[Image.Image], cols: int = 2, title: str | None = None) -> Image.Image:
    """Key-moment frames side by side (cols per row), with an optional title band on top."""
    if not images:
        raise ValueError("story_sheet needs at least one image")
    cols = max(1, min(cols, len(images)))
    rows = int(np.ceil(len(images) / cols))
    tw, th = max(i.width for i in images), max(i.height for i in images)
    gap = max(6, tw // 120)
    f = _font(max(14, th // 22), bold=True)
    top = (f.size + 2 * gap) if title else 0
    out = Image.new("RGB", (cols * tw + (cols + 1) * gap, top + rows * th + (rows + 1) * gap), (225, 225, 228))
    d = ImageDraw.Draw(out)
    if title:
        d.text((gap * 2, gap), title, fill=INK, font=f)
    for n, im in enumerate(images):
        out.paste(im, (gap + (n % cols) * (tw + gap), top + gap + (n // cols) * (th + gap)))
    return out

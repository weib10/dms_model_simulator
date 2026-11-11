"""Visualisation: synthetic camera view (what the model 'sees' + outputs) and cabin layout plots.

The composite view (camera view + cabin top/side plots) is rendered fast: the static cabin figure
(targets, cameras, FOV wedges, wheel) is drawn once with matplotlib into a cached PIL image together
with a data->pixel transform per subplot; each frame only overlays head position, gaze and face-direction arrows
with PIL.
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

GT_BOX_COLOR = (80, 220, 120)
PRED_COLOR = (255, 80, 80)
GAZE_COLOR = (255, 255, 0)
PRED_GAZE_COLOR = (255, 140, 0)
FACE_COLOR = (0, 200, 255)          # face direction (where the head points)
PRED_FACE_COLOR = (150, 235, 255)
EVENT_COLOR = (220, 30, 30)
HEAD_COLOR = (51, 102, 204)
CABIN_GT_GAZE = (221, 204, 0)       # matplotlib "#dc0" (readable on white)
CABIN_PRED_GAZE = (255, 136, 0)     # matplotlib "#f80"
CABIN_GT_FACE = (0, 153, 153)       # matplotlib "#099"
CABIN_PRED_FACE = (102, 204, 204)   # matplotlib "#6cc"

# cabin subplots: (horizontal axis idx, vertical axis idx, xlabel, ylabel, sign h, sign v, xlim, ylim)
_CABIN_VIEWS = (
    (1, 0, "left (+Y)  <-  Y  ->  right (-Y)", "X forward", -1, 1, (-1.3, 1.3), (-0.6, 1.3)),
    (0, 2, "X forward", "Z up", 1, 1, (-0.6, 1.3), (-0.9, 0.7)),
)
_CABIN_FIGSIZE = (13, 6)
_HEAD_MS = 14          # head marker size in points (matplotlib "o" ms)


@functools.lru_cache(maxsize=64)
def _font(size=14, bold=False):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", int(size))
    except Exception:
        return ImageFont.load_default()


def _face_width(gaze_width: float) -> int:
    """Line width of a face-direction arrow: about half the gaze arrow's, at least 1 px (drawn on top of it)."""
    return max(1, int(gaze_width) // 2)


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


def _wrap(d: ImageDraw.ImageDraw, segments: list[str], font, max_w: float) -> list[str]:
    """Greedy-pack text segments into lines no wider than max_w (segments are never split)."""
    lines, cur = [], ""
    for seg in segments:
        cand = f"{cur}  {seg}" if cur else seg
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
    """Render one frame as the camera would see it: face wireframe, GT boxes (green), predictions (red).
    clean=True: the face wireframe only (no landmark dots, boxes, gaze or HUD), as a stand-in camera image for a
    program that draws its own boxes (export frame_yaml --image render)."""
    W, H = int(cam.width * scale), int(cam.height * scale)
    img = Image.new("RGB", (W, H), (28, 28, 34))
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
    for eye in ("left_eye", "right_eye"):     # ellipse height follows openness
        o, i_, c = f"{eye}_outer", f"{eye}_inner", f"{eye}_center"
        if not (ok(o) and ok(i_) and ok(c)):
            continue
        w = np.linalg.norm(lm[o] - lm[i_]) * s
        opn = eo["left" if eye == "left_eye" else "right"]
        h = max(w * 0.45 * opn, 1.0)
        cx, cy = P(lm[c])
        col = (240, 240, 240) if vis.get(c) else (110, 90, 90)
        d.ellipse([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=col, width=2)
        if opn > 0.2:
            d.ellipse([cx - h / 3, cy - h / 3, cx + h / 3, cy + h / 3], fill=col)
    if clean:
        return img
    for k, uv in lm.items():
        if ok(k):
            x, y = P(uv)
            col = (200, 230, 200) if vis.get(k) else (200, 70, 70)
            d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=col)

    # ---- GT boxes + gaze -------------------------------------------------------
    for det in tr.get("detections", []):
        x1, y1, x2, y2 = [v * s for v in det["bbox"]]
        d.rectangle([x1, y1, x2, y2], outline=GT_BOX_COLOR, width=2)
        d.text((x1 + 2, y1 - fs - 1), str(det.get("class", det.get("label", "?"))), fill=GT_BOX_COLOR, font=f_small)
    g = tr["gaze"]
    fg = tr.get("face_dir")
    if _finite(g.get("origin_px"), g.get("tip_px")):
        d.line([P(g["origin_px"]), P(g["tip_px"])], fill=GAZE_COLOR, width=3)
        x, y = P(g["tip_px"]); d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=GAZE_COLOR)

    # ---- predictions -----------------------------------------------------------
    dropped = bool(pr.get("dropped"))
    pg = None if dropped else pr.get("gaze")
    pf = None if dropped else pr.get("face_dir")
    if show_pred and not dropped:
        for det in pr.get("detections", []):
            x1, y1, x2, y2 = [v * s for v in det["bbox"]]
            d.rectangle([x1, y1, x2, y2], outline=PRED_COLOR, width=1)
            conf = det.get("conf", det.get("score", 0.0))
            d.text((x1 + 2, y2 + 1), f"{det.get('class', det.get('label', '?'))} {conf:.2f}", fill=PRED_COLOR, font=f_small)
        # same ray length and projection (incl. lens distortion) as the GT gaze in truth.py
        origin_v = np.array(tr["head_center_vehicle"], float)
        if pg and pg.get("valid") and pg.get("dir_vehicle") is not None:
            uv0, _ = cam.project(origin_v)
            uv1, _ = cam.project(origin_v + 0.15 * np.array(pg["dir_vehicle"], float))
            if _finite(uv0, uv1):
                d.line([P(uv0), P(uv1)], fill=PRED_GAZE_COLOR, width=2)
                x, y = P(uv1); d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=PRED_GAZE_COLOR)

    # ---- face directions: on top of both gaze arrows and half their width, so they show when head and eyes
    # point the same way (GT from truth.py's projection, PRED with the same ray length and projection)
    if fg and _finite(fg.get("origin_px"), fg.get("tip_px")):
        d.line([P(fg["origin_px"]), P(fg["tip_px"])], fill=FACE_COLOR, width=_face_width(3))
        x, y = P(fg["tip_px"]); d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=FACE_COLOR)
    if show_pred and pf and pf.get("valid") and pf.get("dir_vehicle") is not None:
        origin_v = np.array(tr["head_center_vehicle"], float)
        uv0, _ = cam.project(origin_v)
        uv1, _ = cam.project(origin_v + 0.15 * np.array(pf["dir_vehicle"], float))
        if _finite(uv0, uv1):
            d.line([P(uv0), P(uv1)], fill=PRED_FACE_COLOR, width=_face_width(2))
            x, y = P(uv1); d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=PRED_FACE_COLOR)

    # ---- HUD text (wrapped to the image width) ---------------------------------
    hp_gt = tr["head_pose_camera"]
    hud = [((235, 235, 235), [cam.name, f"{cam.width}x{cam.height} hfov {cam.hfov:.0f}", f"t={record['t']:.2f}s"]),
           ((200, 255, 200), ["GT", f"head {hp_gt['yaw']:+.0f}/{hp_gt['pitch']:+.0f}/{hp_gt['roll']:+.0f}",
                              f"gaze cam {g['yaw_camera']:+.0f}/{g['pitch_camera']:+.0f}",
                              f"veh {g['yaw_vehicle']:+.0f}/{g['pitch_vehicle']:+.0f}",
                              f"->cam {g['gaze_to_camera_deg']:.0f}deg",
                              *([f"face cam {fg['yaw_camera']:+.0f}/{fg['pitch_camera']:+.0f}"] if fg else []),
                              f"eyes {eo['left']:.2f}/{eo['right']:.2f}",
                              f"mouth {float(tr.get('mouth_openness', 0) or 0):.2f}",
                              f"hand={tr.get('hand')}", f"phone={tr.get('phone')}"])]
    if dropped:
        hud.append((PRED_COLOR, ["PRED", "frame dropped" + (" (latency fill)" if pr.get("latency_fill") else "")]))
    else:
        hp = pr.get("head_pose_camera")
        segs = ["PRED", f"head {hp['yaw']:+.0f}/{hp['pitch']:+.0f}/{hp['roll']:+.0f}" if hp else "head none"]
        if pf and pf.get("valid"):
            segs.append(f"face {pf['yaw_camera']:+.0f}/{pf['pitch_camera']:+.0f}")
        if pg and pg.get("valid"):
            segs += [f"gaze {pg.get('yaw_camera', 0):+.0f}/{pg.get('pitch_camera', 0):+.0f}",
                     f"err {pg.get('angular_error_deg', float('nan')):.1f}deg",
                     f"conf {pg.get('conf', pg.get('score', 0)):.2f}"]
        else:
            segs.append(f"gaze invalid ({pg.get('reason') if pg else 'none'})")
        hud.append((PRED_COLOR, segs))
    lines = [(col, ln) for col, segs in hud for ln in _wrap(d, segs, f_hud, W - 12)]
    lh = fs + 3
    top_h = 4 + lh * len(lines) + 2

    f_big = _font(fs + 3, bold=True)
    leg_h = fs + 2
    # legend entries packed into rows no wider than the image (a row wraps like the event tags)
    legend = (("box", GT_BOX_COLOR, "GT box"), ("box", PRED_COLOR, "PRED box"),
              ("line", GAZE_COLOR, "GT gaze"), ("line", PRED_GAZE_COLOR, "PRED gaze"),
              ("line", FACE_COLOR, "GT face"), ("line", PRED_FACE_COLOR, "PRED face"))
    leg_rows, x = [[]], 6
    for item in legend:
        w = fs + 3 + d.textlength(item[2], font=f_small)
        if leg_rows[-1] and x + w > W - 6:
            leg_rows.append([]); x = 6
        leg_rows[-1].append((x, item))
        x += w + fs
    bot_h = (fs + 7) + leg_h * len(leg_rows) + 8      # target/zone row + legend rows
    band = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bd = ImageDraw.Draw(band)
    bd.rectangle([0, 0, W, top_h], fill=(0, 0, 0, 165))
    bd.rectangle([0, H - bot_h, W, H], fill=(0, 0, 0, 165))
    img = Image.alpha_composite(img.convert("RGBA"), band).convert("RGB")
    d = ImageDraw.Draw(img)
    for i, (col, ln) in enumerate(lines):
        d.text((6, 4 + lh * i), ln, fill=col, font=f_hud)

    # scripted target and zone, large
    y_big, x = H - bot_h + 3, 6
    pz = pg.get("zone") if (pg and pg.get("valid")) else None
    items = [(f"target: {g.get('target')}", (255, 255, 255)), (f"zone: {g.get('zone')}", (200, 255, 200))]
    if pz is not None:
        items.append((f"pred: {pz}", (200, 255, 200) if pz == g.get("zone") else PRED_COLOR))
    for t, col in items:
        d.text((x, y_big), t, fill=col, font=f_big)
        x += d.textlength(t, font=f_big) + fs

    # legend (rows laid out above)
    sw = fs
    for i, row in enumerate(leg_rows):
        y_leg = H - leg_h * (len(leg_rows) - i) - 4
        for x, (kind, col, label) in row:
            if kind == "box":
                d.rectangle([x, y_leg + 2, x + sw, y_leg + sw - 1], outline=col, width=2)
            else:
                d.line([x, y_leg + sw // 2 + 1, x + sw, y_leg + sw // 2 + 1], fill=col, width=3)
            d.text((x + sw + 3, y_leg), label, fill=(220, 220, 220), font=f_small)

    # event tags (top-right, under the HUD band)
    tags = [(e.upper(), EVENT_COLOR) for e in _active_events(tr)]
    if dropped:
        tags.append(("DROPPED", (110, 110, 120)))
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


# ---- cabin plots ---------------------------------------------------------------
def _cabin_static_fig(cabin: Cabin, cameras: list[Camera], highlight: str | None, dpi: float | None = None):
    """Figure with everything that does not change per frame. Returns (fig, axes)."""
    fig, axes = plt.subplots(1, 2, figsize=_CABIN_FIGSIZE, dpi=dpi)
    wheel = [oc["polyline"] for oc in cabin.occluder_geometry("wheel", cabin.head_state(0, 0, 0)) if oc["type"] == "torus"]
    for k, (ax, (ia, ib, la, lb, sa, sb, xl, yl)) in enumerate(zip(axes, _CABIN_VIEWS)):
        for name, p in cabin.targets.items():
            if np.linalg.norm(p) > 3:
                continue
            ax.plot(sa * p[ia], sb * p[ib], "s", color="#4c8", ms=7)
            ax.annotate(name, (sa * p[ia], sb * p[ib]), fontsize=8, xytext=(4, 4), textcoords="offset points")
        for cam in cameras:
            c = cam.position
            col = "#e33" if cam.name == highlight else "#888"
            ax.plot(sa * c[ia], sb * c[ib], "^", color=col, ms=9)
            ax.annotate(cam.name, (sa * c[ia], sb * c[ib]), fontsize=8, color=col, xytext=(4, -10), textcoords="offset points")
            axis = cam.optical_axis
            ang = np.degrees(np.arctan2(sb * axis[ib], sa * axis[ia]))
            fov = cam.hfov if k == 0 else cam.vfov
            ax.add_patch(Wedge((sa * c[ia], sb * c[ib]), 0.6, ang - fov / 2, ang + fov / 2, color=col,
                               alpha=0.12 if cam.name != highlight else 0.3))
        for pl in wheel:
            ax.plot(sa * pl[:, ia], sb * pl[:, ib], color="#555", lw=2)
        ax.set_xlabel(la); ax.set_ylabel(lb); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_xlim(*xl); ax.set_ylim(*yl)
    axes[0].set_title("top-down (vehicle frame, origin = driver eye point)")
    axes[1].set_title("side view")
    fig.tight_layout()
    return fig, axes


def plot_cabin(cabin: Cabin, cameras: list[Camera], record: dict | None = None, highlight: str | None = None):
    """Top-down (left) and side (right) views of targets, cameras with FOV, head, gaze and face-direction rays
    (matplotlib)."""
    fig, axes = _cabin_static_fig(cabin, cameras, highlight)
    for ax, (ia, ib, _la, _lb, sa, sb, _xl, _yl) in zip(axes, _CABIN_VIEWS):
        if record:
            tr = record["truth"]
            hc = np.array(tr["head_center_vehicle"])
            ax.plot(sa * hc[ia], sb * hc[ib], "o", color="#36c", ms=_HEAD_MS, alpha=0.7)
            pr = record.get("pred") or {}
            tip = hc + 1.0 * np.array(tr["gaze"]["dir_vehicle"])
            ax.annotate("", xy=(sa * tip[ia], sb * tip[ib]), xytext=(sa * hc[ia], sb * hc[ib]),
                        arrowprops=dict(arrowstyle="->", color="#dc0", lw=2))
            pg = (record.get("pred") or {}).get("gaze")
            if pg and pg.get("valid"):
                tip = hc + 1.0 * np.array(pg["dir_vehicle"])
                ax.annotate("", xy=(sa * tip[ia], sb * tip[ib]), xytext=(sa * hc[ia], sb * hc[ib]),
                            arrowprops=dict(arrowstyle="->", color="#f80", lw=1.5))
            for fd, col, lw in ((tr.get("face_dir"), "#099", 1.0), (pr.get("face_dir"), "#6cc", 0.75)):
                if fd and fd.get("valid", True) and fd.get("dir_vehicle") is not None:     # on top, half width
                    tip = hc + 1.0 * np.array(fd["dir_vehicle"])
                    ax.annotate("", xy=(sa * tip[ia], sb * tip[ib]), xytext=(sa * hc[ia], sb * hc[ib]),
                                arrowprops=dict(arrowstyle="->", color=col, lw=lw))
        else:
            hc = cabin.nominal_head_center()
            ax.plot(sa * hc[ia], sb * hc[ib], "o", color="#36c", ms=_HEAD_MS, alpha=0.7)
    if record:
        from matplotlib.lines import Line2D
        axes[0].legend([Line2D([], [], color=c, lw=2) for c in ("#dc0", "#f80", "#099", "#6cc")],
                       ["GT gaze", "PRED gaze", "GT face", "PRED face"], loc="upper left", fontsize=8)
    return fig


def _arrow(d: ImageDraw.ImageDraw, p0, p1, color, width=2):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    v = p1 - p0
    L = float(np.linalg.norm(v))
    if not np.isfinite(L) or L < 1:
        return
    u = v / L
    n = np.array([-u[1], u[0]])
    hl = min(4 + 3 * width, L * 0.5)
    base = p1 - u * hl
    d.line([tuple(p0), tuple(base)], fill=color, width=width)
    d.polygon([tuple(p1), tuple(base + n * hl * 0.45), tuple(base - n * hl * 0.45)], fill=color)


class CabinBackground:
    """Static cabin figure rasterised once, plus a data->pixel affine transform for each subplot."""

    def __init__(self, cabin: Cabin, cameras: list[Camera], highlight: str | None, height: int):
        dpi = height / _CABIN_FIGSIZE[1]
        fig, axes = _cabin_static_fig(cabin, cameras, highlight, dpi=dpi)
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
                              "ox": p0[0], "gx": p1[0] - p0[0], "oy": fh - p0[1], "gy": -(p1[1] - p0[1])})
        plt.close(fig)
        if self.image.height != height:                         # guard against canvas rounding
            k = height / self.image.height
            self.image = self.image.resize((round(self.image.width * k), height), Image.BILINEAR)
            self.px_per_pt *= k
            for a in self.axes:
                a["box"] = tuple(int(round(v * k)) for v in a["box"])
                for key in ("ox", "gx", "oy", "gy"):
                    a[key] *= k

    def to_px(self, a: dict, p_vehicle) -> tuple[float, float]:
        """Vehicle-frame point -> pixel in the cabin image for subplot `a`."""
        p = np.asarray(p_vehicle, float)
        return (a["ox"] + a["gx"] * a["sa"] * p[a["ia"]], a["oy"] + a["gy"] * a["sb"] * p[a["ib"]])

    def render(self, record: dict | None) -> Image.Image:
        out = self.image.copy()
        if not record:
            return out
        tr = record["truth"]
        hc = np.array(tr["head_center_vehicle"], float)
        gd = np.array(tr["gaze"]["dir_vehicle"], float)
        pg = (record.get("pred") or {}).get("gaze")
        pd_ = np.array(pg["dir_vehicle"], float) if pg and pg.get("valid") and pg.get("dir_vehicle") is not None else None
        fg = tr.get("face_dir")
        fd_ = np.array(fg["dir_vehicle"], float) if fg else None
        pf = (record.get("pred") or {}).get("face_dir")
        pfd = np.array(pf["dir_vehicle"], float) if pf and pf.get("valid") and pf.get("dir_vehicle") is not None else None
        r_head = _HEAD_MS * self.px_per_pt / 2
        w_gt, w_pr = max(2, round(2 * self.px_per_pt)), max(1, round(1.5 * self.px_per_pt))
        for a in self.axes:
            x0, y0, x1, y1 = a["box"]
            sub = out.crop((x0, y0, x1, y1)).convert("RGBA")    # draw clipped to the subplot
            loc = lambda p: tuple(np.subtract(self.to_px(a, p), (x0, y0)))
            hx, hy = loc(hc)
            layer = Image.new("RGBA", sub.size, (0, 0, 0, 0))
            ImageDraw.Draw(layer).ellipse([hx - r_head, hy - r_head, hx + r_head, hy + r_head], fill=HEAD_COLOR + (180,))
            sub = Image.alpha_composite(sub, layer)
            d = ImageDraw.Draw(sub)
            _arrow(d, (hx, hy), loc(hc + gd), CABIN_GT_GAZE, width=w_gt)
            if pd_ is not None:
                _arrow(d, (hx, hy), loc(hc + pd_), CABIN_PRED_GAZE, width=w_pr)
            if fd_ is not None:                                  # face direction on top of the gaze, half width
                _arrow(d, (hx, hy), loc(hc + fd_), CABIN_GT_FACE, width=_face_width(w_gt))
            if pfd is not None:
                _arrow(d, (hx, hy), loc(hc + pfd), CABIN_PRED_FACE, width=_face_width(w_pr))
            out.paste(sub.convert("RGB"), (x0, y0))
        # legend, top-left of the first subplot
        d = ImageDraw.Draw(out)
        f = _font(max(9, round(out.height / 40)))
        x0, y0 = self.axes[0]["box"][:2]
        x, y = x0 + 6, y0 + 4
        for col, label in ((CABIN_GT_GAZE, "GT gaze"), (CABIN_PRED_GAZE, "PRED gaze"), (CABIN_GT_FACE, "GT face"),
                           (CABIN_PRED_FACE, "PRED face"), (HEAD_COLOR, "head")):
            d.line([x, y + f.size // 2 + 1, x + 18, y + f.size // 2 + 1], fill=col, width=3)
            d.text((x + 22, y), label, fill=(40, 40, 40), font=f)
            y += f.size + 3
        return out


_BG_CACHE: dict = {}


def cabin_background(cabin: Cabin, cameras: list[Camera], highlight: str | None, height: int) -> CabinBackground:
    """Cached CabinBackground for this (cabin, cameras, highlight, height)."""
    key = (id(cabin), tuple(id(c) for c in cameras), highlight, int(height))
    hit = _BG_CACHE.get(key)
    if hit is not None and hit[0] is cabin and all(a is b for a, b in zip(hit[1], cameras)):
        return hit[2]
    if len(_BG_CACHE) > 16:
        _BG_CACHE.clear()
    bg = CabinBackground(cabin, cameras, highlight, int(height))
    _BG_CACHE[key] = (cabin, list(cameras), bg)
    return bg


def fig_to_image(fig) -> Image.Image:
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=80); plt.close(fig); buf.seek(0)
    return Image.open(buf).convert("RGB")


def composite(record: dict, cam: Camera, cabin: Cabin, cameras: list[Camera], scale=0.5) -> Image.Image:
    view = draw_camera_view(record, cam, cabin, scale)
    cab = cabin_background(cabin, cameras, cam.name, view.height).render(record)
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


def save_gif(frames: list[Image.Image], path: str | Path, fps: float):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)


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
    open-closed class flip, any false-positive box, any box jump. Rows that never occur are left out."""
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

"""Summary metrics comparing emulated output to ground truth."""
from __future__ import annotations
import numpy as np
from . import geometry as G

NAN = float("nan")


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else NAN


def _rate(a: int, b: int) -> float:
    return a / b if b else NAN


def _emulator_cfg(cfg: dict | None) -> dict:
    if cfg is None:
        from .config import CONFIG_DIR, load_yaml
        return load_yaml(CONFIG_DIR / "emulator.yaml")
    return cfg.get("emulator", cfg)          # full load_configs() dict or the emulator config itself


def summarize(records: list[dict], cfg: dict | None = None) -> dict:
    """Model-vs-truth metrics of simulator records. Every metric compares a record's pred with the truth of the
    same record (with latency the pred is that of an earlier frame, so the errors include the lag). cfg: the
    emulator config of the run (or the whole load_configs() dict; None = configs/emulator.yaml), used for
    gaze.valid_min_openness. Rates with an empty denominator (and every metric of an empty list) are NaN."""
    vmo = float(_emulator_cfg(cfg)["gaze"]["valid_min_openness"])
    n = len(records)
    face_truth = [r for r in records if any(d["class"] == "face" for d in r["truth"]["detections"])]
    face_pred = [r for r in records if any(d["class"] == "face" and not d.get("false_positive") for d in r["pred"]["detections"])]
    gaze_valid = [r for r in records if r["pred"]["gaze"] and r["pred"]["gaze"].get("valid")]
    gaze_err = [G.angle_between(np.array(r["pred"]["gaze"]["dir_vehicle"]), np.array(r["truth"]["gaze"]["dir_vehicle"]))
                for r in gaze_valid]
    zone_ok = [r["pred"]["gaze"]["zone"] == r["truth"]["gaze"]["zone"] for r in gaze_valid]
    face_dir = [r for r in records if r["pred"].get("face_dir") and "face_dir" in r["truth"]]
    face_err = [G.angle_between(np.array(r["pred"]["face_dir"]["dir_vehicle"]), np.array(r["truth"]["face_dir"]["dir_vehicle"]))
                for r in face_dir]
    face_zone_ok = [r["pred"]["face_dir"]["zone"] == r["truth"]["face_dir"]["zone"] for r in face_dir]
    hp = [(r["pred"]["head_pose_camera"], r["truth"]["head_pose_camera"]) for r in records if r["pred"]["head_pose_camera"]]
    hp_err = [abs(p["yaw"] - t["yaw"]) for p, t in hp]
    fw = [d["width_px"] for r in face_truth for d in r["truth"]["detections"] if d["class"] == "face"]
    ew = [d["width_px"] for r in records for d in r["truth"]["detections"] if d["class"].endswith("_eye")]
    fp = sum(1 for r in records for d in r["pred"]["detections"] if d.get("false_positive"))
    eyes_open = {r["frame"] for r in records if min(r["truth"]["eye_openness"].values()) >= vmo}
    return {
        "frames": n,
        "face_in_view_rate": _rate(len(face_truth), n),
        "face_detection_rate": _rate(len(face_pred), len(face_truth)),
        "mean_face_width_px": _mean(fw),
        "mean_eye_width_px": _mean(ew),
        "mean_abs_head_yaw_rel_camera_deg": _mean([abs(r["truth"]["head_pose_camera"]["yaw"]) for r in records]),
        "mean_gaze_to_camera_deg": _mean([r["truth"]["gaze"]["gaze_to_camera_deg"] for r in records]),
        "mean_face_occlusion": _mean([d["occlusion"] for r in records for d in r["truth"]["detections"] if d["class"] == "face"]),
        "gaze_valid_rate_eyes_open": _rate(sum(1 for r in gaze_valid if r["frame"] in eyes_open), len(eyes_open)),
        "gaze_mean_error_deg": _mean(gaze_err),
        "gaze_p95_error_deg": float(np.percentile(gaze_err, 95)) if gaze_err else float("nan"),
        "zone_accuracy": _mean(zone_ok),
        "face_dir_mean_error_deg": _mean(face_err),
        "face_zone_accuracy": _mean(face_zone_ok),
        "head_yaw_mae_deg": _mean(hp_err),
        "false_positives_per_frame": _rate(fp, n),
    }


def segments(flags) -> list[tuple[int, int]]:
    """Runs of truthy values as (first index, last index + 1) pairs."""
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i)); start = None
    if start is not None:
        out.append((start, len(flags)))
    return out


def markdown_table(rows: dict[str, dict], cols: list[str] | None = None, key_header: str = "camera") -> str:
    cols = cols or list(next(iter(rows.values())).keys())
    out = [f"| {key_header} | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, m in rows.items():
        out.append(f"| {name} | " + " | ".join(_fmt(m[c]) for c in cols) + " |")
    return "\n".join(out)


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if v != v:
            return "n/a"
        return f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}"
    return str(v)

"""Flat, pipeline-style records for DMS post-processing (see docs/SPEC.md "Output contract").

`to_pipeline` turns one simulator record (`{"frame", "t", "camera", "truth", "pred"}`) into what a model
pipeline would emit; `to_truth` gives the matching ground-truth record (same frame_id). Every emitted field /
label name lives in FIELDS and LABELS below, so matching another schema is a change in one place.

Label modes (`to_pipeline(..., labels=)`): "sim" (default) emits the simulator classes (LABELS); "mapped" emits
the consumer classes of a layout's `class_map` / `conf_threshold` (the same mapping the dms.bin writer uses, see
configs/binlog.yaml). gaze / head_pose / openness are kept in the record in both modes.
"""
from __future__ import annotations
import collections
from typing import Any

# internal key -> emitted name
FIELDS: dict[str, str] = {
    # record
    "frame_id": "frame_id", "timestamp_ms": "timestamp_ms", "camera_id": "camera_id",
    # object detection
    "od": "od", "label": "label", "bbox": "bbox", "score": "score",
    # head pose
    "head_pose": "head_pose", "head_pose_vehicle": "head_pose_vehicle",
    "yaw": "yaw", "pitch": "pitch", "roll": "roll",
    # gaze: face direction and eye gaze
    "gaze": "gaze", "face": "face", "eye": "eye", "vector": "vector", "valid": "valid",
    "yaw_vehicle": "yaw_vehicle", "pitch_vehicle": "pitch_vehicle", "vector_vehicle": "vector_vehicle",
    "zone": "zone", "target": "target",
    # openness
    "eye_openness": "eye_openness", "left": "left", "right": "right", "mouth_openness": "mouth_openness",
    # truth-only
    "blink": "blink", "closure": "closure", "hand": "hand", "phone": "phone",
    "cigarette": "cigarette", "seatbelt": "seatbelt", "held": "held", "events": "events",
    "phone_visible": "phone_visible", "observable": "observable",
}

# simulator detection class -> emitted OD label
LABELS: dict[str, str] = {
    "face": "face", "left_eye": "left_eye", "right_eye": "right_eye", "mouth": "mouth",
    "hand": "hand", "phone": "phone", "cigarette": "cigarette", "seatbelt": "seatbelt",
}

LABEL_MODES = ("sim", "mapped")

F = FIELDS

_DEFAULT_LABEL_SPEC: dict | None = None


def label_spec(spec: dict | None = None) -> dict:
    """Class-map spec ({class_map, conf_threshold, threshold keys}); None = configs/binlog.yaml (cached)."""
    global _DEFAULT_LABEL_SPEC
    if spec is None:
        if _DEFAULT_LABEL_SPEC is None:
            from .config import CONFIG_DIR, load_yaml
            _DEFAULT_LABEL_SPEC = load_yaml(CONFIG_DIR / "binlog.yaml")
        spec = _DEFAULT_LABEL_SPEC
    if not spec.get("class_map"):
        raise ValueError("labels='mapped' needs a layout with a class_map (see configs/binlog.yaml)")
    return spec


def _state_value(state: str, cls: str, pred: dict, truth: dict | None) -> float | None:
    """Predicted state used by a class_map rule; the truth value when the prediction is null."""
    truth = truth or {}
    if state == "eye_openness":
        side = "left" if cls.startswith("left") else "right"
        v = (pred.get("eye_openness") or {}).get(side)
        return v if v is not None else (truth.get("eye_openness") or {}).get(side)
    if state == "mouth_openness":
        v = pred.get("mouth_openness")
        return v if v is not None else truth.get("mouth_openness")
    raise ValueError(f"unknown class_map state {state!r} (eye_openness | mouth_openness)")


def map_class(cls: str, pred: dict, truth: dict | None, spec: dict) -> str | None:
    """Simulator class -> output class under spec["class_map"]; None = not emitted (null or not listed).
    A rule {state, threshold, below, otherwise} gives `below` when the state is strictly below the threshold
    (a number or the name of a spec key); a state that is null in pred and truth counts as not below."""
    cm = spec["class_map"]
    rule = cm.get(cls)
    if rule is None or isinstance(rule, str):
        return rule
    thr = rule["threshold"]
    thr = float(spec[thr]) if isinstance(thr, str) else float(thr)
    v = _state_value(rule["state"], cls, pred, truth)
    return rule["below"] if v is not None and float(v) < thr else rule["otherwise"]


def mapped_detections(record: dict, spec: dict, include_face: bool = True):
    """Model detections of one frame under a class map, in model order. Returns (dets, skipped, below):
    dets = [{"class": output class, "source_class": simulator class, "bbox": [x1,y1,x2,y2], "conf"}] at or above
    the class's conf_threshold; skipped = {simulator class: n} not mapped; below = {output class: n} dropped by
    the threshold. A dropped frame has no detections. include_face=False drops the face before mapping."""
    pred, truth = record["pred"], record.get("truth")
    thr = spec.get("conf_threshold") or {}
    dets, skipped, below = [], collections.Counter(), collections.Counter()
    if pred.get("dropped"):
        return dets, dict(skipped), dict(below)
    for d in pred["detections"]:
        if d["class"] == "face" and not include_face:
            continue
        out = map_class(d["class"], pred, truth, spec)
        if out is None:
            skipped[d["class"]] += 1
            continue
        conf = float(d["conf"])
        if out in thr and conf < float(thr[out]):
            below[out] += 1
            continue
        dets.append({"class": out, "source_class": d["class"], "bbox": [float(v) for v in d["bbox"]], "conf": conf})
    return dets, dict(skipped), dict(below)


def _ms(t: float) -> int:
    return int(round(float(t) * 1000.0))


def _header(record: dict, camera: Any) -> dict:
    cam_id = camera if isinstance(camera, str) else getattr(camera, "name", None) or record.get("camera")
    return {F["frame_id"]: int(record["frame"]), F["timestamp_ms"]: _ms(record["t"]), F["camera_id"]: str(cam_id)}


def _od(dets: list[dict], include_face: bool, score_key: str | None) -> list[dict]:
    out = []
    for d in dets:
        if d["class"] == "face" and not include_face:
            continue
        o = {F["label"]: LABELS.get(d["class"], d["class"]), F["bbox"]: [float(v) for v in d["bbox"]]}
        if score_key is not None:
            o[F["score"]] = float(d[score_key])
        out.append(o)
    return out


def _openness(eo: dict | None) -> dict | None:
    """{left, right}; a side is null when the model did not detect that eye (null as a whole if neither)."""
    if eo is None:
        return None
    side = lambda v: None if v is None else float(v)
    return {F["left"]: side(eo.get("left")), F["right"]: side(eo.get("right"))}


def _direction(g: dict) -> dict:
    """One direction of the pipeline gaze (face or eye): camera angles and vector, score, vehicle angles, zone;
    valid=false with null angles and score 0 when the model gives no direction."""
    if not g.get("valid"):
        return {F["yaw"]: None, F["pitch"]: None, F["vector"]: None, F["valid"]: False, F["score"]: 0.0,
                F["yaw_vehicle"]: None, F["pitch_vehicle"]: None, F["zone"]: None}
    return {F["yaw"]: float(g["yaw_camera"]), F["pitch"]: float(g["pitch_camera"]),
            F["vector"]: [float(v) for v in g["dir_camera"]], F["valid"]: True, F["score"]: float(g["conf"]),
            F["yaw_vehicle"]: float(g["yaw_vehicle"]), F["pitch_vehicle"]: float(g["pitch_vehicle"]),
            F["zone"]: str(g["zone"])}


def _truth_direction(g: dict) -> dict:
    """One direction of the truth gaze (face or eye): camera and vehicle angles and vectors, zone."""
    return {F["yaw"]: float(g["yaw_camera"]), F["pitch"]: float(g["pitch_camera"]),
            F["vector"]: [float(v) for v in g["dir_camera"]],
            F["yaw_vehicle"]: float(g["yaw_vehicle"]), F["pitch_vehicle"]: float(g["pitch_vehicle"]),
            F["vector_vehicle"]: [float(v) for v in g["dir_vehicle"]], F["zone"]: str(g["zone"])}


def to_pipeline(record: dict, camera: Any = None, include_face: bool | None = None, labels: str = "sim",
                label_map: dict | None = None) -> dict:
    """Model-like output for one frame. Dropped frames: empty od and null fields.

    labels: "sim" (simulator classes) or "mapped" (consumer classes via label_map, a layout dict with a
    class_map, see mapped_detections). include_face: emit the face box; None = no for "sim", yes for "mapped".
    A dropped frame is `od: []` with the other fields null in both modes.
    gaze is {face, eye}: the face direction (where the head points) and the eye gaze, each with the same fields.
    gaze is null when there is no face (or the frame is dropped). With a face, gaze.face is always valid; when the
    gaze model refuses the eye gaze (eyes closed / not visible), gaze.eye has valid=false, null angles, score 0.
    A record stored before the face direction existed (pred without face_dir) gives gaze.face null.
    eye_openness sides / mouth_openness are null when that eye / the mouth was not detected.
    """
    if labels not in LABEL_MODES:
        raise ValueError(f"labels must be one of {LABEL_MODES}, got {labels!r}")
    if include_face is None:
        include_face = labels == "mapped"
    pred = record["pred"]
    out = _header(record, camera)
    if pred.get("dropped"):
        out.update({F["od"]: [], F["head_pose"]: None, F["gaze"]: None,
                    F["eye_openness"]: None, F["mouth_openness"]: None})
        return out

    if labels == "mapped":
        dets, _, _ = mapped_detections(record, label_spec(label_map), include_face)
        out[F["od"]] = [{F["label"]: d["class"], F["bbox"]: d["bbox"], F["score"]: d["conf"]} for d in dets]
    else:
        out[F["od"]] = _od(pred["detections"], include_face, "conf")

    hp = pred.get("head_pose_camera")
    out[F["head_pose"]] = None if hp is None else {
        F["yaw"]: float(hp["yaw"]), F["pitch"]: float(hp["pitch"]), F["roll"]: float(hp["roll"]),
        F["score"]: float(hp["conf"])}

    g, fd = pred.get("gaze"), pred.get("face_dir")
    if g is None or (not g.get("valid") and g.get("reason") == "no_face"):
        out[F["gaze"]] = None
    else:                                    # a record stored before face_dir existed: face null, eye as before
        out[F["gaze"]] = {F["face"]: None if fd is None else _direction(fd), F["eye"]: _direction(g)}

    out[F["eye_openness"]] = _openness(pred.get("eye_openness"))
    mo = pred.get("mouth_openness")
    out[F["mouth_openness"]] = None if mo is None else float(mo)
    return out


def to_truth(record: dict, camera: Any = None) -> dict:
    """Ground truth for one frame, same frame_id as the pipeline record, plus scripted behaviour labels."""
    tr = record["truth"]
    out = _header(record, camera)
    out[F["od"]] = _od(tr["detections"], True, None)
    hp, hv = tr["head_pose_camera"], tr["head_pose_vehicle"]
    out[F["head_pose"]] = {F["yaw"]: float(hp["yaw"]), F["pitch"]: float(hp["pitch"]), F["roll"]: float(hp["roll"])}
    out[F["head_pose_vehicle"]] = {F["yaw"]: float(hv["yaw"]), F["pitch"]: float(hv["pitch"]), F["roll"]: float(hv["roll"])}
    g = tr["gaze"]
    out[F["gaze"]] = {F["face"]: _truth_direction(tr["face_dir"]),
                      F["eye"]: {**_truth_direction(g), F["target"]: str(g["target"])}}
    out[F["eye_openness"]] = _openness(tr["eye_openness"])
    out[F["mouth_openness"]] = float(tr.get("mouth_openness", 0.0))
    out[F["blink"]] = bool(tr["blink"]); out[F["closure"]] = bool(tr["closure"])
    out[F["hand"]] = tr["hand"]; out[F["phone"]] = tr["phone"]
    out[F["cigarette"]] = tr.get("cigarette", "none"); out[F["seatbelt"]] = tr.get("seatbelt", "on")
    out[F["held"]] = tr.get("held", "none")
    if "phone_visible" in tr:
        out[F["phone_visible"]] = bool(tr["phone_visible"])
    if "observable" in tr:
        out[F["observable"]] = {k: bool(v) for k, v in tr["observable"].items()}
    out[F["events"]] = dict(tr.get("events", {}))
    return out

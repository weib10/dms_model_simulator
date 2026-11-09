"""Per-frame behaviour labels derived from the scripted driver state (never from model output).

These describe what the driver does, not what an alert logic should decide: there are no duration thresholds
here (no "distracted after N s"). Keys in configs/behavior.yaml under ``events``:
- eyes_off_road: scripted gaze target not in ``on_road_targets``, or (with a cabin) the actual gaze direction
                 outside the road box: the yaw / pitch range of the on-road targets widened by the zone tolerance
                 (``zone_classifier.tolerance_deg``) for at least ``gaze_miss_min_s``. The second part is a
                 safety net for a head turned so far that the eyes cannot reach the scripted road target;
                 sweeping between road targets stays inside the box, and the short time the gaze needs to
                 come back (a saccade) is shorter than gaze_miss_min_s, so scripted durations are unchanged.
- off_road_s:    current continuous eyes-off-road duration (s), counting the current frame.
- eyes_closed:   mean scripted eye openness strictly below ``closed_openness_threshold``.
- closed_s:      current continuous eyes-closed duration (s), counting the current frame.
- phone_use:     phone in ``phone_states`` or hand in ``phone_hand_states``.
- yawning:       scripted yawn in progress.
- smoking:       cigarette in ``smoking_cigarette_states`` (i.e. not none).
- seatbelt_off:  seatbelt in ``seatbelt_off_states``.
"""
from __future__ import annotations
from .behavior import DriverFrame, on_off
from . import geometry as G
from .world import Cabin


def road_box(cabin: Cabin, on_road) -> tuple[float, float, float, float] | None:
    """(yaw_lo, yaw_hi, pitch_lo, pitch_hi) degrees, seen from the nominal eye point (vehicle origin): the on-road
    targets' directions widened by the zone tolerance. None when no on-road target is in the cabin."""
    yp = [G.dir_to_yaw_pitch(cabin.targets[t]) for t in on_road if t in cabin.targets]
    if not yp:
        return None
    tol = float(cabin.zone_tol)
    ys, ps = [y for y, _ in yp], [p for _, p in yp]
    return min(ys) - tol, max(ys) + tol, min(ps) - tol, max(ps) + tol


def _in_box(point, box) -> bool:
    y, p = G.dir_to_yaw_pitch(point)
    return box[0] <= y <= box[1] and box[2] <= p <= box[3]


def label_events(frames: list[DriverFrame], cabin: Cabin | None, cfg: dict) -> list[dict]:
    """One label dict per frame. ``cfg`` is the behaviour config (uses ``fps`` and ``events``).
    ``cabin`` gives the road box for the gaze check of eyes_off_road; None = scripted targets only."""
    ev = cfg["events"]
    dt = 1.0 / float(cfg["fps"])
    on_road = set(ev["on_road_targets"])
    thr = float(ev["closed_openness_threshold"])
    phone_states, hand_states = set(ev["phone_states"]), set(ev["phone_hand_states"])
    smoke_states = set(ev["smoking_cigarette_states"])
    belt_off_states = {on_off(v) for v in ev["seatbelt_off_states"]}
    box = road_box(cabin, on_road) if cabin is not None else None
    named_off = [f.target_name not in on_road for f in frames]
    miss = [box is not None and not n and not _in_box(f.gaze_point, box) for f, n in zip(frames, named_off)]
    min_run = int(round(float(ev.get("gaze_miss_min_s", 0.3)) / dt))
    i = 0
    while i < len(miss):                    # keep only gaze misses that last at least gaze_miss_min_s
        if miss[i]:
            j = i
            while j < len(miss) and miss[j]:
                j += 1
            if j - i < min_run:
                miss[i:j] = [False] * (j - i)
            i = j
        else:
            i += 1
    off_road_s, closed_s = 0.0, 0.0
    out = []
    for f, n, m in zip(frames, named_off, miss):
        off = n or m
        off_road_s = off_road_s + dt if off else 0.0
        closed = 0.5 * (f.openness_left + f.openness_right) < thr
        closed_s = closed_s + dt if closed else 0.0
        out.append({
            "eyes_off_road": off,
            "off_road_s": round(off_road_s, 6),
            "eyes_closed": closed,
            "closed_s": round(closed_s, 6),
            "phone_use": f.phone in phone_states or f.hand in hand_states,
            "yawning": bool(f.yawning),
            "smoking": getattr(f, "cigarette", "none") in smoke_states,
            "seatbelt_off": on_off(getattr(f, "seatbelt", "on")) in belt_off_states,
        })
    return out

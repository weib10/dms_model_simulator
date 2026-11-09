"""Per-frame behaviour labels derived from the scripted driver state (never from model output).

These describe what the driver does, not what an alert logic should decide: there are no duration thresholds
here (no "distracted after N s"). Keys in configs/behavior.yaml under ``events``:
- eyes_off_road: scripted gaze target not in ``on_road_targets``.
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
from .world import Cabin


def label_events(frames: list[DriverFrame], cabin: Cabin | None, cfg: dict) -> list[dict]:
    """One label dict per frame. ``cfg`` is the behaviour config (uses ``fps`` and ``events``).
    ``cabin`` is accepted for future geometric definitions and may be None."""
    ev = cfg["events"]
    dt = 1.0 / float(cfg["fps"])
    on_road = set(ev["on_road_targets"])
    thr = float(ev["closed_openness_threshold"])
    phone_states, hand_states = set(ev["phone_states"]), set(ev["phone_hand_states"])
    smoke_states = set(ev["smoking_cigarette_states"])
    belt_off_states = {on_off(v) for v in ev["seatbelt_off_states"]}
    off_road_s, closed_s = 0.0, 0.0
    out = []
    for f in frames:
        off = f.target_name not in on_road
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

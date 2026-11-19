"""Drive side: left-hand drive (the configs as written) or right-hand drive (the whole world mirrored).

The vehicle frame has its origin at the driver's eye point with Y to the left, so a right-hand-drive car is the
left-hand-drive one mirrored in the plane y = 0: every cabin point, camera mount and scripted head turn is
mirrored, and a scenario written for left-hand drive plays as its mirror image (looking at the passenger still
looks at the passenger, now on the left; the driver-side mirror is now the right one).

What mirrors (y -> -y for points and directions; yaw and roll change sign, pitch stays):
- cabin: gaze targets, fixed hand positions, head-frame hand / phone / cigarette / held-object offsets and
  directions, seat belt, occluders, the driver's eye-point offset;
- cameras: positions, aim offsets, fixed {yaw, pitch} orientations, mounting errors;
- scenario: scripted head poses, the rest head pose.

Names: only the gaze targets that come in left / right pairs swap (left_mirror <-> right_mirror, road_left <->
road_right), so the output zone names stay physically right. Hand states, occluders and camera presets keep
their names as IDs: in right-hand drive `dash_right` is the dash camera beside the wheel on the passenger side
(left of the wheel), `chest_left` the hand across the chest at the shoulder strap, which is then the right one.

What does not mirror: the face (landmarks, face outline, the nose as self-occluder: the driver's left eye is still
the left eye) and the model-error config. The driver's working hand mirrors with the cabin (a phone call is at
the left ear), so a right-hand-drive run is exactly the mirror image of the left-hand-drive one.
"""
from __future__ import annotations
import copy

DRIVES = ("left", "right")
NAME_SUFFIX = {"left": "", "right": "_rhd"}      # appended to the scenario name, so outputs do not collide


def check(drive: str) -> str:
    if drive not in DRIVES:
        raise ValueError(f"drive must be one of {DRIVES}, got {drive!r}")
    return drive


def mirror_point(p) -> list:
    """[x, y, z] -> [x, -y, z] (a point or a direction)."""
    x, y, z = (float(v) for v in p)
    return [x, -y, z]


def mirror_angles(a) -> list:
    """[yaw, pitch, roll] -> [-yaw, pitch, -roll]."""
    y, p, r = a
    return [_neg(y), p, _neg(r)]


def _neg(v):
    return -float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def mirror_target_name(name):
    """left_mirror <-> right_mirror, road_left <-> road_right; every other name unchanged."""
    if not isinstance(name, str):
        return name
    parts = name.split("_")
    swapped = ["right" if p == "left" else "left" if p == "right" else p for p in parts]
    return "_".join(swapped)


def _point_or_keep(v):
    """A 3-number list is mirrored; anything else (e.g. "head_frame", "hand") is kept."""
    if isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
        return mirror_point(v)
    return v


def mirror_cabin(cfg: dict) -> dict:
    """configs/cabin.yaml (or a partial `cabin:` block) mirrored; keys that are absent stay absent."""
    out = copy.deepcopy(cfg)
    if "targets" in out:
        out["targets"] = {mirror_target_name(k): mirror_point(v) for k, v in out["targets"].items()}
    drv = out.get("driver") or {}
    for k in ("eye_point_offset", "head_pivot", "head_center_offset"):
        if k in drv:
            drv[k] = mirror_point(drv[k])
    if "hands" in out:
        out["hands"] = {k: _point_or_keep(v) for k, v in out["hands"].items()}
    for k in list(out):
        if k.endswith("_offset") and isinstance(out[k], list):          # head-frame hand / phone offsets
            out[k] = mirror_point(out[k])
    for k in ("cigarette_mouth", "cigarette_hand"):
        for kk in ("start", "dir", "offset"):
            if kk in (out.get(k) or {}):
                out[k][kk] = mirror_point(out[k][kk])
    for h in (out.get("held_objects") or {}).values():
        for kk in ("start", "dir"):
            if kk in h:
                h[kk] = mirror_point(h[kk])
    sb = out.get("seatbelt_strap") or {}
    for kk in ("top", "bottom"):
        if kk in sb:
            sb[kk] = mirror_point(sb[kk])
    if "pillar_path" in sb:
        sb["pillar_path"] = [mirror_point(p) for p in sb["pillar_path"]]
    for oc in out.get("occluders") or []:
        for kk in ("center", "normal", "a", "b"):
            if kk in oc:
                oc[kk] = _point_or_keep(oc[kk])
    return out


def mirror_camera_spec(spec: dict) -> dict:
    """One camera preset (or the `defaults` block, or a scenario / variant `camera:` block) mirrored."""
    out = copy.deepcopy(spec)
    for k in ("position", "look_at_offset", "mount_offset"):
        if k in out:
            out[k] = mirror_point(out[k])
    if isinstance(out.get("orientation"), dict) and "yaw" in out["orientation"]:
        out["orientation"]["yaw"] = _neg(out["orientation"]["yaw"])
    if "mount_rotation_deg" in out:
        out["mount_rotation_deg"] = mirror_angles(out["mount_rotation_deg"])
    return out


def mirror_cameras(cfg: dict) -> dict:
    """configs/cameras.yaml mirrored (camera names kept)."""
    out = copy.deepcopy(cfg)
    if "defaults" in out:
        out["defaults"] = mirror_camera_spec(out["defaults"])
    out["cameras"] = {k: mirror_camera_spec(v) for k, v in (out.get("cameras") or {}).items()}
    return out


def mirror_behavior(cfg: dict) -> dict:
    """configs/behavior.yaml (or a scenario's partial `behavior:` block) mirrored."""
    out = copy.deepcopy(cfg)
    if "rest_head_pose" in out:
        out["rest_head_pose"] = mirror_angles(out["rest_head_pose"])
    if "phone" in out:
        out["phone"] = mirror_target_name(out["phone"])
    ev = out.get("events") or {}
    if "on_road_targets" in ev:
        ev["on_road_targets"] = [mirror_target_name(t) for t in ev["on_road_targets"]]
    return out


def mirror_segment(seg: dict) -> dict:
    out = copy.deepcopy(seg)
    if "target" in out:
        out["target"] = mirror_target_name(out["target"])
    if "phone" in out:                                    # lap | call | a target name
        out["phone"] = mirror_target_name(out["phone"])
    if "head_pose" in out:
        out["head_pose"] = mirror_angles(out["head_pose"])
    if "segments" in out:                                 # an unexpanded repeat block
        out["segments"] = [mirror_segment(s) for s in out["segments"]]
    return out


def mirror_scenario(sc: dict) -> dict:
    """Segments and the `behavior:` block of a scenario mirrored (its `cabin:` / `camera:` blocks are merged into the
    configs before mirroring, see Simulation)."""
    out = copy.deepcopy(sc)
    out["segments"] = [mirror_segment(s) for s in out.get("segments") or []]
    if out.get("behavior"):
        out["behavior"] = mirror_behavior(out["behavior"])
    return out

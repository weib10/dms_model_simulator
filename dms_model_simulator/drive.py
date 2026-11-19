"""Drive side: left-hand drive (the configs as written) or right-hand drive (the whole world mirrored).

The vehicle frame has its origin at the driver's eye point with Y to the left, so a right-hand-drive car is the
left-hand-drive one mirrored in the plane y = 0: every cabin point, camera mount and scripted head turn is
mirrored, and a scenario written for left-hand drive plays as its mirror image (looking at the passenger still
looks at the passenger, now on the left; the driver-side mirror is now the right one).

What mirrors (y -> -y for points and directions; yaw and roll change sign, pitch stays):
- cabin: gaze targets, fixed hand positions, head-frame hand / phone / cigarette / held-object offsets and
  directions, seat belt, occluders, the driver's eye-point offset;
- car body (configs/body.yaml): not its config keys but the built geometry. The cabin config is marked `drive:
  right`; Cabin builds the body from the left-hand-drive targets and eye point and mirrors the triangles
  (body_occluder, body_parts), so whatever body.yaml says (parts following the driver or the seat included) comes
  out mirrored and keeps its part names;
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
    """configs/cabin.yaml mirrored; keys that are absent stay absent. The car body (`body`) is kept as written and
    the result is marked `drive: right` (removed again by a second mirror): Cabin mirrors the body once it is
    built (body_occluder)."""
    out = copy.deepcopy(cfg)
    if out.get("drive", "left") == "left":
        out["drive"] = "right"
    else:
        out.pop("drive")
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


def _lhd_inputs(targets: dict, eye):
    """The gaze targets (names and points) and eye point of a mirrored cabin as the left-hand-drive configs have
    them: what configs/body.yaml is written against."""
    import numpy as np
    return ({mirror_target_name(k): np.array(mirror_point(v)) for k, v in targets.items()},
            np.array(mirror_point(eye)))


def _mirror_tris(tris):
    """Triangles (N x 3 x 3) mirrored y -> -y; two vertices swap so the front side stays the front side."""
    import numpy as np
    t = np.array(tris, float)[:, [0, 2, 1]]
    t[..., 1] *= -1.0
    return t


def body_parts(body: dict, targets: dict, eye, drive: str = "left") -> list[dict]:
    """body.parts for a cabin of this drive side (targets and eye point as the Cabin has them): in right-hand
    drive the left-hand-drive parts mirrored, names kept."""
    from . import body as B
    if check(drive) == "left":
        return B.parts(body, targets, eye)
    return [dict(p, tris=list(_mirror_tris(p["tris"])) if len(p["tris"]) else p["tris"])
            for p in B.parts(body, *_lhd_inputs(targets, eye))]


def body_occluder(body: dict, targets: dict, eye, drive: str = "left") -> dict | None:
    """body.occluder for a cabin of this drive side: in right-hand drive the left-hand-drive mesh mirrored (the
    part name and two-sidedness of each triangle kept)."""
    from . import body as B
    if check(drive) == "left":
        return B.occluder(body, targets, eye)
    occ = B.occluder(body, *_lhd_inputs(targets, eye))
    return None if occ is None else dict(occ, tris=_mirror_tris(occ["tris"]))


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


# ---- text: the catalog's words for a right-hand-drive run ----------------------------------------------------
# Entry texts are written for left-hand drive. In a right-hand-drive run every direction word swaps, as the world
# does: "head turned right 60°" is a turn to the left, "right hand" the left hand. Kept: 左右 ("left and right", or
# "about"), English left / right as a verb or adverb (_EN_KEEP), and identifiers that are not gaze targets (camera
# presets such as dash_right, hand states such as chest_left, parameter names such as left_s), which keep their
# names in right-hand drive.
_EN_KEEP = (r"left(?= (?:out|in the mouth|to the emulator)\b)", r"(?<=is )left(?=[;,.])",
            r"right(?= (?:next to|on the mirror|at the start)\b)")
_A = r"(?<![A-Za-z0-9_])"          # ASCII word edges: a Chinese character next to an English word is an edge too
_Z = r"(?![A-Za-z0-9_])"
_TEXT_RE = None


def _text_re():
    import re
    global _TEXT_RE
    if _TEXT_RE is None:
        _TEXT_RE = re.compile("|".join([
            rf"(?P<ident>{_A}[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+{_Z})",   # identifier: swapped only if a gaze target
            rf"(?P<keep>左右|{_A}(?:" + "|".join(_EN_KEEP) + "))",
            r"(?P<zh>[左右])",
            rf"(?P<en>{_A}(?:left|right|Left|Right|LEFT|RIGHT){_Z})",
        ]))
    return _TEXT_RE


_EN_SWAP = {"left": "right", "right": "left", "Left": "Right", "Right": "Left", "LEFT": "RIGHT", "RIGHT": "LEFT"}


def mirror_text(text, targets=()):
    """A left-hand-drive description as it reads in right-hand drive: 左 <-> 右, left <-> right, and gaze-target
    names in `targets` swapped like the targets (left_mirror -> right_mirror). Non-strings are returned as they are."""
    if not isinstance(text, str):
        return text
    targets = set(targets)

    def rep(m):
        if m.group("ident"):
            w = m.group("ident")
            return mirror_target_name(w) if w in targets else w
        if m.group("keep"):
            return m.group("keep")
        if m.group("zh"):
            return "右" if m.group("zh") == "左" else "左"
        return _EN_SWAP[m.group("en")]
    return _text_re().sub(rep, text)


def target_names(*trees) -> set:
    """Every gaze-target name in config / scenario / catalog-entry trees: keys of a `targets:` mapping and values of
    `target:` (the names mirror_text may swap)."""
    out = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "targets" and isinstance(v, dict):
                    out.update(str(t) for t in v)
                elif k == "target" and isinstance(v, str):
                    out.add(v)
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    for t in trees:
        walk(t)
    return out


ENTRY_TEXT_FIELDS = ("title", "title_en", "summary", "summary_en", "short", "short_en", "brief", "brief_en")


def mirror_entry_text(entry: dict, cabin_cfg: dict | None = None) -> dict:
    """A catalog entry with its texts (title, summary, short, brief; both languages) as they read in right-hand drive
    (mirror_text). Gaze-target names: those of cabin_cfg (default configs/cabin.yaml) and of the entry itself."""
    if cabin_cfg is None:
        from .config import load_configs
        cabin_cfg = load_configs()["cabin"]
    targets = target_names(cabin_cfg, entry)
    out = dict(entry)
    for k in ENTRY_TEXT_FIELDS:
        v = entry.get(k)
        out[k] = {kk: mirror_text(vv, targets) for kk, vv in v.items()} if isinstance(v, dict) else mirror_text(v, targets)
        if v is None:
            out.pop(k)
    return out

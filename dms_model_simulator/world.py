"""Cabin geometry, driver head model and occluders."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from . import geometry as G


@dataclass
class HeadState:
    """Pose of the head in the vehicle frame."""
    yaw: float
    pitch: float
    roll: float
    R: np.ndarray                # vehicle_from_head
    origin: np.ndarray           # eye midpoint, vehicle frame
    landmarks: dict[str, np.ndarray] = field(default_factory=dict)   # vehicle frame

    def head_to_vehicle(self, p_head: np.ndarray) -> np.ndarray:
        return self.origin + self.R @ np.asarray(p_head, float)


class Cabin:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.targets = {k: np.array(v, float) for k, v in cfg["targets"].items()}
        self.zone_tol = float(cfg["zone_classifier"]["tolerance_deg"])
        self.default_zone = cfg["zone_classifier"]["default_zone"]
        d = cfg["driver"]
        self.eye_point = np.array(d["eye_point_offset"], float)
        # head_scale scales every head-frame length (landmarks, pivot, lids, lips, head-frame hand / phone /
        # cigarette offsets, self-occluders); 1 = the template as written
        s = self.head_scale = float(d.get("head_scale", 1.0))
        if not s > 0:
            raise ValueError(f"driver.head_scale must be > 0, got {d.get('head_scale')!r}")
        # fixed camera mounts aim at the reference driver (no eye_point_offset, head_scale 1), not at this one
        self.reference_head_center = np.array(d["head_pivot"], float) + np.array(d["head_center_offset"], float)
        self.ipd = s * float(d["ipd"])
        self.pivot = s * np.array(d["head_pivot"], float)
        self.head_center_offset = s * np.array(d["head_center_offset"], float)
        self.template = {k: s * np.array(v, float) for k, v in cfg["landmarks"].items()}
        self.face_outline = list(cfg["face_outline"])
        self.eye_half_height = s * float(cfg["eye_half_height"])
        self.eye_min_openness_frac = float(cfg["eye_min_openness_frac"])
        self.mouth_half_height = s * float(cfg["mouth_half_height"])
        self.mouth_min_openness_frac = float(cfg["mouth_min_openness_frac"])
        self.face_bbox_pad = float(cfg["face_bbox_pad"])
        self.box_style = cfg.get("box_style", "geometric")
        if self.box_style not in ("annotation", "geometric"):
            raise ValueError(f"box_style must be annotation | geometric, got {self.box_style!r}")
        self.annotation_boxes = dict(cfg.get("annotation_boxes", {}))
        for k in ("eye_top", "eye_bottom", "mouth_top", "mouth_below_lower_lip", "mouth_side_extend"):
            if k in self.annotation_boxes:                      # head-frame lengths
                self.annotation_boxes[k] = s * float(self.annotation_boxes[k])
        need = {"eye_top", "eye_bottom", "mouth_top", "mouth_below_lower_lip", "mouth_side_extend"}
        if self.box_style == "annotation" and not need <= set(self.annotation_boxes):
            raise ValueError(f"box_style annotation needs annotation_boxes keys {sorted(need - set(self.annotation_boxes))}")
        lv = cfg["landmark_visibility"]
        self.lm_normal_forward_bias = s * float(lv["normal_forward_bias"])      # head-frame length
        self.lm_min_facing_cos = float(lv["min_facing_cos"])
        self.hands = cfg["hands"]
        self.call_offset = s * np.array(cfg["call_offset"], float)
        self.phone_call_offset = s * np.array(cfg["phone_call_offset"], float)
        # head-frame hand states ("head_frame") read their position from "<state>_offset"
        self.hand_offsets = {k: s * np.array(cfg[f"{k}_offset"], float) for k, v in self.hands.items() if v == "head_frame"}
        # hand_scale scales the hand: hand box size and the radius of every occluder tied to a hand state
        # (hand spheres, forearms); hand positions stay where the config puts them
        hs = self.hand_scale = float(d.get("hand_scale", 1.0))
        if not hs > 0:
            raise ValueError(f"driver.hand_scale must be > 0, got {d.get('hand_scale')!r}")
        self.hand_size = hs * float(cfg["hand_size"])
        self.phone_size = tuple(cfg["phone_size"])
        self.cigarette_length, self.cigarette_diameter = (float(v) for v in cfg["cigarette_size"])
        self.cigarette_mouth = {k: np.array(v, float) for k, v in cfg["cigarette_mouth"].items()}
        self.cigarette_mouth["start"] = s * self.cigarette_mouth["start"]          # head-frame point; dir is a direction
        self.cigarette_hand = {k: np.array(v, float) for k, v in cfg["cigarette_hand"].items()}
        sb = cfg["seatbelt_strap"]
        self.seatbelt_top = self.eye_point + np.array(sb["top"], float)
        self.seatbelt_bottom = self.eye_point + np.array(sb["bottom"], float)
        self.seatbelt_pillar = [self.eye_point + np.array(v, float) for v in sb.get("pillar_path", [])]
        self.seatbelt_width = float(sb["width"])
        self.seatbelt_edge_samples = max(2, int(sb["edge_samples"]))
        self.seatbelt_occlusion_samples = max(1, int(sb["occlusion_samples"]))
        self.occluders = [dict(o, radius=hs * float(o["radius"])) if "active_when_hand" in o and "radius" in o else o
                          for o in cfg.get("occluders", [])]
        self.held_objects = {k: {"start": s * np.array(v["start"], float), "dir": np.array(v["dir"], float),
                                 "length": float(v["length"]), "radius": float(v["radius"]),
                                 "needs_hand": v.get("needs_hand")}
                             for k, v in (cfg.get("held_objects") or {}).items()}
        for k, h in self.held_objects.items():
            if not h["length"] > 2 * h["radius"] > 0:
                raise ValueError(f"held_objects.{k}: length must exceed its diameter (2 x radius > 0)")
        oc_cfg = cfg.get("occlusion", {})
        self.occlusion_eps = float(oc_cfg.get("surface_eps", 0.0))
        self.torus_segments = int(oc_cfg.get("torus_segments", 36))
        # parts of the head that hide other landmarks (self-occlusion, head frame); see compute_truth
        self.self_occluders = [{**so, "radius": s * float(so["radius"]),
                                **{k: (so[k] if isinstance(so[k], str) else (s * np.array(so[k], float)).tolist())
                                   for k in ("a", "b")}} for so in cfg.get("self_occluders", [])]

    # ---- head ---------------------------------------------------------------
    def head_pose(self, yaw: float, pitch: float, roll: float) -> tuple[np.ndarray, np.ndarray]:
        """(R vehicle_from_head, eye midpoint) of a head pose: the head rotates about the pivot."""
        R = G.euler_to_R(yaw, pitch, roll)
        return R, self.eye_point + self.pivot + R @ (-self.pivot)

    def head_state(self, yaw: float, pitch: float, roll: float) -> HeadState:
        """Place the head: landmarks rotate about the pivot, so the eye midpoint moves."""
        R, origin = self.head_pose(yaw, pitch, roll)
        base = self.eye_point + self.pivot          # pivot in vehicle frame
        lms = {k: base + R @ (v - self.pivot) for k, v in self.template.items()}
        return HeadState(yaw, pitch, roll, R, origin, lms)

    def nominal_head_center(self) -> np.ndarray:
        return self.eye_point + self.pivot + self.head_center_offset

    # ---- zones --------------------------------------------------------------
    def classify_zone(self, origin: np.ndarray, direction: np.ndarray) -> tuple[str, float]:
        best, best_ang = self.default_zone, 1e9
        for name, p in self.targets.items():
            ang = G.angle_between(direction, p - origin)
            if ang < best_ang:
                best, best_ang = name, ang
        if best_ang > self.zone_tol:
            return self.default_zone, best_ang
        return best, best_ang

    # ---- hands / phone --------------------------------------------------------
    def hand_position(self, hand: str, head: HeadState) -> np.ndarray | None:
        if hand in (None, "none"):
            return None
        spec = self.hands[hand]
        if spec == "head_frame":
            return head.head_to_vehicle(self.hand_offsets[hand])
        return np.array(spec, float)

    def phone_position(self, phone: str, head: HeadState) -> np.ndarray | None:
        if phone in (None, "none"):
            return None
        if phone == "lap":
            return self.targets["phone_lap"]
        if phone == "call":
            return head.head_to_vehicle(self.phone_call_offset)
        return np.array(self.targets[phone], float)

    @staticmethod
    def _head_frame_segment(head: HeadState, start: np.ndarray, direction: np.ndarray, d0: float,
                            d1: float) -> tuple[np.ndarray, np.ndarray]:
        """Vehicle-frame points d0 and d1 metres along a head-frame direction from a head-frame start point."""
        a, u = head.head_to_vehicle(start), head.R @ G.normalize(direction)
        return a + d0 * u, a + d1 * u

    def cigarette_segment(self, cigarette: str, hand: str, head: HeadState) -> tuple[np.ndarray, np.ndarray] | None:
        """(filter end, tip) of the cigarette in the vehicle frame; None when absent (or 'hand' with no hand)."""
        if cigarette in (None, "none"):
            return None
        L = self.cigarette_length
        if cigarette == "mouth":
            return self._head_frame_segment(head, self.cigarette_mouth["start"], self.cigarette_mouth["dir"], 0.0, L)
        if cigarette == "hand":
            hp = self.hand_position(hand, head)
            if hp is None:
                return None
            # offset / dir are vehicle frame for a fixed hand, head frame for a hand that moves with the head
            R = head.R if self.hands[hand] == "head_frame" else np.eye(3)
            c = hp + R @ self.cigarette_hand["offset"]
            d = 0.5 * L * (R @ G.normalize(self.cigarette_hand["dir"]))
            return c - d, c + d
        raise ValueError(f"unknown cigarette state {cigarette!r} (none | mouth | hand)")

    # ---- seatbelt -------------------------------------------------------------
    def seatbelt_quad(self, a: np.ndarray | None = None, b: np.ndarray | None = None) -> np.ndarray:
        """Band corners (4x3, vehicle frame): top-left, bottom-left, bottom-right, top-right of the belt piece
        a -> b (default: the shoulder strap top -> bottom)."""
        a = self.seatbelt_top if a is None else a
        b = self.seatbelt_bottom if b is None else b
        across = np.cross(G.normalize(b - a), [1.0, 0.0, 0.0])          # in the chest plane, across the strap
        if np.linalg.norm(across) < 1e-9:                               # piece along X: any perpendicular
            across = np.array([0.0, 0.0, 1.0])
        across = 0.5 * self.seatbelt_width * G.normalize(across)
        return np.array([a + across, b + across, b - across, a - across])

    def seatbelt_labelled_pieces(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Belt pieces in the labelled seatbelt box: the shoulder strap and, in box_style annotation, the belt on
        the B-pillar (pillar_path, joined to the strap top)."""
        pieces = [(self.seatbelt_top, self.seatbelt_bottom)]
        if self.box_style == "annotation" and self.seatbelt_pillar:
            path = self.seatbelt_pillar + [self.seatbelt_top]
            pieces += list(zip(path[:-1], path[1:]))
        return pieces

    # ---- occlusion ----------------------------------------------------------
    def held_segment(self, held: str, head: HeadState) -> tuple[np.ndarray, np.ndarray, float] | None:
        """(a, b, radius) of a held object's capsule in the vehicle frame (ray tests treat a cylinder occluder as a
        capsule: segment a-b plus radius, rounded ends): the segment is pulled in by the radius at both ends, so the
        rear cap stops at the rim at the lips (`start`) and the whole object is `length` long. None for none."""
        if held in (None, "none"):
            return None
        h = self.held_objects[held]
        r = h["radius"]
        return (*self._head_frame_segment(head, h["start"], h["dir"], r, h["length"] - r), r)

    def occluder_geometry(self, hand: str, head: HeadState, held: str = "none") -> list[dict]:
        """Resolve occluders into world-space primitives for the current frame.
        torus: polyline of the rim; sphere: center_world (center "head_frame" / "hand" = at the hand position of
        active_when_hand, else a vehicle point); disc: center/normal/radius; cylinder: segment a-b + radius.
        A held object (held_segment) is a cylinder named held_<object>."""
        out = []
        seg = self.held_segment(held, head)
        if seg is not None:
            out.append({"name": f"held_{held}", "type": "cylinder", "a_world": seg[0], "b_world": seg[1],
                        "radius": seg[2]})
        for oc in self.occluders:
            if "active_when_hand" in oc and oc["active_when_hand"] != hand:
                continue
            o = dict(oc)
            if oc["type"] == "torus":
                c = np.array(oc["center"], float)
                n = G.normalize(np.array(oc["normal"], float))
                a = G.normalize(np.cross(n, [0, 0, 1.0])) if abs(n[2]) < 0.99 else np.array([1.0, 0, 0])
                b = np.cross(n, a)
                ang = np.linspace(0, 2 * np.pi, self.torus_segments + 1)
                o["polyline"] = c + oc["radius"] * (np.outer(np.cos(ang), a) + np.outer(np.sin(ang), b))
            elif oc["type"] == "sphere":
                if oc["center"] in ("head_frame", "hand"):
                    h = oc.get("active_when_hand")
                    p = self.hand_position(h, head) if h else None
                    o["center_world"] = p if p is not None else head.head_to_vehicle(self.call_offset)
                else:
                    o["center_world"] = np.array(oc["center"], float)
            elif oc["type"] == "disc":
                o["center_world"] = np.array(oc["center"], float)
                o["normal_world"] = G.normalize(np.array(oc["normal"], float))
            elif oc["type"] == "cylinder":
                o["a_world"], o["b_world"] = np.array(oc["a"], float), np.array(oc["b"], float)
            else:
                raise ValueError(f"unknown occluder type {oc['type']!r} (torus | sphere | disc | cylinder)")
            out.append(o)
        return out

    def self_occluder_geometry(self, head: HeadState) -> list[dict]:
        """Head-frame self-occluders (e.g. the nose ridge) as vehicle-frame cylinders for this head pose."""
        def pt(v):
            return head.head_to_vehicle(self.template[v] if isinstance(v, str) else np.array(v, float))
        return [{"name": so["name"], "type": "cylinder", "a_world": pt(so["a"]), "b_world": pt(so["b"]),
                 "radius": float(so["radius"]), "applies_to": set(so["applies_to"])} for so in self.self_occluders]

    @staticmethod
    def ray_blocked(cam_pos: np.ndarray, point: np.ndarray, occluders: list[dict], eps: float = 0.0) -> str | None:
        """Name of the first occluder the ray camera -> point meets strictly before the point (more than eps short
        of it), else None. A point inside a sphere is hidden by it; a point on an occluder surface is not."""
        for o in occluders:
            t = o["type"]
            if t == "torus":
                if G.ray_hits_polyline(cam_pos, point, o["polyline"], o["tube_radius"], eps):
                    return o["name"]
            elif t == "sphere":
                if G.ray_hits_sphere(cam_pos, point, o["center_world"], o["radius"], eps):
                    return o["name"]
            elif t == "disc":
                if G.ray_hits_disc(cam_pos, point, o["center_world"], o["normal_world"], o["radius"], eps):
                    return o["name"]
            elif t == "cylinder":
                if G.ray_hits_capsule(cam_pos, point, o["a_world"], o["b_world"], o["radius"], eps):
                    return o["name"]
        return None

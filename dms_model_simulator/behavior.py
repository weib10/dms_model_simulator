"""Scenario -> per-frame driver state (head pose, gaze target point, eye/mouth openness, hands)."""
from __future__ import annotations
import bisect
import math
import warnings
from dataclasses import dataclass
from fractions import Fraction
import numpy as np
from . import geometry as G
from .config import deep_merge
from .world import Cabin


@dataclass
class DriverFrame:
    t: float
    head_yaw: float
    head_pitch: float
    head_roll: float
    gaze_point: np.ndarray        # vehicle frame
    target_name: str              # scripted target (segment target)
    openness_left: float
    openness_right: float
    blink: bool
    closure: bool
    hand: str
    phone: str
    mouth_openness: float = 0.0   # 0 closed .. 1 fully open
    yawning: bool = False
    cigarette: str = "none"       # none | mouth | hand
    seatbelt: str = "on"          # on | off
    held: str = "none"            # none | an object at the lips (configs/cabin.yaml held_objects), hand at_mouth


def on_off(v) -> str:
    """Seatbelt state from YAML: unquoted on/off are parsed as booleans by YAML 1.1."""
    if isinstance(v, bool):
        return "on" if v else "off"
    v = str(v).lower()
    if v not in ("on", "off"):
        raise ValueError(f"seatbelt must be on | off, got {v!r}")
    return v


class BehaviorGenerator:
    def __init__(self, cabin: Cabin, behavior_cfg: dict, scenario: dict, rng: np.random.Generator):
        self.cabin = cabin
        self.cfg = deep_merge(behavior_cfg, scenario.get("behavior"))
        self.rng = rng
        # independent child streams: enabling / retuning one event kind never shifts another kind's draws
        self.rng_blink, self.rng_closure, self.rng_yawn = rng.spawn(3)
        self.fps = float(self.cfg["fps"])
        fps_q = _exact(self.cfg["fps"])
        # Segment boundaries in exact arithmetic (no float drift): frame k is the instant k / fps and belongs to the
        # segment with start <= k / fps < end, i.e. frames [ceil(start * fps), ceil(end * fps)).
        self.segments = []
        cum = Fraction(0)
        for seg in scenario["segments"]:
            s = dict(seg)
            s["start"], s["start_frame"] = float(cum), math.ceil(cum * fps_q)
            cum += _exact(seg["duration"])
            s["end"], s["end_frame"] = float(cum), math.ceil(cum * fps_q)
            held = str(s.get("held", self.cfg["held"]))
            if held != "none":
                if held not in cabin.held_objects:
                    raise ValueError(f"held {held!r} is not none or one of {sorted(cabin.held_objects)} "
                                     "(configs/cabin.yaml held_objects)")
                if s.get("hand", self.cfg["hand"]) != "at_mouth":
                    raise ValueError(f"held {held!r} is an object at the lips: it needs hand: at_mouth")
            s["held"] = held
            self.segments.append(s)
        self.duration = float(cum)
        self.n_frames = math.ceil(cum * fps_q)          # every instant k / fps in [0, duration)
        self._end_frames = [s["end_frame"] for s in self.segments]

    # ---- helpers ------------------------------------------------------------
    def _seg_at_frame(self, i: int) -> int:
        """Segment index of frame i (integer comparison against the precomputed boundaries)."""
        return min(bisect.bisect_right(self._end_frames, i), len(self.segments) - 1)

    def _seg_at(self, t: float) -> int:
        """Segment index at time t; a frame instant k / fps is mapped through the exact frame boundaries."""
        k = t * self.fps
        if abs(k - round(k)) < 1e-6:
            return self._seg_at_frame(int(round(k)))
        return next((i for i, s in enumerate(self.segments) if t < s["end"]), len(self.segments) - 1)

    def _target_point(self, seg: dict) -> np.ndarray:
        return self.cabin.targets[seg["target"]]

    def _head_pose_for(self, seg: dict) -> np.ndarray:
        """Desired head pose (yaw, pitch, roll) for a segment."""
        if "head_pose" in seg:
            return np.array(seg["head_pose"], float)
        d = self._target_point(seg) - self.cabin.eye_point
        yaw, pitch = G.dir_to_yaw_pitch(d)
        rest = np.array(self.cfg["rest_head_pose"], float)
        hy = np.clip(self.cfg["head_gain_yaw"] * yaw, -self.cfg["head_yaw_limit_deg"], self.cfg["head_yaw_limit_deg"])
        hp = np.clip(self.cfg["head_gain_pitch"] * pitch, -self.cfg["head_pitch_limit_deg"], self.cfg["head_pitch_limit_deg"])
        return rest + np.array([hy, hp, 0.0])

    def _n_frames(self, duration_s: float) -> int:
        """Frames an event of duration_s lasts (at least one)."""
        return max(1, int(round(float(duration_s) * self.fps)))

    def _start_probs(self, events: list[tuple[float, int]]) -> list[float]:
        """Per-frame start probabilities of mutually exclusive events [(rate_per_min, n_frames)], drawn only in
        event-free frames, each conditional on the earlier ones not starting, such that the long-run rate of
        event i is rate_per_min. Renewal argument: idle-frame start probabilities p_i give event i at p_i /
        (1 - sum p + sum p_j n_j) per frame, so p_i = q_i / (1 - sum q_j (n_j - 1)) with q_i = rate / 60 / fps,
        the per-frame form of the hazard lambda / (60 (1 - lambda D / 60)) over event-free time."""
        q = [r / 60.0 / self.fps for r, _ in events]
        den = 1.0 - sum(qi * (n - 1) for qi, (_, n) in zip(q, events))
        if sum(q) <= 0:
            return [0.0] * len(events)
        if den <= 0 or sum(q) / den > 1.0:
            warnings.warn(f"event rates {[r for r, _ in events]}/min with {[n for _, n in events]}-frame events do not "
                          "fit (rate x duration >= 60 s per minute); events now follow back to back")
            p = [qi / sum(q) for qi in q]
        else:
            p = [qi / den for qi in q]
        out, left = [], 1.0
        for pi in p:
            out.append(min(1.0, pi / left) if left > 1e-12 else 0.0)
            left -= pi
        return out

    # ---- generation -----------------------------------------------------------
    def generate(self) -> list[DriverFrame]:
        dt = 1.0 / self.fps
        lim = self.cfg["eye_in_head_limit_deg"]
        lim_yaw, lim_pitch = float(lim["yaw"]), float(lim["pitch"])
        v_max = float(self.cfg["saccade_peak_speed_deg_s"])

        def eye_limited(R: np.ndarray, d: np.ndarray) -> np.ndarray:
            """d, or the closest direction (per axis) the eyes reach from head orientation R."""
            e_yaw, e_pitch = G.dir_to_yaw_pitch(R.T @ d)
            if abs(e_yaw) <= lim_yaw and abs(e_pitch) <= lim_pitch:
                return d
            return R @ G.yaw_pitch_to_dir(float(np.clip(e_yaw, -lim_yaw, lim_yaw)),
                                          float(np.clip(e_pitch, -lim_pitch, lim_pitch)))

        frames = []
        # events last a whole number of frames (round(duration_s * fps)); *_left = frames still to play
        blink_left = closure_left = yawn_left = 0
        blink_n = closure_n = yawn_n = 1
        yawn_peak = 0.0
        k_cur, hp_start = None, None
        hp = g_dir = g_dist = None
        g_dir0 = g_dist0 = None
        for i in range(self.n_frames):
            t = i * dt
            k = self._seg_at_frame(i)
            seg = self.segments[k]
            if k != k_cur:
                # transitions start from where gaze / head actually are (a short previous segment may have
                # been left before its own transition finished); the first segment starts on its target
                hp_start = self._head_pose_for(seg) if hp is None else hp
                g_dir0, g_dist0, t_gaze = g_dir, g_dist, None
                closure_cfg = deep_merge(self.cfg["closure"], seg.get("closure"))
                yawn_cfg = deep_merge(self.cfg["yawn"], seg.get("yawn"))
                n_blink = self._n_frames(self.cfg["blink"]["duration_s"])
                n_closure = self._n_frames(closure_cfg["duration_s"])
                p_blink, p_closure = self._start_probs([(float(self.cfg["blink"]["rate_per_min"]), n_blink),
                                                        (float(closure_cfg["rate_per_min"]), n_closure)])
                n_yawn = self._n_frames(yawn_cfg["duration_s"])
                (p_yawn,) = self._start_probs([(float(yawn_cfg["rate_per_min"]), n_yawn)])
                k_cur = k
            since = t - seg["start"]
            # head: slower move
            a_h = G.smoothstep(since / max(self.cfg["transition_head_s"], 1e-6))
            hp = (1 - a_h) * hp_start + a_h * self._head_pose_for(seg)
            # gaze: saccade of the gaze *direction* from the current eye origin (smoothstep in angle, slerp), the
            # distance along the ray interpolated alongside. The saccade aims at the target as far as the eyes
            # reach from the current head pose (eye_in_head_limit_deg): beyond the limit the gaze lags, held at the
            # limit, until the head has turned far enough; from then on it points exactly at the target.
            R_head, origin = self.cabin.head_pose(*hp)
            to_tgt = self._target_point(seg) - origin
            d1, r1 = eye_limited(R_head, G.normalize(to_tgt)), float(np.linalg.norm(to_tgt))
            if g_dir0 is None:
                g_dir0, g_dist0 = d1, r1
            if t_gaze is None:      # saccade duration: transition_gaze_s, longer when the peak speed would exceed v_max
                t_gaze = max(float(self.cfg["transition_gaze_s"]), 1.5 * G.angle_between(g_dir0, d1) / v_max, 1e-6)
            a_g = G.smoothstep(since / t_gaze)
            g_dir = eye_limited(R_head, G.slerp(g_dir0, d1, a_g))
            g_dist = (1 - a_g) * g_dist0 + a_g * r1
            gaze_pt = origin + g_dist * g_dir
            # eyes: one draw per frame from each event stream, used only when the eyes are free
            u_blink, u_closure, u_yawn = self.rng_blink.random(), self.rng_closure.random(), self.rng_yawn.random()
            base_open = float(seg.get("eye_openness", self.cfg["eye_openness"]))
            if blink_left <= 0 and closure_left <= 0:
                if u_blink < p_blink:
                    blink_n = blink_left = n_blink
                elif u_closure < p_closure:
                    closure_n = closure_left = n_closure
            openness, blink, closure = base_open, False, False
            if blink_left > 0:
                s_b = _bump(blink_n - blink_left, blink_n)                  # 0 -> 1 -> 0 over the blink
                openness = base_open * (1 - s_b) + self.cfg["blink"]["min_openness"] * s_b
                blink, blink_left = True, blink_left - 1
            elif closure_left > 0:
                openness, closure, closure_left = float(closure_cfg["openness"]), True, closure_left - 1
            # mouth / yawn (independent of the eye events)
            base_mouth = float(seg.get("mouth_openness", self.cfg["mouth_openness"]))
            if yawn_left <= 0 and u_yawn < p_yawn:
                yawn_n = yawn_left = n_yawn
                yawn_peak = float(yawn_cfg["peak_openness"])
            mouth, yawning = base_mouth, False
            if yawn_left > 0:
                s_y = _bump(yawn_n - yawn_left, yawn_n)
                mouth = base_mouth * (1 - s_y) + max(yawn_peak, base_mouth) * s_y
                yawning, yawn_left = True, yawn_left - 1
            frames.append(DriverFrame(
                t=t, head_yaw=float(hp[0]), head_pitch=float(hp[1]), head_roll=float(hp[2]),
                gaze_point=gaze_pt, target_name=seg["target"],
                openness_left=float(openness), openness_right=float(openness),
                blink=blink, closure=closure,
                hand=seg.get("hand", self.cfg["hand"]), phone=seg.get("phone", self.cfg["phone"]),
                mouth_openness=float(mouth), yawning=yawning,
                cigarette=str(seg.get("cigarette", self.cfg["cigarette"])),
                seatbelt=on_off(seg.get("seatbelt", self.cfg["seatbelt"])),
                held=seg["held"],
            ))
        return frames


def _exact(v) -> Fraction:
    """Exact rational of a YAML number (a float is taken at its shortest decimal text, 0.1 -> 1/10)."""
    return Fraction(str(v)) if isinstance(v, float) else Fraction(v)


def _bump(j: int, n: int) -> float:
    """Frame j of an n-frame event: sin profile sampled at frame centres, scaled so the middle frame (both
    middle frames for even n) reaches 1."""
    peak = np.sin(np.pi * ((n - 1) // 2 + 0.5) / n)
    return float(min(1.0, np.sin(np.pi * (j + 0.5) / n) / peak))

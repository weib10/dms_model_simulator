"""Turn ground truth into model-like outputs with condition-dependent errors."""
from __future__ import annotations
import copy
import math
import numpy as np
from . import geometry as G
from .camera import Camera
from .world import Cabin


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def _empty(**flags) -> dict:
    """Output of a frame with no model result (dropped, or latency pipeline still filling)."""
    return {**flags, "detections": [], "head_pose_camera": None, "gaze": None, "face_dir": None, "eye_openness": None,
            "mouth_openness": None}


# Time parameters of the emulator config, all given at `reference_fps` (see at_fps)
RHO_KEYS = ("temporal_rho", "fast_rho")                              # AR(1) frame-to-frame correlation
RUN_KEYS = ("mean_frames", "mean_miss_frames", "drop_mean_frames",   # mean run length in frames
            "outlier_mean_frames", "outlier_max_frames")
LATENCY_KEYS = ("latency_frames",)                                    # delay in frames


def at_fps(cfg: dict, fps: float) -> dict:
    """The emulator config for a run at `fps`: the time parameters are given at cfg["reference_fps"] and are
    converted so that the behaviour per second stays the same. Every key named in RHO_KEYS (rho -> rho ** (ref /
    fps)), RUN_KEYS (mean run N -> N * fps / ref, at least 1 frame; null stays null) and LATENCY_KEYS (L ->
    round(L * fps / ref), halves up) is converted, wherever it sits (per class, per rule). Shares and
    probabilities (drop_frame_prob, p_*, p, rate_per_frame, seatbelt_off_false_positive_rate) are fractions of
    frames or of time and stay as they are. At fps == reference_fps (or without reference_fps) the result equals
    cfg. Returns a new dict; cfg is not changed."""
    out = copy.deepcopy(cfg)
    ref = cfg.get("reference_fps")
    if ref is None or float(fps) == float(ref):
        return out
    k = float(fps) / float(ref)

    def walk(node):
        if isinstance(node, list):
            for x in node:
                walk(x)
        if not isinstance(node, dict):
            return
        for key, v in node.items():
            if isinstance(v, (dict, list)):
                walk(v)
            elif v is None or isinstance(v, bool):
                continue
            elif key in RHO_KEYS:
                node[key] = float(v) ** (1.0 / k)
            elif key in RUN_KEYS:
                node[key] = max(1.0, float(v) * k)
            elif key in LATENCY_KEYS:
                node[key] = int(math.floor(float(v) * k + 0.5))
    walk(out)
    return out


# Quantities with the structured pose / gaze error (Emulator._pose_error) and their number of axes
POSE_KEYS = {"head_pose": 3, "gaze": 2, "face_dir": 2}      # yaw, pitch(, roll)


def _per_axis(v, dim: int) -> np.ndarray:
    """A config value given for all axes (a number) or per axis (a list of dim numbers) as an array of dim."""
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.size == 1:
        return np.full(dim, float(a[0]))
    if a.size != dim:
        raise ValueError(f"expected one number or a list of {dim}, got {v!r}")
    return a


def _clip_box(box, w: int, h: int) -> list[float] | None:
    """Model boxes are in image pixels: clip a noisy box to the image; None if less than 1 px remains."""
    x1, x2 = sorted(float(np.clip(v, 0, w)) for v in (box[0], box[2]))
    y1, y2 = sorted(float(np.clip(v, 0, h)) for v in (box[1], box[3]))
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return [x1, y1, x2, y2]


OBSERVABLE_CHANNELS = ("face", "head_pose", "head_away", "gaze", "eyes", "mouth", "hand", "phone", "cigarette",
                       "seatbelt_region")       # head_away is added by Simulation.truths (it needs the road box)


def gaze_refusal(truth: dict, c: dict, face_found: bool, eye_found: dict) -> str | None:
    """Why the gaze model gives no direction this frame (None = it gives one). c: the emulator's `gaze` config;
    eye_found: {left_eye | right_eye: detected this frame}."""
    if not face_found:
        return "no_face"
    eyes_vis = [k for k, v in truth["eye_occluded"].items() if not v and eye_found.get(k, False)]
    if min(truth["eye_openness"]["left"], truth["eye_openness"]["right"]) < c["valid_min_openness"]:
        return "eyes_closed"
    if c["invalid_if_both_eyes_occluded"] and not eyes_vis:
        return "eyes_not_visible"
    if truth["gaze"]["gaze_to_camera_deg"] > c["max_gaze_to_camera_deg"]:
        return "gaze_too_far_from_camera"
    return None


def observable(truth: dict, cfg: dict) -> dict:
    """Per evidence channel (OBSERVABLE_CHANNELS): could a model that reports every box of the ground truth output
    it this frame? That is what `perfect_detection` (profile `exact`) outputs, so a frame that is not observable
    carries no evidence however good the detector is: the phone in the lap, the mouth behind a hand, the eyes of a
    head turned away. cfg: the emulator config (requires_face and the gaze validity rules).
    face / head_pose: the face box is in the truth; eyes / mouth / hand / phone / cigarette: that box is (eyes and
    mouth also need the face when the class has requires_face); gaze: the gaze model gives an eye gaze direction
    (gaze_refusal; pipeline gaze.eye.valid; the face direction gaze.face follows face / head_pose); seatbelt_region: the place where a worn belt is boxed is in the image, worn or not.
    Simulation.truths adds head_away: the head pose is there and the head itself points off the road (outside
    the road box), so the head pose alone shows the eyes are off the road; a glance with the eyes only shows in
    the gaze and nowhere else."""
    present = {d["class"] for d in truth["detections"]}
    face = "face" in present
    with_face = lambda key: face or not cfg["detection"][key].get("requires_face")
    eyes = {c: c in present and with_face("eye") for c in ("left_eye", "right_eye")}
    return {"face": face, "head_pose": face, "gaze": gaze_refusal(truth, cfg["gaze"], face, eyes) is None,
            "eyes": any(eyes.values()), "mouth": "mouth" in present and with_face("mouth"),
            "hand": "hand" in present, "phone": "phone" in present, "cigarette": "cigarette" in present,
            "seatbelt_region": truth.get("seatbelt_strap_bbox") is not None}


def markov_step(state: bool | None, r: float, mean_run: float, u: float) -> bool:
    """One frame of a two-state chain for a "rare" state S (a missed detection, a flipped class, a false positive)
    with stationary probability r and mean run length mean_run frames (>= 1). state: S last frame, None = no
    history (first frame, or reset): S with probability r. u ~ U(0, 1), one draw per frame.
    Exit S with b = 1 / mean_run, enter with a = b r / (1 - r), so P(S) = a / (a + b) = r. If a would exceed 1
    (r > mean_run / (mean_run + 1)) a = 1 and b = (1 - r) / r instead: still P(S) = r, with longer S runs.
    r may change from frame to frame (it follows the frame's conditions); the chain then tracks it."""
    r = min(max(float(r), 0.0), 1.0)
    if state is None:
        return u < r
    if r <= 0.0:
        return False
    if r >= 1.0:
        return True
    b = 1.0 / max(float(mean_run), 1.0)
    a = b * r / (1.0 - r)
    if a > 1.0:
        a, b = 1.0, (1.0 - r) / r
    return (u >= b) if state else (u < a)


def capped_markov_step(n: int | None, r: float, mean_run: float, max_run: float | None, u: float) -> int:
    """markov_step for a rare state S whose runs are also cut at max_run frames (None = no cap). n: length of the
    current S run (0 = not in S), None = no history (S with probability r). Returns the new run length (0 = not in
    S). u ~ U(0, 1), one draw per frame.
    In S: leave after max_run frames (rounded, at least 1), else with b = 1 / mean_run. A run then lasts
    L = (1 - (1 - b) ** max_run) / b frames on average (1 / b without a cap). Out of S: enter with
    a = r / ((1 - r) L), so the share of S frames stays r (a is capped at 1; the share is then below r)."""
    r = min(max(float(r), 0.0), 1.0)
    if n is None:
        return 1 if u < r else 0
    if r <= 0.0:
        return 0
    if r >= 1.0:
        return n + 1
    b = 1.0 / max(float(mean_run), 1.0)
    cap = None if max_run is None else max(1, int(math.floor(float(max_run) + 0.5)))
    if n > 0:
        if cap is not None and n >= cap:
            return 0
        return n + 1 if u >= b else 0
    mean_len = 1.0 / b if cap is None else (1.0 - (1.0 - b) ** cap) / b
    a = min(1.0, r / ((1.0 - r) * mean_len))
    return 1 if u < a else 0


class AR1:
    """Temporally correlated unit-variance noise, scaled per call by the current std.

    Stationary from the first sample (x0 ~ N(0, 1)). step(frame) advances to frame index `frame`: after a gap of
    D frames (quantity not produced: no face, gaze invalid, dropped frame) the correlation with the last sample
    is rho ** D, i.e. x = rho**D * x + sqrt(1 - rho**(2D)) * n. Calling it again for the same frame returns the
    same sample. frame=None advances by one frame."""
    def __init__(self, rng, rho, dim):
        self.rng, self.rho, self.dim = rng, float(rho), int(dim)
        self.x, self.frame = None, None

    def step(self, frame: int | None = None):
        if frame is None:
            frame = 0 if self.frame is None else self.frame + 1
        if self.x is None:
            self.x = self.rng.standard_normal(self.dim)
        elif frame > self.frame:
            a = self.rho ** (frame - self.frame)
            self.x = a * self.x + np.sqrt(1 - a * a) * self.rng.standard_normal(self.dim)
        self.frame = frame
        return self.x


class Emulator:
    """Per-frame model output from truth. Call emulate() once per frame, in order.

    Random streams: `rng` is split (Generator.spawn) into independent child streams, one each for frame drops,
    false positives, seatbelt-off false positives, the detection draws of every class, and every AR(1) noise
    process. Changing e.g. the false-positive rate or one class's detection curve therefore leaves the draws of
    the other streams unchanged. Remaining couplings: within one class the detection draw and its score noise
    share a stream; a drop or a missed face changes which quantities are produced, hence when their noise
    streams advance. The eye / mouth class-flip chains and the contextual false positives have streams of their
    own, spawned after all the others whether or not they are enabled, so enabling them changes nothing else.

    Optional error modes (all off in the shipped configs/emulator.yaml):
    burst misses (`detection.<class>.mean_miss_frames`) and burst frame drops (`pipeline.drop_mean_frames`),
    persistent eye / mouth class flips (`eye_state_flip`, `mouth_state_flip`), context-dependent false
    positives (`contextual_false_positives`) and box jumps (`box_jump`). Each is a two-state Markov chain
    (markov_step). `box_conditions` scales the box noise with the view (occlusion, head yaw, small boxes); its gains
    are 0 in the shipped config. The box-jump stream is spawned after those, so enabling it changes nothing else.
    The face-direction noise (`face_dir`) stream came next: adding it left every earlier draw unchanged.
    The structured pose / gaze error (`_pose_error`: per-run bias, fast jitter, outlier runs of head_pose, gaze and
    face_dir, plus the per-run eye-openness bias) has the last streams: one for all per-run biases (drawn once,
    here, in a fixed order and size whatever the config), then one fast AR(1) and one outlier stream per quantity.
    Their gains default to 0, so the default output is what it was before they existed."""

    def __init__(self, cfg: dict, cabin: Cabin, cam: Camera, rng: np.random.Generator):
        self.cfg, self.cabin, self.cam, self.rng = cfg, cabin, cam, rng
        d = cfg["detection"]
        classes = [c for c in d if c not in ("false_positives", "eye")] + ["left_eye", "right_eye"]
        spec = {c: d["eye" if c.endswith("_eye") else c] for c in classes}
        streams = iter(rng.spawn(3 + 2 * len(classes) + 4 + 3 + 1 + 1 + 1 + 2 * len(POSE_KEYS)))
        self.drop_rng, self.fp_rng, self.sb_fp_rng = next(streams), next(streams), next(streams)
        self.det_rng = {c: next(streams) for c in classes}
        self.box_noise = {c: AR1(next(streams), spec[c]["temporal_rho"], 4) for c in classes}
        self.hp_noise = AR1(next(streams), cfg["head_pose"]["temporal_rho"], 3)
        self.gaze_noise = AR1(next(streams), cfg["gaze"]["temporal_rho"], 2)
        self.open_noise = AR1(next(streams), cfg["eye_openness"]["temporal_rho"], 2)
        self.mouth_noise = AR1(next(streams), cfg["mouth_openness"]["temporal_rho"], 1)
        self.eye_flip_rng, self.mouth_flip_rng, self.ctx_fp_rng = next(streams), next(streams), next(streams)
        self.jump_rng = next(streams)
        self.face_dir_noise = AR1(next(streams), cfg["face_dir"]["temporal_rho"], 2)
        bias_rng = next(streams)                     # per-run biases: unit normals in a fixed order and size
        z = {key: bias_rng.standard_normal(dim) for key, dim in POSE_KEYS.items()}
        z_open = float(bias_rng.standard_normal())
        self.run_bias = {key: z[key] * _per_axis(cfg[key].get("bias_std_deg", 0.0), dim)
                         for key, dim in POSE_KEYS.items()}
        self.open_bias = z_open * float(cfg["eye_openness"].get("bias_std", 0.0))
        self.fast_noise = {key: AR1(next(streams), cfg[key].get("fast_rho", 0.0), dim) for key, dim in POSE_KEYS.items()}
        self.outlier_rng = {key: next(streams) for key in POSE_KEYS}
        self.outlier_state: dict[str, int | None] = {key: None for key in POSE_KEYS}   # outlier run length so far
        self.outlier_offset: dict[str, np.ndarray] = {}
        self.miss_state: dict[str, bool] = {}        # class -> missed last time it was evaluated (burst misses)
        self.jump_state: dict[str, bool | None] = {} # class -> box jumped last time it was produced (box_jump)
        self.jump_offset: dict[str, tuple] = {}      # class -> (dx, dy as box fractions, size factor) of the jump
        self.face_shift = None                       # pixel shift of this frame's jumped face box (features follow)
        self.drop_state: bool | None = None          # last frame dropped (burst drops)
        self.flip_state = {"eye": None, "mouth": None}        # reported class wrong last frame (None = no history)
        self.flip_truth_cls = {"eye": None, "mouth": None}    # truth class the flip state refers to
        self.ctx_state: dict[str, bool | None] = {}  # contextual false-positive rule -> active last frame
        self.frame = -1                  # index of the current frame (advanced by emulate / _infer)
        self.queue = []

    # ---- detections -----------------------------------------------------------
    def _p_detect(self, spec: dict, width_px: float, rel_yaw: float | None, occ: float) -> float:
        if self.cfg.get("perfect_detection"):        # logic tests: every truth box is reported
            return 1.0
        p = spec["p_max"] * sigmoid((width_px - spec["min_width_px"]) / spec["width_scale_px"])
        if rel_yaw is not None and "yaw_limit_deg" in spec:
            p *= sigmoid((spec["yaw_limit_deg"] - abs(rel_yaw)) / spec["yaw_scale_deg"])
        p *= (1 - occ) ** spec.get("occlusion_power", 1.0)
        return float(p)

    def _detect(self, det: dict, rel_yaw: float, face_found: bool) -> dict | None:
        cls = det["class"]
        key = "eye" if cls.endswith("_eye") else cls
        spec = self.cfg["detection"][key]
        if spec.get("requires_face") and not face_found:
            return None
        p = self._p_detect(spec, det["width_px"], rel_yaw if key in ("face", "eye", "mouth") else None, det["occlusion"])
        rng = self.det_rng[cls]
        L = spec.get("mean_miss_frames")
        if L is None:                                # independent draw per frame
            if rng.random() > p:
                return None
        else:                                        # burst misses: P(miss) = 1 - p, mean miss run L frames
            missed = markov_step(self.miss_state.get(cls), 1.0 - p, L, rng.random())
            self.miss_state[cls] = missed            # kept while eyes / mouth wait for a missed face; reset
                                                     # when the class leaves the truth (see _infer)
            if missed:
                return None
        x1, y1, x2, y2 = det["bbox"]
        w, h = x2 - x1, y2 - y1
        hard = self._hardness(self.cfg.get("box_conditions") or {}, spec, det, rel_yaw, key)
        n = self.box_noise[cls].step(self.frame) * (spec["bbox_noise_frac"] * hard) * np.array([w, h, w, h])
        raw, jumped = self._jump(cls, key, spec, det, rel_yaw, np.array(det["bbox"]) + n)
        box = _clip_box(raw, self.cam.width, self.cam.height)
        conf = spec["conf_base"] * (0.5 + 0.5 * p) + rng.normal(0, spec["conf_noise_std"])
        if box is None:
            return None
        out = {"class": cls, "bbox": box, "conf": float(np.clip(conf, 0.0, 1.0)), "p_detect": p}
        if jumped:
            out["jump"] = True
        return out

    @staticmethod
    def _hardness(g: dict, spec: dict, det: dict, rel_yaw: float | None, key: str) -> float:
        """1 + occlusion * occluded_fraction + rel_yaw * min(|yaw rel. camera| / 45 deg, 2) (face / eye / mouth)
        + small * min(min_width_px / width_px, 4): how much harder this view is than an easy one (gains 0 -> 1)."""
        x = float(g.get("occlusion", 0.0)) * float(det["occlusion"])
        if key in ("face", "eye", "mouth") and rel_yaw is not None:
            x += float(g.get("rel_yaw", 0.0)) * min(abs(float(rel_yaw)) / 45.0, 2.0)
        if g.get("small"):
            x += float(g["small"]) * min(float(spec["min_width_px"]) / max(float(det["width_px"]), 1.0), 4.0)
        return 1.0 + x

    def _jump(self, cls: str, key: str, spec: dict, det: dict, rel_yaw: float | None, box: np.ndarray):
        """box_jump: now and then a class's box lands somewhere else (centre shifted by N(0, shift_frac) of its
        size, size scaled by U(scale_range)) for a run of mean_frames frames, then comes back. One chain per class
        (markov_step) with share p, times the box_conditions-style `gain` of the view; the offset is drawn when a
        run starts and kept for it. features_follow_face: eyes / mouth move with a jumped face (a whole-face
        miscrop) unless they jump themselves. Returns (box, jumped)."""
        c = self.cfg.get("box_jump") or {}
        if not c.get("enabled") or key not in c.get("classes", []):
            if key in ("eye", "mouth") and self.face_shift is not None and c.get("features_follow_face"):
                return box + np.tile(self.face_shift, 2), False
            return box, False
        p = float(c["p"]) * self._hardness(c.get("gain") or {}, spec, det, rel_yaw, key)
        prev = self.jump_state.get(cls)
        on = self.jump_state[cls] = markov_step(prev, p, c["mean_frames"], self.jump_rng.random())
        if on and not prev:
            dx, dy = self.jump_rng.normal(0.0, float(c["shift_frac"]), 2)
            self.jump_offset[cls] = (float(dx), float(dy), float(self.jump_rng.uniform(*c["scale_range"])))
        if not on:
            if key in ("eye", "mouth") and self.face_shift is not None and c.get("features_follow_face"):
                return box + np.tile(self.face_shift, 2), False
            return box, False
        dx, dy, s = self.jump_offset[cls]
        w, h = box[2] - box[0], box[3] - box[1]
        cx, cy = 0.5 * (box[0] + box[2]) + dx * w, 0.5 * (box[1] + box[3]) + dy * h
        out = np.array([cx - s * w / 2, cy - s * h / 2, cx + s * w / 2, cy + s * h / 2])
        if key == "face":
            self.face_shift = np.array([dx * w, dy * h])
        return out, True

    def _false_positives(self) -> list[dict]:
        """Spurious boxes: Poisson count per frame with mean rate_per_frame (any rate >= 0)."""
        fp = self.cfg["detection"]["false_positives"]
        rng = self.fp_rng
        out = []
        for _ in range(int(rng.poisson(float(fp["rate_per_frame"])))):
            s = rng.uniform(*fp["size_px_range"])
            x, y = rng.uniform(0, self.cam.width - s), rng.uniform(0, self.cam.height - s)
            out.append({"class": str(rng.choice(fp["classes"])), "bbox": [x, y, x + s, y + s],
                        "conf": float(rng.uniform(*fp["conf_range"])), "false_positive": True})
        return out

    def _seatbelt_off_false_positive(self, truth: dict) -> dict | None:
        """Belt off: the model occasionally still reports a strap where it would be (e.g. clothing seams)."""
        spec = self.cfg["detection"]["seatbelt"]
        box = truth.get("seatbelt_strap_bbox")
        if truth.get("seatbelt") != "off" or not box:
            return None
        if self.sb_fp_rng.random() >= spec["seatbelt_off_false_positive_rate"]:
            return None
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        n = self.box_noise["seatbelt"].step(self.frame) * spec["bbox_noise_frac"] * np.array([w, h, w, h])
        conf = float(self.sb_fp_rng.uniform(*spec["seatbelt_off_false_positive_conf_range"]))
        box = _clip_box(np.array(box) + n, self.cam.width, self.cam.height)
        return None if box is None else {"class": "seatbelt", "bbox": box, "conf": conf, "false_positive": True}

    def _contextual_false_positives(self, truth: dict) -> list[dict]:
        """Boxes the model reports because of what the hand is doing (hand at the ear -> phone, hand at the mouth ->
        cigarette): per rule a Markov chain that runs only while the truth hand state is in `when_hand` and the hand
        box is in the image (reset otherwise), active a share `p` of those frames in runs of mean `mean_frames`. The
        box is centred on the truth hand box, scaled by `box_scale` [w, h], with `jitter_frac` uniform jitter.
        Optional `when_held`: the rule also needs the truth held object (configs/cabin.yaml held_objects) in it.
        `anchor: held` centres the box on the held object's box instead of the hand box (an object in the mouth
        without a hand); `when_hand` is then optional (missing = any hand state)."""
        c = self.cfg.get("contextual_false_positives") or {}
        if not c.get("enabled"):
            return []
        rng = self.ctx_fp_rng
        hand = next((d["bbox"] for d in truth["detections"] if d["class"] == "hand"), None)
        out = []
        for rule in c.get("rules") or []:
            name = rule["name"]
            when, held = rule.get("when_hand"), rule.get("when_held")
            if when is None and rule.get("anchor", "hand") != "held":
                raise ValueError(f"contextual_false_positives rule {name!r}: when_hand is required unless anchor: held")
            base = truth.get("held_bbox") if rule.get("anchor", "hand") == "held" else hand
            if base is None or (when is not None and truth.get("hand") not in ([when] if isinstance(when, str) else when)) or (
                    held is not None and truth.get("held", "none") not in ([held] if isinstance(held, str) else held)):
                self.ctx_state[name] = None
                continue
            on = self.ctx_state[name] = markov_step(self.ctx_state.get(name), rule["p"], rule["mean_frames"], rng.random())
            if not on:
                continue
            x1, y1, x2, y2 = base
            sw, sh = rule.get("box_scale", [1.0, 1.0])
            w, h = (x2 - x1) * sw, (y2 - y1) * sh
            j = float(rule.get("jitter_frac", 0.05))
            cx = 0.5 * (x1 + x2) + rng.uniform(-j, j) * w
            cy = 0.5 * (y1 + y2) + rng.uniform(-j, j) * h
            conf = float(rng.uniform(*rule["conf_range"]))       # drawn before clipping: stream stays aligned
            box = _clip_box([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], self.cam.width, self.cam.height)
            if box is not None:
                out.append({"class": rule["class"], "bbox": box, "conf": conf, "false_positive": True, "context": name})
        return out

    # ---- eye / mouth class flips ---------------------------------------------------
    def _flip_prob(self, d: dict, truth: dict, size_px: float | None, pitch_key: str) -> float:
        """Stationary probability that the reported class is wrong this frame (one direction's terms, summed)."""
        q = float(d.get("p_base", 0.0))
        if d.get("p_gaze_down"):
            pitch = float(truth["gaze"][pitch_key])
            q += d["p_gaze_down"] * sigmoid((d["gaze_down_pitch_deg"] - pitch) / d["gaze_down_scale_deg"])
        if d.get("p_small") and size_px is not None:
            q += d["p_small"] * sigmoid((d["small_width_px"] - size_px) / d["small_scale_px"])
        q += float((d.get("p_hand_state") or {}).get(truth.get("hand"), 0.0))
        return float(np.clip(q, 0.0, 1.0))

    def _flip(self, key: str, truth_value: float, truth: dict, size_px: float | None) -> float | None:
        """Openness to report instead of the noisy one when the model gets the open / closed class wrong (None =
        not flipped / disabled). One chain per face for both eyes and one for the mouth: the state is "reported
        class wrong"; it is reset to correct whenever the truth class changes (so a latched flip never hides a
        blink's class change) and dropped when the eyes / mouth leave the truth (see _infer), then entered with
        the frame's flip probability and left after mean_frames frames on average (markov_step). The class boundary
        `threshold` must match the layout's class_map threshold. A flipped frame is marked in the output
        (`class_flip`: ["eye"] / ["mouth"])."""
        fc = self.cfg.get(f"{key}_state_flip") or {}
        if not fc.get("enabled"):
            return None
        low = float(truth_value) < float(fc["threshold"])          # eye: closed; mouth: closed
        if self.flip_truth_cls[key] is not None and self.flip_truth_cls[key] != low:
            self.flip_state[key] = False
        self.flip_truth_cls[key] = low
        d = fc.get("closed_to_open" if low else "open_to_closed") or {}
        q = self._flip_prob(d, truth, size_px, fc.get("gaze_pitch_key", "eye_in_head_pitch"))
        rng = self.eye_flip_rng if key == "eye" else self.mouth_flip_rng
        wrong = self.flip_state[key] = markov_step(self.flip_state[key], q, fc["mean_frames"], rng.random())
        if not wrong:
            return None
        return float(fc["open_value"] if low else fc["closed_value"])

    # ---- structured pose / gaze error ------------------------------------------------
    def _pose_error(self, key: str, std: float) -> tuple[np.ndarray, float, bool]:
        """Error of head_pose / gaze / face_dir this frame beyond the slow AR(1): per-run bias + fast jitter +
        outlier offset, and the std to report. key: a POSE_KEYS quantity; std: the slow part's std this frame.
        - per-run bias: N(0, bias_std_deg) per axis, drawn once per Emulator (one run = one person): no
          smoothing removes it;
        - fast jitter: a second AR(1) (fast_std_deg, fast_rho; rho given at reference_fps);
        - outliers: a two-state chain (capped_markov_step) in the outlier state on a share outlier_p of the frames
          the quantity is produced, in runs that end with probability 1 / outlier_mean_frames per frame and at the
          latest after outlier_max_frames frames; each run adds one offset of magnitude
          U(outlier_min_deg, outlier_max_deg) in a random direction (drawn when the run starts, kept for it).
        All gains default to 0. Reported std: the slow std unchanged while fast_std_deg and bias_std_deg are 0,
        else sqrt(std^2 + fast_std_deg^2 + mean over axes of bias_std_deg^2), so a profile that moves error from
        the slow part into the bias / jitter does not report a higher confidence. Outlier frames do not raise the
        reported std (the model does not know). Returns (error, std, outlier)."""
        c, dim = self.cfg[key], POSE_KEYS[key]
        fast = float(c.get("fast_std_deg", 0.0))
        err = self.fast_noise[key].step(self.frame) * fast + self.run_bias[key]
        rng = self.outlier_rng[key]
        prev = self.outlier_state[key]
        n_run = self.outlier_state[key] = capped_markov_step(
            prev, float(c.get("outlier_p", 0.0)), float(c.get("outlier_mean_frames", 1.0)),
            c.get("outlier_max_frames"), rng.random())
        on = n_run > 0
        if on:
            if not prev:
                d = rng.standard_normal(dim)
                d /= max(float(np.linalg.norm(d)), 1e-12)
                self.outlier_offset[key] = d * rng.uniform(float(c["outlier_min_deg"]), float(c["outlier_max_deg"]))
            err = err + self.outlier_offset[key]
        bias_var = float(np.mean(_per_axis(c.get("bias_std_deg", 0.0), dim) ** 2))
        if fast > 0.0 or bias_var > 0.0:
            std = math.sqrt(std * std + fast * fast + bias_var)
        return err, std, on

    # ---- directions (eye gaze, face direction) ------------------------------------
    def _direction(self, yaw_c: float, pitch_c: float, origin, truth_dir_vehicle, std: float, c: dict) -> dict:
        """Output of a noisy direction given as camera face-frame angles: vector in camera and vehicle frame,
        vehicle angles, zone from `origin`, angular error to the truth direction, conf from the predicted std
        (c: conf_base, conf_per_deg_std)."""
        dir_cam = G.face_to_cv(G.yaw_pitch_to_dir(yaw_c, pitch_c))
        dir_v = self.cam.dir_to_vehicle(dir_cam)
        yaw_v, pitch_v = G.dir_to_yaw_pitch(dir_v)
        zone, zerr = self.cabin.classify_zone(np.array(origin), dir_v)
        conf = float(np.clip(c["conf_base"] - c["conf_per_deg_std"] * std, 0, 1))
        return {"valid": True, "yaw_camera": float(yaw_c), "pitch_camera": float(pitch_c),
                "dir_camera": dir_cam.tolist(), "yaw_vehicle": float(yaw_v), "pitch_vehicle": float(pitch_v),
                "dir_vehicle": dir_v.tolist(), "zone": zone, "zone_angle_err_deg": float(zerr),
                "angular_error_deg": G.angle_between(dir_v, np.array(truth_dir_vehicle)),
                "std_deg": float(std), "conf": conf}

    # ---- per-frame --------------------------------------------------------------
    def emulate(self, truth: dict) -> dict:
        pipe = self.cfg["pipeline"]
        if pipe.get("drop_mean_frames") is None:
            dropped = self.drop_rng.random() < pipe["drop_frame_prob"]
        else:                                        # burst drops: P(drop) = drop_frame_prob, mean run drop_mean_frames
            dropped = self.drop_state = markov_step(self.drop_state, pipe["drop_frame_prob"], pipe["drop_mean_frames"],
                                                    self.drop_rng.random())
        if dropped:
            self.frame += 1
            out = _empty(dropped=True)
        else:
            out = self._infer(truth)
        return self._delay(out)

    def _delay(self, out: dict) -> dict:
        """latency: report the output of frame i at frame i + latency_frames. Dropped frames go through the
        queue too, so a drop never changes the latency. With latency > 0 every output carries `source_frame`
        (the frame whose truth it was computed from; None for the initial latency fill) and its diagnostics
        (e.g. gaze angular_error_deg) refer to that frame, not to the truth record it is written next to."""
        lat = int(self.cfg["pipeline"]["latency_frames"])
        if lat <= 0:
            return out
        out["source_frame"] = self.frame
        self.queue.append(out)
        if len(self.queue) <= lat:
            return _empty(dropped=True, latency_fill=True, source_frame=None)
        return self.queue.pop(0)

    def _infer(self, truth: dict) -> dict:
        """Model output for the next frame (advances the frame index)."""
        self.frame += 1
        k = self.frame
        self.face_shift = None
        rel_yaw = truth["head_pose_camera"]["yaw"]
        face_truth = next((d for d in truth["detections"] if d["class"] == "face"), None)
        dets = []
        face_pred = self._detect(face_truth, rel_yaw, True) if face_truth else None
        if face_pred:
            dets.append(face_pred)
        eye_found, mouth_found = {}, False
        for d in truth["detections"]:
            if d["class"] == "face":
                continue
            p = self._detect(d, rel_yaw, face_pred is not None)
            if p:
                dets.append(p)
            if d["class"].endswith("_eye"):
                eye_found[d["class"]] = p is not None
            elif d["class"] == "mouth":
                mouth_found = p is not None
        present = {d["class"] for d in truth["detections"]}
        for c in [c for c in self.miss_state if c not in present]:
            del self.miss_state[c]                   # e.g. phone put away: the next episode starts afresh
        for c in [c for c in self.jump_state if c not in present]:
            del self.jump_state[c]
        for key, gone in (("eye", not any(c.endswith("_eye") for c in present)), ("mouth", "mouth" not in present)):
            if gone:                                 # out of view (e.g. a long head turn): a latched class flip
                self.flip_state[key] = self.flip_truth_cls[key] = None      # does not come back with it
        sb = self._seatbelt_off_false_positive(truth)
        if sb:
            dets.append(sb)
        dets += self._false_positives()
        dets += self._contextual_false_positives(truth)

        # head pose
        hp_out = None
        if face_pred:
            c = self.cfg["head_pose"]
            fw = max(face_truth["width_px"], 1.0)
            std = c["base_std_deg"] + c["per_deg_rel_yaw"] * abs(rel_yaw) + c["per_px_inv_face_width"] / fw
            tp = truth["head_pose_camera"]
            n = self.hp_noise.step(k) * std + np.array(c["bias_deg"])
            e, std, outlier = self._pose_error("head_pose", std)
            n = n + e
            hp_out = {"yaw": tp["yaw"] + n[0], "pitch": tp["pitch"] + n[1], "roll": tp["roll"] + n[2], "std_deg": float(std),
                      "conf": float(np.clip(c["conf_base"] - c["conf_per_deg_std"] * std, 0, 1))}
            if outlier:
                hp_out["outlier"] = True

        # eye openness: per eye, only for an eye the model detected this frame (else null); null if neither was
        eo = None
        flipped = []
        if face_pred and any(eye_found.get(f"{s}_eye") for s in ("left", "right")):
            c = self.cfg["eye_openness"]
            n = self.open_noise.step(k) * c["noise_std"]
            if self.open_bias:                       # per-run open level (bias_std; 0 = off)
                n = n + self.open_bias
            eo = {s: float(np.clip(truth["eye_openness"][s] + n[i], 0, 1)) if eye_found.get(f"{s}_eye") else None
                  for i, s in enumerate(("left", "right"))}
            t_eo = truth["eye_openness"]
            eye_w = max([d["width_px"] for d in truth["detections"] if d["class"].endswith("_eye")] or [0.0])
            v = self._flip("eye", 0.5 * (t_eo["left"] + t_eo["right"]), truth, eye_w or None)
            if v is not None:                        # both detected eyes reported in the wrong class
                eo = {s: (v if x is not None else None) for s, x in eo.items()}
                flipped.append("eye")

        # mouth openness: only when the mouth was detected this frame
        mo = None
        if face_pred and mouth_found:
            c = self.cfg["mouth_openness"]
            n = self.mouth_noise.step(k)[0] * c["noise_std"]
            mo = float(np.clip(truth["mouth_openness"] + n, 0, 1))
            mouth_w = next((d["width_px"] for d in truth["detections"] if d["class"] == "mouth"), None)
            v = self._flip("mouth", truth["mouth_openness"], truth, mouth_w)
            if v is not None:
                mo = v
                flipped.append("mouth")

        # gaze
        c = self.cfg["gaze"]
        reason = gaze_refusal(truth, c, face_pred is not None, eye_found)
        gz = {"valid": False, "reason": reason}
        if reason is None:
            tg = truth["gaze"]
            eye_w = max([d["width_px"] for d in truth["detections"] if d["class"].endswith("_eye")] or [1.0])
            std = (c["base_std_deg"] + c["per_deg_gaze_to_camera"] * tg["gaze_to_camera_deg"]
                   + c["per_deg_rel_head_yaw"] * abs(rel_yaw) + c["per_px_inv_eye_width"] / max(eye_w, 1.0))
            n = self.gaze_noise.step(k) * std + np.array(c["bias_deg"])
            e, std, outlier = self._pose_error("gaze", std)
            n = n + e
            gz = self._direction(tg["yaw_camera"] + n[0], tg["pitch_camera"] + n[1], tg["origin_vehicle"],
                                 tg["dir_vehicle"], std, c)
            if outlier:
                gz["outlier"] = True

        # face direction (where the head points): whenever the face is found, also when the eye gaze is refused
        fd = None
        if face_pred:
            c = self.cfg["face_dir"]
            tf = truth["face_dir"]
            fw = max(face_truth["width_px"], 1.0)
            std = c["base_std_deg"] + c["per_deg_rel_yaw"] * abs(rel_yaw) + c["per_px_inv_face_width"] / fw
            n = self.face_dir_noise.step(k) * std + np.array(c["bias_deg"])
            e, std, outlier = self._pose_error("face_dir", std)
            n = n + e
            fd = self._direction(tf["yaw_camera"] + n[0], tf["pitch_camera"] + n[1], tf["origin_vehicle"],
                                 tf["dir_vehicle"], std, c)
            if outlier:
                fd["outlier"] = True

        out = {"dropped": False, "detections": dets, "head_pose_camera": hp_out, "gaze": gz, "face_dir": fd,
               "eye_openness": eo, "mouth_openness": mo}
        if flipped:                                  # open / closed class reported wrong this frame (diagnostic)
            out["class_flip"] = flipped
        return out

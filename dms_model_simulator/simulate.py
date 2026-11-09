"""Run a scenario through one camera: behaviour -> truth -> emulated model output."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .behavior import BehaviorGenerator
from .camera import Camera
from .config import load_configs, load_scenario
from .emulator import Emulator
from .world import Cabin

FORMATS = ("full", "pipeline", "both", "dmsbin", "all")   # all = full + pipeline + dmsbin


def rng_streams(seed: int, emulator_seed: int | None = None) -> tuple[np.random.Generator, np.random.Generator]:
    """(behaviour rng, emulator rng). Both are children of SeedSequence(seed), so they are independent: changing
    how many draws the behaviour makes (e.g. enabling yawns) does not change the model noise. emulator_seed
    replaces the emulator stream by the emulator child of SeedSequence(emulator_seed) (emulator_seed == seed is
    the default), so one behaviour can be run with different model noise."""
    beh_ss, emu_ss = np.random.SeedSequence(int(seed)).spawn(2)
    if emulator_seed is not None:
        emu_ss = np.random.SeedSequence(int(emulator_seed)).spawn(2)[1]
    return np.random.default_rng(beh_ss), np.random.default_rng(emu_ss)


class Simulation:
    def __init__(self, scenario, camera_name: str, seed: int = 0, cfgs: dict | None = None, emulator_overrides: dict | None = None,
                 emulator_seed: int | None = None, fps: float | None = None, variant: int | dict | None = None):
        """fps: frame rate override (beats configs/behavior.yaml and the scenario's behavior.fps); None = config.
        The scenario's optional `cabin:` block is merged into configs/cabin.yaml and its `camera:` block into the
        simulated camera's preset (e.g. driver.eye_point_offset, driver.head_scale, mount_offset,
        mount_rotation_deg). variant: driver / camera-mount variant drawn from configs/variation.yaml (an int N, see
        variation.sample_variant, or an already sampled dict), applied on top of those; its number is appended to
        the scenario name ("<name>_v<N>") so output files of different variants do not collide."""
        from .config import deep_merge
        self.cfgs = dict(cfgs) if cfgs else load_configs()       # shallow copy: overrides below do not leak
        if emulator_overrides:
            self.cfgs["emulator"] = deep_merge(self.cfgs["emulator"], emulator_overrides)
        self.scenario = load_scenario(scenario) if isinstance(scenario, (str, Path)) else scenario
        if fps is not None:
            if not float(fps) > 0:
                raise ValueError(f"fps must be > 0, got {fps!r}")
            self.scenario = deep_merge(self.scenario, {"behavior": {"fps": float(fps)}})
        if isinstance(variant, int) and not isinstance(variant, bool):
            from .variation import sample_variant
            variant = sample_variant(variant)
        if variant and "variant" not in variant:
            raise ValueError("variant dict needs a 'variant' number (use variation.sample_variant)")
        self.variant = variant
        cab_ov, cam_ov = [self.scenario.get("cabin")], [self.scenario.get("camera")]
        if variant:
            cab_ov.append(variant.get("cabin")); cam_ov.append(variant.get("camera"))
            self.scenario = {**self.scenario, "name": f"{self.scenario['name']}_v{variant['variant']}"}
        cams = self.cfgs["cameras"]
        if camera_name not in cams["cameras"]:
            raise KeyError(f"unknown camera {camera_name!r} (configs/cameras.yaml: {sorted(cams['cameras'])})")
        for ov in cab_ov:
            self.cfgs["cabin"] = deep_merge(self.cfgs["cabin"], ov)
        for ov in cam_ov:
            cams = deep_merge(cams, {"cameras": {camera_name: ov or {}}})
        self.cfgs["cameras"] = cams
        self.cabin = Cabin(self.cfgs["cabin"])
        self.camera = Camera.from_config(camera_name, self.cfgs["cameras"], self.cabin.reference_head_center)
        self.seed, self.emulator_seed = seed, emulator_seed
        self.rng, self.emulator_rng = rng_streams(seed, emulator_seed)      # behaviour, emulator
        self.behavior = BehaviorGenerator(self.cabin, self.cfgs["behavior"], self.scenario, self.rng)
        self.frames = self.behavior.generate()
        self.emulator = Emulator(self.cfgs["emulator"], self.cabin, self.camera, self.emulator_rng)
        self.records: list[dict] = []
        self._truths: list[dict] | None = None

    def truths(self) -> list[dict]:
        """Per-frame ground truth (with behaviour labels) of this run; computed once and cached. The emulator only
        reads it, so the same truths can be emulated many times (see emulate)."""
        if self._truths is None:
            from .events import label_events
            from .truth import compute_truth
            self.events = label_events(self.frames, self.cabin, self.behavior.cfg)
            out = []
            for i, f in enumerate(self.frames):
                truth = compute_truth(self.cabin, self.camera, f)
                truth.setdefault("mouth_openness", f.mouth_openness)
                truth.setdefault("yawning", f.yawning)
                # phone_use is scripted; the phone may be outside the image (e.g. in the lap), where no model can
                # see it. phone_visible / phone_use_visible say whether the truth phone box is in the image.
                truth["phone_visible"] = any(d["class"] == "phone" for d in truth["detections"])
                truth["events"] = dict(self.events[i], phone_use_visible=bool(self.events[i].get("phone_use"))
                                       and truth["phone_visible"])
                out.append(truth)
            self._truths = out
        return self._truths

    def _records(self, emulator: Emulator) -> list[dict]:
        return [{"frame": i, "t": f.t, "camera": self.camera.name, "truth": truth, "pred": emulator.emulate(truth)}
                for i, (f, truth) in enumerate(zip(self.frames, self.truths()))]

    def run(self) -> list[dict]:
        self.records = self._records(self.emulator)
        return self.records

    def emulate(self, emulator_overrides: dict | None = None, emulator_seed: int | None = None) -> list[dict]:
        """Records of the same behaviour / truth with another emulator config (overrides deep-merged into this
        run's emulator config) and / or emulator seed, without recomputing the truth (the costly part). Equal to
        Simulation(..., emulator_overrides=..., emulator_seed=...).run() for the same seed (emulator_seed None =
        this run's emulator seed). Does not touch
        self.records."""
        from .config import deep_merge
        _, rng = rng_streams(self.seed, self.emulator_seed if emulator_seed is None else emulator_seed)
        emu = Emulator(deep_merge(self.cfgs["emulator"], emulator_overrides or {}), self.cabin, self.camera, rng)
        return self._records(emu)

    def summarize(self) -> dict:
        """metrics.summarize of the records with this run's emulator config."""
        from .metrics import summarize
        return summarize(self.records, self.cfgs["emulator"])

    def write(self, out_dir: str | Path, fmt: str = "full", include_face: bool | None = None,
              dms_layout=None, labels: str = "sim") -> list[Path]:
        """Write the run. fmt: "full" (nested truth+pred JSONL), "pipeline" (flat pipeline-format JSONL plus
        the matching truth JSONL, see schema.py), "both" (full + pipeline), "dmsbin" (binary log, layout
        from `dms_layout` / configs/binlog.yaml, see binlog.py) or "all". include_face: emit the optional
        face OD box in the pipeline JSONL and dms.bin; None = format default (pipeline: no with labels "sim", yes with
        "mapped"; dms.bin: yes, if the layout maps `face`). labels: pipeline JSONL label mode ("sim" | "mapped" =
        the class map of `dms_layout`, see schema.py).
        Returns the written paths; dms.bin write stats are kept in self.dmsbin_stats."""
        if fmt not in FORMATS:
            raise ValueError(f"fmt must be one of {FORMATS}, got {fmt!r}")
        from . import schema
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        stem = f"{self.scenario['name']}__{self.camera.name}"
        label_map = None
        if labels == "mapped":
            from .binlog import DmsBinLayout
            label_map = DmsBinLayout.coerce(dms_layout).spec
        paths = []
        if fmt in ("full", "both", "all"):
            paths.append(_write_jsonl(out / f"{stem}.jsonl", self.records))
        if fmt in ("pipeline", "both", "all"):
            paths.append(_write_jsonl(out / f"{stem}.pipeline.jsonl",
                                      (schema.to_pipeline(r, self.camera, include_face, labels=labels, label_map=label_map)
                                       for r in self.records)))
            paths.append(_write_jsonl(out / f"{stem}.truth.jsonl", (schema.to_truth(r, self.camera) for r in self.records)))
        if fmt in ("dmsbin", "all"):
            from .binlog import write_dms_bin
            self.dmsbin_stats = write_dms_bin(self.records, self.camera, dms_layout, out / f"{stem}.dms.bin",
                                              include_face=True if include_face is None else bool(include_face))
            paths.append(self.dmsbin_stats["path"])
        with open(out / f"{stem}__camera.json", "w", encoding="utf-8", newline="\n") as fh:
            desc = self.camera.describe()
            drv = self.cfgs["cabin"]["driver"]
            desc["driver"] = {"eye_point_offset": list(drv["eye_point_offset"]), "head_scale": self.cabin.head_scale}
            if self.variant:
                desc["variant"] = self.variant
            json.dump(desc, fh, indent=2)
        return paths


def _write_jsonl(path: Path, rows) -> Path:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=_json_default) + "\n")
    return path

def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))

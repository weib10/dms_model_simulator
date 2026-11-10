"""Export simulated model outputs for an external post-processing program (settings in configs/export.yaml).

    simulate -> export the OD boxes (dms.bin or one YAML per frame) + the pipeline / truth JSONL

The post-processing itself (tracking, time windows, alerts) is not part of this repo: run it on the exported
files and compare its alerts with the behaviour description of the scenario (catalog rows write one, see
behavior_card.py).

    from dms_model_simulator.export import export_scenario
    sim, stats, truth_path = export_scenario("phone_distraction", "dash_right", "frame_yaml", "out/x")

Steps on their own: export_dms_bin / export_frame_yaml (also `dms-sim export`); read_frame_yaml reads an export
back (round trip).
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterator
from .config import CONFIG_DIR, load_yaml
from .schema import FIELDS as F

DEFAULT_CONFIG = CONFIG_DIR / "export.yaml"
INPUT_FORMATS = ("dms_bin", "frame_yaml")
SUFFIX = {"dms_bin": ".dms.bin", "frame_yaml": ".frames"}   # frame_yaml: a directory
IMAGE_MODES = ("none", "render")


class ExportError(RuntimeError):
    """An export step failed: bad config or unreadable export."""


def load_config(cfg: dict | str | Path | None = None) -> dict:
    """Export config (dict as is, else YAML; None = configs/export.yaml)."""
    if isinstance(cfg, dict):
        return cfg
    return load_yaml(Path(cfg) if cfg else DEFAULT_CONFIG)


def _layout(cfg: dict, layout=None):
    from .binlog import DmsBinLayout
    return DmsBinLayout.coerce(layout if layout is not None else cfg.get("layout"))


# ---- exporters ------------------------------------------------------------------------------------
def export_dms_bin(records, camera, path, layout=None, include_face: bool = True) -> dict:
    """dms.bin (binlog.write_dms_bin). Returns the writer stats plus "format"."""
    from .binlog import write_dms_bin
    st = write_dms_bin(records, camera, layout, path, include_face=include_face)
    return {**st, "format": "dms_bin"}


def _frame_name(i: int, digits: int) -> str:
    return f"{int(i):0{digits}d}"


def _clear_frames(d: Path, ext: str) -> None:
    """Remove the frame files (all-digit stem, given extension) a previous export left in d."""
    for f in d.glob(f"*.{ext}"):
        if f.stem.isdigit():
            f.unlink()


def export_frame_yaml(records, camera, out_dir, cfg: dict | str | Path | None = None, layout=None,
                      include_face: bool = True, image: str = "none", cabin=None) -> dict:
    """One YAML per frame in <out_dir>/labels, named by the 0-based frame index zero-padded to `frame_yaml.digits`
    (8), so lexicographic order = play order:

        detections:                  # [] for a frame without objects (every frame gets a file)
          - class_id: 0              # class id of the layout
            confidence: 0.93
            left: 812.4              # LTWH, original-image pixels
            top: 301.0
            width: 410.2
            height: 488.7
            class_name: face
        timestamp_msec: 50           # round(t * 1000), frame 0 = 0 (same rule as dms.bin timestamp_ms)

    Objects come from the same encoder as dms.bin (binlog.encode_frames: class map, conf thresholds). image
    "render" also writes <out_dir>/images/<same stem>.<image_ext>: the camera view at full camera resolution
    (viz.draw_camera_view clean: face only, no boxes / HUD); it needs `camera` as a Camera and `cabin`.
    <out_dir>/images exists either way (empty for "none"). Frame files of an earlier export in these directories
    are removed first. Returns {"path" (out_dir), "labels_dir", "image_dir", "format", "frames", "objects",
    "images", "skipped", "below_threshold"}."""
    from .binlog import encode_frames
    bc = load_config(cfg)
    yc = bc.get("frame_yaml") or {}
    L = _layout(bc, layout)
    if image not in IMAGE_MODES:
        raise ExportError(f"image must be one of {IMAGE_MODES}, got {image!r}")
    if image == "render" and (cabin is None or isinstance(camera, dict)):
        raise ExportError("image render needs the simulation's Camera and Cabin")
    digits, ext = int(yc.get("digits", 8)), str(yc.get("image_ext", "png"))
    bd, cd = int(yc.get("box_decimals", 3)), int(yc.get("conf_decimals", 6))
    records = list(records)
    if records and max(int(r["frame"]) for r in records) >= 10 ** digits:
        raise ExportError(f"frame_yaml.digits {digits} is too small for {len(records)} frames (file names must sort "
                          f"in play order)")
    out = Path(out_dir)
    labels, images = out / "labels", out / "images"
    labels.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    _clear_frames(labels, "yaml")
    _clear_frames(images, ext)
    by_frame = {int(r["frame"]): r for r in records} if image == "render" else {}
    counts: dict = {}
    n_frames = n_obj = n_img = 0
    for fr in encode_frames(records, camera, L, include_face, counts):
        stem = _frame_name(fr["frame_index"], digits)
        objs = fr["objects"]
        lines = ["detections:" + ("" if objs else " []")]
        for d in objs:
            x1, y1, x2, y2 = (float(v) for v in d["bbox"])
            lines += [f"  - class_id: {d['class_id']}", f"    confidence: {d['conf']:.{cd}f}",
                      f"    left: {x1:.{bd}f}", f"    top: {y1:.{bd}f}",
                      f"    width: {x2 - x1:.{bd}f}", f"    height: {y2 - y1:.{bd}f}",
                      f"    class_name: {L.class_names[d['class_id']]}"]
        lines.append(f"timestamp_msec: {int(round(fr['t'] * 1e3))}")
        (labels / f"{stem}.yaml").write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
        if image == "render":
            from .viz import draw_camera_view
            draw_camera_view(by_frame[fr["frame_index"]], camera, cabin, scale=1.0, clean=True).save(
                images / f"{stem}.{ext}")
            n_img += 1
        n_frames += 1
        n_obj += len(objs)
    return {"path": out, "labels_dir": labels, "image_dir": images, "format": "frame_yaml", "frames": n_frames,
            "objects": n_obj, "images": n_img, "skipped": dict(counts["skipped"]),
            "below_threshold": dict(counts["below_threshold"])}


def export(records, camera, fmt: str, path, cfg: dict | str | Path | None = None, include_face: bool = True,
           layout=None, image: str = "none", cabin=None) -> dict:
    """export_dms_bin or export_frame_yaml (path = its directory) by format name."""
    bc = load_config(cfg)
    if fmt != "frame_yaml" and image != "none":
        raise ExportError(f"image {image!r} needs format frame_yaml, got {fmt!r}")
    if fmt == "dms_bin":
        return export_dms_bin(records, camera, path, _layout(bc, layout), include_face)
    if fmt == "frame_yaml":
        return export_frame_yaml(records, camera, path, bc, layout, include_face, image, cabin)
    raise ExportError(f"format must be one of {INPUT_FORMATS}, got {fmt!r}")


def _labels_dir(path) -> Path:
    p = Path(path)
    return p / "labels" if (p / "labels").is_dir() else p


def read_frame_yaml(path, layout=None) -> Iterator[dict]:
    """Read a frame_yaml export (its directory or its labels directory). Yields frames shaped like
    replay.iter_dms_bin: "frame_id" (from the file name), "timestamp_ms" (timestamp_msec, None when absent), "od"
    [{label, bbox [x1,y1,x2,y2], score}], in file-name order. A class_id outside the layout, a class_name that does
    not match it, or a malformed file raises ExportError naming the file."""
    import yaml
    from .binlog import DmsBinLayout
    L = DmsBinLayout.coerce(layout)
    d = _labels_dir(path)
    files = sorted((f for f in d.glob("*.yaml") if f.stem.isdigit()), key=lambda f: int(f.stem))
    if not files:
        raise ExportError(f"{d}: no frame YAML files (<digits>.yaml)")
    for f in files:
        try:
            doc = yaml.safe_load(f.read_text()) or {}
            dets = doc.get("detections")
            if not isinstance(dets, list):
                raise ValueError("detections must be a list")
            od = []
            for o in dets:
                cid = int(o["class_id"])
                if cid not in L.class_names:
                    raise ValueError(f"class_id {cid} not in the layout")
                name = o.get("class_name")
                if name is not None and name != L.class_names[cid]:
                    raise ValueError(f"class_name {name!r} does not match class_id {cid} ({L.class_names[cid]})")
                x, y, w, h = (float(o[k]) for k in ("left", "top", "width", "height"))
                od.append({F["label"]: L.class_names[cid], F["bbox"]: [x, y, x + w, y + h],
                           F["score"]: float(o["confidence"])})
            ts = doc.get("timestamp_msec")
        except (yaml.YAMLError, KeyError, TypeError, ValueError, AttributeError) as e:
            raise ExportError(f"{f}: bad frame file: {e}") from e
        yield {F["frame_id"]: int(f.stem), F["timestamp_ms"]: None if ts is None else int(ts), F["od"]: od}


def sniff_format(path) -> str:
    """frame_yaml for a directory, else dms_bin."""
    return "frame_yaml" if Path(path).is_dir() else "dms_bin"


def export_scenario(scenario, camera: str, fmt: str, out_dir, seed: int = 0, fps: float | None = None,
                    config: dict | str | Path | None = None, include_face: bool = True, image: str = "none",
                    variant: int | None = None, emulator_overrides: dict | None = None,
                    emulator_seed: int | None = None, full: bool = False, pipeline_face: bool | None = None):
    """Simulate and export one run: the export (<stem>.dms.bin | directory <stem>.frames) plus the pipeline /
    truth JSONL (Simulation.write, labels from `simulation.labels`; full=True also writes the full truth + pred
    JSONL with the emulator's error flags; pipeline_face: the face box in the pipeline JSONL, None = the label
    mode's default). image: none | render (frame_yaml only). variant: driver / camera-mount
    variant; emulator_overrides / emulator_seed: model-error config overrides and noise seed (Simulation).
    Returns (sim, export stats, truth path)."""
    from .simulate import Simulation
    bc = load_config(config)
    sc = bc.get("simulation") or {}
    sim = Simulation(scenario, camera, seed=seed, fps=fps if fps is not None else sc.get("fps"), variant=variant,
                     emulator_overrides=emulator_overrides, emulator_seed=emulator_seed)
    sim.run()
    out = Path(out_dir).resolve()
    labels = sc.get("labels", "sim")
    sim.write(out, fmt="both" if full else "pipeline", labels=labels, include_face=pipeline_face,
              dms_layout=_layout(bc) if labels == "mapped" else None)
    stem = f"{sim.scenario['name']}__{sim.camera.name}"
    if fmt not in INPUT_FORMATS:
        raise ExportError(f"format must be one of {INPUT_FORMATS}, got {fmt!r}")
    st = export(sim.records, sim.camera, fmt, out / f"{stem}{SUFFIX[fmt]}", bc, include_face,
                image=image, cabin=sim.cabin)
    return sim, st, out / f"{stem}.truth.jsonl"

"""Driver and camera-mount variation: one variant = overrides for the cabin driver block and the simulated
camera, drawn uniformly from the ranges in configs/variation.yaml.

A variant is applied like a scenario's own `cabin:` / `camera:` blocks (Simulation), on top of them, so the same
scenario and seed can be replayed with another driver (seat position, sitting height, head size) and another
camera mount (position and aim error)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .config import CONFIG_DIR, load_yaml

AXES = {"eye_point_offset": ("x", "y", "z"), "mount_offset": ("x", "y", "z"),
        "mount_rotation_deg": ("yaw", "pitch", "roll")}
VARIANT_STREAM = 0x5641          # stream id of variant draws: default_rng([variant, VARIANT_STREAM])


def load_variation_config(path: str | Path | None = None) -> dict:
    return load_yaml(path or CONFIG_DIR / "variation.yaml")


def _uniform(rng: np.random.Generator, rng_spec, name: str) -> float:
    lo, hi = (float(v) for v in rng_spec)
    if hi < lo:
        raise ValueError(f"variation {name}: range [{lo}, {hi}] has hi < lo")
    return float(rng.uniform(lo, hi)) if hi > lo else lo


def _block(rng, spec: dict | None, section: str) -> dict:
    out = {}
    for key, val in (spec or {}).items():
        name = f"{section}.{key}"
        if key in AXES:                                  # per-axis ranges -> vector
            val = val or {}
            unknown = set(val) - set(AXES[key])
            if unknown:
                raise ValueError(f"variation {name}: unknown axes {sorted(unknown)} (expected {AXES[key]})")
            out[key] = [round(_uniform(rng, val[a], f"{name}.{a}"), 6) if a in val else 0.0 for a in AXES[key]]
        else:
            out[key] = round(_uniform(rng, val, name), 6)
    return out


def sample_variant(variant: int, cfg: dict | None = None) -> dict:
    """{"variant": N, "cabin": {"driver": {...}}, "camera": {...}} for variant N (deterministic in N). Vector
    entries (eye_point_offset, mount_offset, mount_rotation_deg) are absolute values, axes without a range 0;
    scalar entries without a range keep the nominal value (they are not written)."""
    if int(variant) < 0:
        raise ValueError(f"variant must be >= 0, got {variant}")
    cfg = load_variation_config() if cfg is None else cfg
    unknown = set(cfg) - {"driver", "camera", "hands"}
    if unknown:
        raise ValueError(f"variation config: unknown sections {sorted(unknown)} (expected driver, camera, hands)")
    rng = np.random.default_rng([int(variant), VARIANT_STREAM])
    driver = _block(rng, cfg.get("driver"), "driver")
    camera = _block(rng, cfg.get("camera"), "camera")
    # hands: own stream, so its ranges can change without moving the driver / camera draws of any variant
    driver.update(_block(np.random.default_rng([int(variant), VARIANT_STREAM, 1]), cfg.get("hands"), "hands"))
    return {"variant": int(variant), "cabin": {"driver": driver}, "camera": camera}

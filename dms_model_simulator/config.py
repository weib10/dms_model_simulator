"""YAML loading with deep-merge overrides."""
from __future__ import annotations
import copy
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
SCENARIO_DIR = ROOT / "scenarios"


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def unknown_paths(over, base, path: str = "") -> list[str]:
    """Dotted key paths of `over` that are not in `base` (a mistyped override is silently ignored by deep_merge)."""
    out = []
    if not isinstance(over, dict):
        return out
    for k, v in over.items():
        p = f"{path}.{k}" if path else str(k)
        if not isinstance(base, dict) or k not in base:
            out.append(p)
        elif isinstance(v, dict) and isinstance(base[k], dict):
            out += unknown_paths(v, base[k], p)
    return out


def load_configs(cabin=None, cameras=None, behavior=None, emulator=None) -> dict:
    return {
        "cabin": load_yaml(cabin or CONFIG_DIR / "cabin.yaml"),
        "cameras": load_yaml(cameras or CONFIG_DIR / "cameras.yaml"),
        "behavior": load_yaml(behavior or CONFIG_DIR / "behavior.yaml"),
        "emulator": load_yaml(emulator or CONFIG_DIR / "emulator.yaml"),
    }


def load_scenario(name_or_path: str | Path) -> dict:
    p = Path(name_or_path)
    if not p.exists():
        p = SCENARIO_DIR / f"{name_or_path}.yaml"
    return load_yaml(p)

"""Random scenario generator: Markov chain over gaze targets with lognormal dwell times, plus
Poisson phone, drowsiness, smoking and seatbelt-off episodes. Produces the same ``segments`` structure as
scenarios/*.yaml.

Config: configs/scenario_gen.yaml.

Random streams (all children of the seed, see STREAMS): the gaze-target chain and each episode kind have their
own stream, so changing one kind's rate / duration / modes leaves the other kinds' episodes unchanged. Remaining
coupling: episodes with a `target_bias` (phone, drowsy by default) re-weight the target transitions while they
are active, so changing such a kind changes the target sequence from the first biased transition on.
"""
from __future__ import annotations
import math
import warnings
import zlib
from pathlib import Path
import numpy as np
import yaml
from .config import CONFIG_DIR, deep_merge, load_yaml


def load_gen_config(path: str | Path | None = None) -> dict:
    return load_yaml(path or CONFIG_DIR / "scenario_gen.yaml")


def _lognormal(rng: np.random.Generator, d: dict) -> float:
    """Sample a lognormal with the given *mean* (not median), clipped to [min_s, max_s]."""
    sigma = float(d["sigma"])
    mu = np.log(float(d["mean_s"])) - 0.5 * sigma ** 2
    return float(np.clip(rng.lognormal(mu, sigma), float(d["min_s"]), float(d["max_s"])))


def _lognormal_mean(d: dict) -> float:
    """Mean of the clipped lognormal sampled by _lognormal."""
    sigma, m = float(d["sigma"]), float(d["mean_s"])
    lo, hi = float(d["min_s"]), float(d["max_s"])
    if sigma <= 0:
        return float(np.clip(m, lo, hi))
    mu = math.log(m) - 0.5 * sigma ** 2
    cdf = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    za, zb = (math.log(lo) - mu) / sigma, (math.log(hi) - mu) / sigma
    inner = m * (cdf(zb - sigma) - cdf(za - sigma))          # E[X; lo < X < hi]
    return lo * cdf(za) + hi * (1.0 - cdf(zb)) + inner


def _mean_duration(ecfg: dict, modes: dict | None) -> float:
    """Mean episode duration: modes with their own `duration` weighted by p, the others use the kind's."""
    if not modes or not any(m.get("duration") for m in modes.values()):
        return _lognormal_mean(ecfg["duration"])
    p = np.array([float(m["p"]) for m in modes.values()])
    d = [_lognormal_mean(m.get("duration") or ecfg["duration"]) for m in modes.values()]
    return float(np.dot(p / p.sum(), d))


def _episodes(rng, ecfg: dict, n_ticks: int, res: float, modes: dict | None, name: str = "episodes"
              ) -> list[tuple[int, int, str | None]]:
    """Non-overlapping episodes as (start_tick, end_tick, mode), a renewal process whose long-run rate is
    rate_per_min: the gap from one episode's end to the next start is exponential with mean
    60 / rate - mean duration (clipped at 0, with a warning, when the rate cannot be reached)."""
    rate = float(ecfg["rate_per_min"])
    out = []
    if rate <= 0:
        return out
    gap_mean = 60.0 / rate - _mean_duration(ecfg, modes)
    if gap_mean < 0:
        warnings.warn(f"{name}: rate_per_min {rate} x mean duration exceeds 1 min; episodes are back to back "
                      f"(actual rate {60.0 / _mean_duration(ecfg, modes):.3f}/min)", stacklevel=3)
        gap_mean = 0.0
    t = 0.0
    total = n_ticks * res
    while True:
        t += rng.exponential(gap_mean) if gap_mean > 0 else 0.0
        if t >= total:
            break
        dur = _lognormal(rng, ecfg["duration"])
        mode = None
        if modes:
            names = list(modes)
            p = np.array([float(modes[m]["p"]) for m in names])
            mode = names[int(rng.choice(len(names), p=p / p.sum()))]
            if modes[mode].get("duration"):       # a mode's own duration (drawn after the mode: kinds without
                dur = _lognormal(rng, modes[mode]["duration"])      # mode durations draw exactly as before)
        a = int(round(t / res))
        b = min(n_ticks, a + max(1, int(round(dur / res))))
        if a < b:
            out.append((a, b, mode))
        t = b * res
    return out


def _active(eps, tick):
    for a, b, m in eps:
        if a <= tick < b:
            return (a, b, m)
    return None


# episode kinds in attribute priority order: (config key, generator output key)
EPISODE_KINDS = (("phone_episodes", "phone_episodes"), ("drowsy_episodes", "drowsy_episodes"),
                 ("smoking_episodes", "smoking_episodes"), ("seatbelt_off_episodes", "seatbelt_off_episodes"))
# fixed stream id per consumer: stream k of seed s = default_rng([s, k]); new kinds get new ids
STREAMS = {"targets": 0, "phone_episodes": 1, "drowsy_episodes": 2, "smoking_episodes": 3, "seatbelt_off_episodes": 4}


def _stream(seed: int, name: str) -> np.random.Generator:
    """Stream of a consumer. Extra episode kinds (any other `*_episodes` block) get an id from a hash of their name,
    so adding, removing or reordering kinds never shifts another kind's draws."""
    k = STREAMS.get(name, 1000 + zlib.crc32(name.encode()) % 1_000_000)
    return np.random.default_rng([int(seed), k])


def episode_kinds(cfg: dict) -> list[tuple[str, str]]:
    """The four built-in kinds in priority order, then every other `*_episodes` block of the config in file
    order (lower priority: they only set attributes the earlier kinds leave free)."""
    builtin = [k for k, _ in EPISODE_KINDS]
    return list(EPISODE_KINDS) + [(k, k) for k in cfg if k.endswith("_episodes") and k not in builtin]


def _attrs(ecfg: dict, mode: str | None) -> dict:
    return (ecfg["modes"][mode].get("attrs") if mode else ecfg.get("attrs")) or {}


def _bias(ecfg: dict, mode: str | None) -> dict:
    return (ecfg["modes"][mode].get("target_bias") if mode else ecfg.get("target_bias")) or {}


def generate(cfg: dict, duration_s: float, seed: int = 0, name: str | None = None) -> dict:
    """Return a scenario dict {name, description, behavior, generator, segments}. duration_s is rounded to the
    nearest time_resolution_s tick and must be at least one tick."""
    res = float(cfg["time_resolution_s"])
    n_ticks = int(math.floor(float(duration_s) / res + 0.5 + 1e-9))      # nearest tick (5.05 s -> 5.1 s at 0.1 s)
    if n_ticks < 1:
        raise ValueError(f"duration_s={duration_s} rounds to 0 ticks of time_resolution_s={res} s; "
                         f"a scenario needs at least one tick")
    trans = cfg["transitions"]
    dwell = cfg["dwell"]
    kinds = []                       # (cfg, episodes, output key), priority order
    builtin = {k for k, _ in EPISODE_KINDS}
    for key, out_key in episode_kinds(cfg):
        ecfg = cfg.get(key)
        if not ecfg or (key not in builtin and float(ecfg.get("rate_per_min", 0)) <= 0):
            continue                 # an extra kind at rate 0 leaves no trace (scenario and row ids unchanged)
        kinds.append((ecfg, _episodes(_stream(seed, key), ecfg, n_ticks, res, ecfg.get("modes"), key), out_key))
    rng = _stream(seed, "targets")

    # 1) gaze targets: Markov chain, transition weights biased by the episodes active at the transition
    visits = []                      # (start_tick, end_tick, target)
    tick, target = 0, cfg["start_target"]
    while tick < n_ticks:
        d = _lognormal(rng, dwell.get(target, dwell["default"]))
        end = min(n_ticks, tick + max(1, int(round(d / res))))
        visits.append((tick, end, target))
        tick = end
        row = trans[target]
        names = list(row)
        w = np.array([float(row[n]) for n in names])
        bias = {}
        for ecfg, eps, _ in kinds:
            e = _active(eps, tick)
            if e:
                b = _bias(ecfg, e[2])
                bias = {k: bias.get(k, 1.0) * float(b.get(k, 1.0)) for k in set(bias) | set(b)}
        w = w * np.array([float(bias.get(n, 1.0)) for n in names])
        if w.sum() <= 0:
            w = np.array([float(row[n]) for n in names])
        target = names[int(rng.choice(len(names), p=w / w.sum()))]

    # 2) split visits at episode boundaries and attach episode attributes (earlier kinds keep their keys)
    all_eps = [ep for _, eps, _ in kinds for ep in eps]
    cuts = sorted({a for a, _, _ in all_eps} | {b for _, b, _ in all_eps})
    segments = []
    for a, b, tgt in visits:
        pts = [a] + [c for c in cuts if a < c < b] + [b]
        for s, e in zip(pts[:-1], pts[1:]):
            seg, owned = {"target": tgt}, set()
            for ecfg, eps, _ in kinds:
                ep = _active(eps, s)
                if ep:
                    at = {k: v for k, v in _attrs(ecfg, ep[2]).items() if k not in owned}
                    seg = deep_merge(seg, at)
                    owned |= set(at)
            seg["duration"] = round((e - s) * res, 6)
            segments.append(seg)
    gen = {"seed": int(seed), "duration_s": float(duration_s)}
    for ecfg, eps, out_key in kinds:
        gen[out_key] = [[round(a * res, 6), round(b * res, 6)] + ([m] if ecfg.get("modes") else []) for a, b, m in eps]
    return {
        "name": name or f"random_{int(round(duration_s))}s_seed{seed}",
        "description": f"Randomly generated scenario (Markov targets, seed {seed}, {duration_s} s).",
        "behavior": {},
        "generator": gen,
        "segments": segments,
    }


def save_yaml(scenario: dict, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    head = {k: v for k, v in scenario.items() if k != "segments"}
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(head, fh, sort_keys=False, default_flow_style=None)
        fh.write("segments:\n")
        for seg in scenario["segments"]:
            fh.write("  - " + yaml.safe_dump(seg, sort_keys=False, default_flow_style=True, width=1000).strip() + "\n")
    return p

"""Scenario catalog: parameterised scenario entries -> concrete scenarios x cameras x variants -> exported runs.

An entry (catalog/entries/<category>/<id>.yaml) is a logical scenario: a scenario template whose numbers are
parameters, with a title / summary saying what the driver does. `expand` draws parameter values and writes a
manifest (one row per concrete run); `run` exports each row (frame_yaml + pipeline / truth / full JSONL) with a
behaviour card (behavior_card.py) for whoever checks a post-processing program on it. See catalog/README.md.
"""
from __future__ import annotations
import copy
import hashlib
import itertools
import json
import re
from fractions import Fraction
from pathlib import Path
import numpy as np
from .config import ROOT, deep_merge, load_configs, load_scenario, load_yaml, unknown_paths as _unknown_paths

CATALOG_DIR = ROOT / "catalog"
CATEGORIES = {   # first level of the id: what kind of driving the entry exercises
    "POS": "行為真的發生：分心、閉眼、打哈欠、講手機、抽菸、沒繫安全帶",
    "NEG": "長得像但不是：看後照鏡、喝東西、手放臉旁、正常眨眼",
    "BND": "時長掃描：同一個行為從短到長成對排開",
    "CLR": "停止與再發生：行為停下、隔一段時間再來一次",
    "ROB": "偵測變差：漏框、掉幀、延遲、抖動加重",
    "NAT": "長時間日常：隨機產生的長段駕駛",
    "REG": "回歸：之前出過問題的片段",
}
FAMILIES = {     # second level: the behaviour family
    "DIS": "視線或頭離開前方",
    "DRW": "閉眼、瞌睡",
    "YWN": "打哈欠",
    "PHN": "手機",
    "SMK": "抽菸",
    "SBT": "安全帶",
    "ALL": "混合或日常",
}
STATUSES = ("ready", "planned")
SOURCES = ("written", "recorded", "generated")
ID_RE = re.compile(r"^(POS|NEG|BND|CLR|ROB|NAT|REG)-(DIS|DRW|YWN|PHN|SMK|SBT|ALL)-[A-Z0-9]+(-[A-Z0-9]+)*$")
PARAM_KINDS = ("fixed", "choice", "uniform", "grid", "levels")
LEVELS = ("small", "medium", "large")   # difficulty: how close the behaviour gets to the threshold it resembles
RESERVED_PARAMS = {"level"}             # set by the level axis; templates may use ${level}
SEGMENT_KEYS = {"target", "duration", "head_pose", "eye_openness", "closure", "yawn", "blink", "mouth_openness",
                "hand", "phone", "cigarette", "seatbelt", "held"}
DEFAULT_VARIATION = {"cameras": ["dash_right", "steering_column"], "variants": 0, "nominal": True, "seeds": [0],
                     "emulator_seeds": [0, 1, 2], "profiles": ["nominal"]}
DEFAULT_FPS = 20.0          # model output rate
ENTRY_KEYS = {"id", "title", "summary", "status", "needs", "source", "tags", "behaviors", "params", "template",
              "from_scenario", "generator", "variation", "fps", "_path"}
VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class CatalogError(ValueError):
    pass


# ---- behaviour list (catalog/behaviors.yaml) ------------------------------------------------------------
BEHAVIOR_KINDS = ("violation", "normal", "condition")
COMMON = {"often": "常見", "sometimes": "偶爾", "rare": "少見"}
SEPARABLE = {"easy": "好分", "medium": "普通", "hard": "難分"}
BEHAVIOR_ID_RE = re.compile(r"^[VNC][0-9]{2}$")
EXPECT_KEYS = {"label", "max_s", "never"}


def load_behaviors(path=None) -> dict:
    """{id: behaviour} from catalog/behaviors.yaml (V = violation, N = normal, C = condition)."""
    return dict(load_yaml(path or CATALOG_DIR / "behaviors.yaml").get("behaviors") or {})


def is_core(b: dict) -> bool:
    """Common and easy to tell apart: these must never go wrong (tests/test_behaviors.py)."""
    return b.get("common") == "often" and b.get("separable") == "easy"


def validate_behaviors(behaviors: dict) -> list[str]:
    """Problems of the behaviour list (empty = valid)."""
    errs = []
    for k, b in behaviors.items():
        if not BEHAVIOR_ID_RE.match(str(k)):
            errs.append(f"{k}: id is not V/N/C + two digits")
        for key in ("name", "kind", "family", "common", "separable"):
            if key not in b:
                errs.append(f"{k}: missing {key}")
        if b.get("kind") not in BEHAVIOR_KINDS or str(k)[:1] != {"violation": "V", "normal": "N",
                                                                  "condition": "C"}.get(b.get("kind"), "?"):
            errs.append(f"{k}: kind {b.get('kind')!r} does not match the id prefix")
        if b.get("family") not in FAMILIES:
            errs.append(f"{k}: family {b.get('family')!r} is not one of {list(FAMILIES)}")
        if b.get("common") not in COMMON:
            errs.append(f"{k}: common must be one of {list(COMMON)}")
        if b.get("separable") not in SEPARABLE:
            errs.append(f"{k}: separable must be one of {list(SEPARABLE)}")
        errs += [f"{k}: confused_with {c!r} is not in the list" for c in b.get("confused_with") or []
                 if c not in behaviors]
        ex = b.get("expect") or {}
        errs += [f"{k}: unknown expect key {q!r}" for q in sorted(set(ex) - EXPECT_KEYS)]
        if is_core(b) and b.get("kind") != "condition" and not ex:
            errs.append(f"{k}: core behaviour (often, easy) needs `expect`")
        if b.get("kind") == "violation" and ex and "label" not in ex:
            errs.append(f"{k}: violation expect needs `label`")
    return errs


# ---- loading ----------------------------------------------------------------------------------------
def load_profiles(path=None) -> dict:
    """{name: emulator_overrides dict} from catalog/profiles.yaml. A profile may instead be `compose: "a+b"` (a
    named stack of other profiles, see resolve_profile); it is resolved here, so callers see plain overrides."""
    d = load_yaml(path or CATALOG_DIR / "profiles.yaml")
    raw = d.get("profiles") or {}
    bad = sorted(k for k in raw if "+" in k)
    if bad:
        raise CatalogError(f"profile names cannot contain '+' (it stacks profiles): {bad}")
    plain = {k: (v or {}).get("emulator_overrides") or {} for k, v in raw.items() if "compose" not in (v or {})}
    out = dict(plain)
    for k, v in raw.items():
        if "compose" in (v or {}):
            out[k] = resolve_profile(str(v["compose"]), plain)      # stacks of plain profiles only (no nesting)
    return {k: out[k] for k in raw}


def resolve_profile(name: str, profiles: dict) -> dict:
    """Emulator overrides of a profile name or a stack "a+b+c": each part deep-merged over the previous ones (later
    parts win), so dimension profiles (jitter, jumps, misses, drops, ...) combine on the command line or in an
    entry's variation.profiles. Unknown parts raise CatalogError."""
    parts = [p.strip() for p in str(name).split("+")]
    unknown = [p for p in parts if p not in profiles]
    if unknown or not all(parts):
        raise CatalogError(f"unknown profile {unknown or name!r} (catalog/profiles.yaml: {sorted(profiles)})")
    out: dict = {}
    for p in parts:
        out = deep_merge(out, profiles[p])
    return out


def load_entries(root=None) -> list[dict]:
    """Every entry under catalog/entries (sorted by id); each gets `_path`."""
    root = Path(root or CATALOG_DIR / "entries")
    out = []
    for p in sorted(root.rglob("*.yaml")):
        e = load_yaml(p)
        e["_path"] = str(p)
        out.append(e)
    ids = [e.get("id") for e in out]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise CatalogError(f"duplicate entry ids: {dup}")
    return sorted(out, key=lambda e: str(e.get("id")))


def category(entry: dict) -> str:
    return str(entry["id"]).split("-")[0]


def family(entry: dict) -> str:
    return str(entry["id"]).split("-")[1]


def select(entries: list[dict], selectors: list[str] | None = None, status: str | None = None,
           tag: str | None = None) -> list[dict]:
    """Entries matching any selector (an id, a category like BND, a family like PHN, `BND-PHN`, or `all`) and every
    given filter."""
    sel = [s.upper() for s in (selectors or ["all"])]
    out = []
    for e in entries:
        i = str(e["id"])
        if not ("ALL" in sel or i in sel or category(e) in sel or family(e) in sel
                or any(i.startswith(s + "-") for s in sel)):
            continue
        if status and e.get("status", "ready") != status:
            continue
        if tag and tag not in (e.get("tags") or []):
            continue
        out.append(e)
    return out


# ---- parameters and templates -------------------------------------------------------------------------
def _spec_kind(spec) -> str:
    if not isinstance(spec, dict):
        return "fixed"
    kinds = [k for k in PARAM_KINDS if k in spec]
    if len(kinds) != 1:
        raise CatalogError(f"parameter spec needs exactly one of {PARAM_KINDS}, got {spec!r}")
    return kinds[0]


def _quantize(v, step: float | None):
    if step is None or not isinstance(v, (int, float)) or isinstance(v, bool):
        return v
    return round(round(float(v) / step) * step, 6)


def param_values(spec, mode: str, rng: np.random.Generator | None = None, step: float | None = None) -> list:
    """Values of one parameter. mode grid / boundary: every listed value (uniform: lo, mid, hi for grid, mid for
    boundary; choice: all for grid, the first for boundary); mode random: one draw. Uniform values are rounded to
    `step` (one frame); listed values are kept as written (validate rejects durations that are not whole frames).
    A `levels` spec is not a value list: expand_params resolves it per row with level_specs first."""
    kind = _spec_kind(spec)
    if kind == "levels":
        raise CatalogError("a levels parameter takes its value from the row's level (use level_specs)")
    if kind == "fixed":
        return [spec["fixed"] if isinstance(spec, dict) else spec]
    if kind == "choice":
        ch = list(spec["choice"])
        if mode == "random":
            return [ch[int(rng.integers(len(ch)))]]
        return ch if mode == "grid" else ch[:1]
    if kind == "uniform":
        lo, hi = map(float, spec["uniform"])
        if mode == "random":
            return [_quantize(rng.uniform(lo, hi), step)]
        return [_quantize(x, step) for x in ((lo, (lo + hi) / 2, hi) if mode == "grid" else ((lo + hi) / 2,))]
    if kind == "grid":
        g = list(spec["grid"])
        return [g[int(rng.integers(len(g)))]] if mode == "random" else g


def has_levels(entry: dict) -> bool:
    """True when any parameter of the entry is a `levels` spec (the entry then has small / medium / large rows)."""
    return any(_spec_kind(s) == "levels" for s in (entry.get("params") or {}).values())


def level_specs(specs: dict, level: str | None) -> dict:
    """Parameter specs with every `levels` spec replaced by its value at `level` (a fixed spec)."""
    out = {}
    for k, s in specs.items():
        if _spec_kind(s) == "levels":
            if level is None:
                raise CatalogError(f"parameter {k!r} has levels but no level was chosen")
            out[k] = {"fixed": s["levels"][level]}
        else:
            out[k] = s
    return out


def expand_repeats(segments: list) -> list:
    """Segments with every `{repeat: n, segments: [...]}` block replaced by n copies of its segments (nested blocks
    allowed), so "n glances of t seconds" can be one parameterised block."""
    out = []
    for s in segments:
        if isinstance(s, dict) and "repeat" in s:
            extra = set(s) - {"repeat", "segments"}
            n = s["repeat"]
            if extra or not isinstance(s.get("segments"), list):
                raise CatalogError(f"repeat block takes only `repeat` and a `segments` list, got {sorted(s)}")
            if not isinstance(n, int) or isinstance(n, bool) or n < 0:
                raise CatalogError(f"repeat must be a whole number >= 0, got {n!r}")
            inner = expand_repeats(s["segments"])
            for _ in range(n):
                out += copy.deepcopy(inner)
        else:
            out.append(s)
    return out


def substitute(obj, params: dict):
    """Template with "${name}" replaced: a string that is exactly one placeholder takes the parameter's value (any
    type); placeholders inside a longer string are formatted in. No arithmetic."""
    if isinstance(obj, dict):
        return {k: substitute(v, params) for k, v in obj.items()}
    if isinstance(obj, list):
        return [substitute(v, params) for v in obj]
    if isinstance(obj, str):
        m = VAR_RE.fullmatch(obj)
        if m:
            if m.group(1) not in params:
                raise CatalogError(f"template uses unknown parameter ${{{m.group(1)}}}")
            return copy.deepcopy(params[m.group(1)])

        def rep(mm):
            if mm.group(1) not in params:
                raise CatalogError(f"template uses unknown parameter ${{{mm.group(1)}}}")
            return str(params[mm.group(1)])
        return VAR_RE.sub(rep, obj)
    return obj


def build_scenario(entry: dict, params: dict, name: str, seed: int = 0) -> dict:
    """Concrete scenario dict of an entry for one parameter set."""
    if "from_scenario" in entry:
        sc = load_scenario(ROOT / entry["from_scenario"])
    elif "generator" in entry:
        from . import scenario_gen
        g = substitute(entry["generator"], params)
        cfg = deep_merge(scenario_gen.load_gen_config(), g.get("config"))
        sc = scenario_gen.generate(cfg, float(g["duration_s"]), int(g.get("seed", seed)), name=name)
    else:
        sc = dict(substitute(copy.deepcopy(entry["template"]), params))
        sc["segments"] = expand_repeats(sc.get("segments") or [])
    sc = dict(sc)
    sc["name"] = name
    sc["description"] = f"{entry['id']}: {entry.get('title', '')} {json.dumps(params, ensure_ascii=False)}".strip()
    return sc


def segment_times(scenario: dict) -> list[tuple[float, float]]:
    """(start, end) of every segment in seconds (exact sums)."""
    out, cum = [], Fraction(0)
    for s in scenario["segments"]:
        d = Fraction(str(s["duration"])) if isinstance(s["duration"], float) else Fraction(s["duration"])
        out.append((float(cum), float(cum + d)))
        cum += d
    return out


# ---- validation --------------------------------------------------------------------------------------
def _frame_multiple(v, fps: float) -> bool:
    try:
        x = Fraction(str(v)) * Fraction(str(fps))
    except (ValueError, TypeError):
        return False
    return x.denominator == 1


def validate(entry: dict, profiles: dict, cfgs: dict | None = None) -> list[str]:
    """Problems of one entry (empty = valid). Builds every grid-mode scenario and checks segment keys, targets,
    hand / phone / cigarette states and frame-aligned durations."""
    errs: list[str] = []
    eid = entry.get("id", "?")
    try:
        if not ID_RE.match(str(eid)):
            errs.append(f"id {eid!r} is not <POS|NEG|BND|CLR|ROB|NAT|REG>-<DIS|DRW|YWN|PHN|SMK|SBT|ALL>-<TOPIC>")
        for k in ("title", "status"):
            if k not in entry:
                errs.append(f"missing {k}")
        errs += [f"unknown key {k!r}" for k in sorted(set(entry) - ENTRY_KEYS)]
        if entry.get("status", "ready") not in STATUSES:
            errs.append(f"status must be one of {STATUSES}")
        beh_ids = entry.get("behaviors")
        if not beh_ids:
            errs.append("missing behaviors (ids from catalog/behaviors.yaml, main behaviour first)")
        else:
            known = load_behaviors()
            errs += [f"behaviors: {b!r} is not in catalog/behaviors.yaml" for b in beh_ids if b not in known]
        if entry.get("source", "written") not in SOURCES:
            errs.append(f"source must be one of {SOURCES}")
        rel = Path(entry.get("_path", "")).parent.name
        if entry.get("_path") and rel != category(entry):
            errs.append(f"file is in {rel}/, id says {category(entry)}")
        if entry.get("status") == "planned":
            if not entry.get("needs"):
                errs.append("planned entry needs `needs` (the missing simulator capability)")
            return errs
        kinds = [k for k in ("template", "from_scenario", "generator") if k in entry]
        if len(kinds) != 1:
            errs.append(f"needs exactly one of template / from_scenario / generator, got {kinds}")
            return errs
        errs += [f"variation: unknown key {k!r}" for k in sorted(set(entry.get("variation") or {}) - set(DEFAULT_VARIATION))]
        var = {**DEFAULT_VARIATION, **(entry.get("variation") or {})}
        cfgs = cfgs or load_configs()
        if "generator" in entry:
            from .scenario_gen import load_gen_config
            g = entry["generator"]
            if "duration_s" not in g:
                errs.append("generator needs duration_s")
            errs += [f"generator config {q!r} is not in configs/scenario_gen.yaml"
                     for q in _unknown_paths(g.get("config"), load_gen_config())]
        for c in var["cameras"]:
            if c not in cfgs["cameras"]["cameras"]:
                errs.append(f"unknown camera {c!r}")
        for p in var["profiles"]:
            try:
                ov = resolve_profile(p, profiles)
            except CatalogError:
                errs.append(f"unknown profile {p!r} (catalog/profiles.yaml)")
            else:
                errs += [f"profile {p}: override {q!r} is not in configs/emulator.yaml"
                         for q in _unknown_paths(ov, cfgs["emulator"])]
        fps = float(entry.get("fps", DEFAULT_FPS))
        for name, spec in (entry.get("params") or {}).items():
            if name in RESERVED_PARAMS:
                errs.append(f"parameter name {name!r} is reserved (the row's level)")
            if _spec_kind(spec) == "levels":
                lv = spec["levels"]
                if not isinstance(lv, dict) or set(lv) != set(LEVELS):
                    errs.append(f"parameter {name!r}: levels needs exactly {', '.join(LEVELS)}")
        if errs:
            return errs
        for params in expand_params(entry, "grid", np.random.default_rng(0), max_rows=10_000):
            sc = build_scenario(entry, params, "validate")
            errs += _check_scenario(sc, cfgs, fps, params)
            if errs:
                break
    except (CatalogError, KeyError, TypeError, ValueError) as e:
        errs.append(f"{type(e).__name__}: {e}")
    return list(dict.fromkeys(errs))


def _check_scenario(sc: dict, cfgs: dict, fps: float, params: dict) -> list[str]:
    errs = [f"behavior override {p!r} is not in configs/behavior.yaml"
            for p in _unknown_paths(sc.get("behavior"), cfgs["behavior"])]
    errs += [f"cabin override {p!r} is not in configs/cabin.yaml"
             for p in _unknown_paths({k: v for k, v in (sc.get("cabin") or {}).items() if k not in ("targets", "hands")},
                                     cfgs["cabin"])]
    cab = deep_merge(cfgs["cabin"], sc.get("cabin"))
    beh = deep_merge(cfgs["behavior"], sc.get("behavior"))   # segment defaults, as BehaviorGenerator applies them
    targets = set(cab["targets"])
    hands = set(cab["hands"]) | {"none"}
    if not sc.get("segments"):
        return ["scenario has no segments"]
    for i, s in enumerate(sc["segments"]):
        extra = set(s) - SEGMENT_KEYS
        if extra:
            errs.append(f"segment {i}: unknown keys {sorted(extra)}")
        if s.get("target") not in targets:
            errs.append(f"segment {i}: unknown target {s.get('target')!r}")
        d = s.get("duration")
        if not isinstance(d, (int, float)) or isinstance(d, bool) or d <= 0:
            errs.append(f"segment {i}: duration must be a positive number, got {d!r} ({params})")
        elif not _frame_multiple(d, fps):
            errs.append(f"segment {i}: duration {d} s is not a whole number of frames at {fps:g} fps ({params})")
        if "hand" in s and s["hand"] not in hands:
            errs.append(f"segment {i}: unknown hand state {s['hand']!r}")
        ph = s.get("phone")
        if ph not in (None, "none", "lap", "call") and ph not in targets:
            errs.append(f"segment {i}: phone {ph!r} is not lap / call / none / a target")
        if s.get("cigarette") not in (None, "none", "mouth", "hand"):
            errs.append(f"segment {i}: cigarette must be none / mouth / hand")
        held = str(s.get("held", beh["held"]))
        if held != "none":
            if held not in (cab.get("held_objects") or {}):
                errs.append(f"segment {i}: held {held!r} is not none / {' / '.join(cab.get('held_objects') or {})}")
            if s.get("hand", beh["hand"]) != "at_mouth":
                errs.append(f"segment {i}: held {held!r} needs hand: at_mouth")
        if "head_pose" in s and (not isinstance(s["head_pose"], list) or len(s["head_pose"]) != 3):
            errs.append(f"segment {i}: head_pose must be [yaw, pitch, roll]")
    return errs


def save_scenario(sc: dict, path) -> Path:
    """Scenario YAML: header block style, one flow-style line per segment (readable, unicode kept)."""
    import yaml
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    head = {k: v for k, v in sc.items() if k != "segments"}
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(yaml.safe_dump(head, sort_keys=False, default_flow_style=False, allow_unicode=True, width=1000))
        fh.write("segments:\n")
        for seg in sc["segments"]:
            fh.write("  - " + yaml.safe_dump(seg, sort_keys=False, default_flow_style=True, allow_unicode=True,
                                             width=1000).strip() + "\n")
    return p


# ---- expansion ---------------------------------------------------------------------------------------
def expand_params(entry: dict, mode: str, rng: np.random.Generator, n: int = 1,
                  max_rows: int = 500, levels: list[str] | None = None, overrides: dict | None = None) -> list[dict]:
    """Parameter sets of one entry. grid / boundary: cartesian product of each parameter's values (boundary keeps
    the `grid` values and takes one value of the others); random: n independent draws, repeated
    parameter sets dropped (so fewer than n when the parameters have few values).

    Level axis: when the entry has `levels` parameters, every set belongs to one level (`levels`, default all
    three) and carries it as params["level"]; all `levels` parameters of a set take that level's value (one row
    per level, not a product of levels). In random mode each draw picks one of the given levels.
    overrides: {name: value} fixes those parameters (only the ones the entry has), levels or not."""
    specs = dict(entry.get("params") or {})
    for k, v in (overrides or {}).items():
        if k in specs:
            specs[k] = {"fixed": v}
    step = 1.0 / float(entry.get("fps", DEFAULT_FPS))
    if not specs:
        return [{}]
    lv_list: list = [None]
    if any(_spec_kind(s) == "levels" for s in specs.values()):
        lv_list = list(levels or LEVELS)
        bad = [x for x in lv_list if x not in LEVELS]
        if bad or not lv_list:
            raise CatalogError(f"unknown level {bad or lv_list!r} (levels: {', '.join(LEVELS)})")

    def tag(p: dict, lv) -> dict:
        return {**p, "level": lv} if lv is not None else p
    if mode == "random":     # identical draws would give identical rows (same row_id): keep the first of each
        sets = []
        for _ in range(n):
            lv = lv_list[int(rng.integers(len(lv_list)))] if len(lv_list) > 1 else lv_list[0]
            sp = level_specs(specs, lv)
            sets.append(tag({k: param_values(s, "random", rng, step)[0] for k, s in sp.items()}, lv))
        return list({json.dumps(p, sort_keys=True, default=str): p for p in sets}.values())
    if mode not in ("grid", "boundary"):
        raise CatalogError(f"mode must be grid / boundary / random, got {mode!r}")
    out = []
    for lv in lv_list:
        lists = {}
        for k, s in level_specs(specs, lv).items():
            kind = _spec_kind(s)
            m = "grid" if (mode == "grid" or kind == "grid") else "boundary"
            lists[k] = param_values(s, m, rng, step)
        total = int(np.prod([len(v) for v in lists.values()])) * len(lv_list)
        if total > max_rows:
            raise CatalogError(f"{entry['id']}: {total} parameter sets in mode {mode} (> {max_rows}); use --mode random")
        out += [tag(dict(zip(lists, combo)), lv) for combo in itertools.product(*lists.values())]
    return out


def _row_hash(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:8]


def expand(entries: list[dict], out_dir, mode: str = "boundary", n: int = 5, seed: int = 0,
           profiles: dict | None = None, cameras: list[str] | None = None,
           variants: int | None = None, emulator_seeds: int | None = None, profile_names: list[str] | None = None,
           max_rows: int = 2000, levels: list[str] | None = None, overrides: dict | None = None) -> dict:
    """Write <out>/scenarios/<scn>.yaml and <out>/manifest.json; returns the manifest. One manifest row per
    concrete scenario (x level) x camera x (nominal + variants) x behaviour seed x emulator seed x profile.
    Command-line overrides (cameras, variants, emulator_seeds, profile_names) beat each entry's `variation`;
    levels picks the difficulty levels (default all three) and overrides fixes parameters (see expand_params)."""
    profiles = profiles if profiles is not None else load_profiles()
    cam_names = set(load_configs()["cameras"]["cameras"])
    for c in cameras or []:
        if c not in cam_names:
            raise CatalogError(f"unknown camera {c!r} (configs/cameras.yaml: {sorted(cam_names)})")
    for p in profile_names or []:
        resolve_profile(p, profiles)                 # raises on an unknown name or stack part
    # plan every row first (no files are written when the selection is too large)
    plan, skipped, n_rows = [], [], 0
    for e in entries:
        if e.get("status", "ready") != "ready":
            skipped.append(e["id"])
            continue
        var = {**DEFAULT_VARIATION, **(e.get("variation") or {})}
        nv = var["variants"] if variants is None else variants
        axes = (cameras or var["cameras"], ([None] if var.get("nominal", True) else []) + list(range(int(nv))),
                list(range(emulator_seeds)) if emulator_seeds is not None else list(var["emulator_seeds"]),
                profile_names or var["profiles"])
        # one random stream per entry: an entry's draws do not depend on which other entries are selected
        rng = np.random.default_rng([int(seed), int(hashlib.sha1(str(e["id"]).encode()).hexdigest()[:8], 16)])
        psets = expand_params(e, mode, rng, n, levels=levels, overrides=overrides)
        n_rows += len(psets) * len(var["seeds"]) * int(np.prod([len(x) for x in axes]))
        if n_rows > max_rows:
            raise CatalogError(f"more than {max_rows} rows; narrow the selection or raise --max-rows")
        plan.append((e, var, axes, psets))
    from .repro import fingerprint, fingerprint_id
    fp = fingerprint([e for e, *_ in plan])
    fp_id = fingerprint_id(fp)
    out = Path(out_dir)
    (out / "scenarios").mkdir(parents=True, exist_ok=True)
    rows = []
    for e, var, (cams, vlist, eseeds, profs), psets in plan:
        fps = float(e.get("fps", DEFAULT_FPS))
        for params in psets:
            for bseed in var["seeds"]:
                sc = build_scenario(e, params, "", seed=int(bseed))
                body = {k: v for k, v in sc.items() if k not in ("name", "description")}
                lv = params.get("level")
                scn = sc["name"] = f"{e['id']}__{lv + '__' if lv else ''}{_row_hash(e['id'], params, bseed, body)}"
                spath = save_scenario(sc, out / "scenarios" / f"{scn}.yaml")
                for cam, v, es, pr in itertools.product(cams, vlist, eseeds, profs):
                    rid = f"{scn}__{cam}__v{'n' if v is None else v}__e{es}__{pr}"
                    rows.append({"row_id": rid, "entry": e["id"], "category": category(e), "family": family(e),
                                 "params": params, "level": lv, "scenario": str(spath.resolve()), "camera": cam,
                                 "variant": v,
                                 "seed": int(bseed), "emulator_seed": int(es), "profile": pr,
                                 "emulator_overrides": resolve_profile(pr, profiles), "fps": fps,
                                 "title": e.get("title", ""), "summary": e.get("summary", ""),
                                 "fingerprint_id": fp_id})
    man = {"mode": mode, "seed": seed, "n": n, "levels": list(levels or LEVELS), "overrides": overrides or {},
           "fingerprint": fp, "fingerprint_id": fp_id,
           "entries": sorted({r["entry"] for r in rows}), "skipped_planned": skipped, "rows": rows}
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return man


# ---- index -------------------------------------------------------------------------------------------
def index_markdown(entries: list[dict]) -> str:
    """catalog/INDEX.md: entries grouped by category, then family."""
    L = ["# 情境目錄索引", "",
         "由 `dms-sim catalog index` 產生，不要手改。id = `<類別>-<行為家族>-<主題>`，說明見 [README.md](README.md)。", ""]
    ready = sum(e.get("status") == "ready" for e in entries)
    L += [f"共 {len(entries)} 條：ready {ready}、planned {len(entries) - ready}。", ""]
    L += ["| 類別 | 內容 | 條數 |", "|---|---|---|"]
    for c, q in CATEGORIES.items():
        L.append(f"| {c} | {q} | {sum(category(e) == c for e in entries)} |")
    beh = load_behaviors()
    kind_zh = {"violation": "違規", "normal": "正常", "condition": "外觀條件"}
    L += ["", "## 行為清單", "",
          "編號定義在 [behaviors.yaml](behaviors.yaml)。常見度、好分度是估計值；★ = 常見且好分，一定不能出問題（有測試把關）。", "",
          "| 編號 | 行為 | 類型 | 常見 | 好分 | 會跟誰搞混 | 條目數 |", "|---|---|---|---|---|---|---|"]
    for k, b in beh.items():
        n = sum(k in (e.get("behaviors") or []) for e in entries)
        star = "★ " if is_core(b) else ""
        L.append(f"| {star}{k} | {b.get('name', '')} | {kind_zh.get(b.get('kind'), '')} | "
                 f"{COMMON.get(b.get('common'), '')} | {SEPARABLE.get(b.get('separable'), '')} | "
                 f"{', '.join(b.get('confused_with') or [])} | {n or '—'} |")
    for c, q in CATEGORIES.items():
        es = [e for e in entries if category(e) == c]
        L += ["", f"## {c}：{q}", ""]
        if not es:
            L.append("（還沒有條目）")
            continue
        L += ["| id | 情境 | 行為 | 標籤 | 難度 | 狀態 |", "|---|---|---|---|---|---|"]
        for f in FAMILIES:
            for e in (x for x in es if family(x) == f):
                st = e.get("status", "ready")
                if st == "planned":
                    st = "planned：" + "、".join(map(str, e.get("needs") or []))
                lv = "三段" if has_levels(e) else ""
                L.append(f"| `{e['id']}` | {e.get('title', '')} | {', '.join(e.get('behaviors') or [])} | "
                         f"{', '.join(e.get('tags') or [])} | {lv} | {st} |")
    return "\n".join(L) + "\n"


# ---- run ---------------------------------------------------------------------------------------------
def _rows_dir(run_dir) -> Path:
    return Path(run_dir) / "rows"


def load_manifest(run_dir) -> dict:
    p = Path(run_dir) / "manifest.json"
    if not p.exists():
        raise CatalogError(f"{p} not found (run `dms-sim catalog expand` first)")
    return json.loads(p.read_text(encoding="utf-8"))


def _run_row(r: dict, run_dir: str) -> dict:
    from .behavior_card import write_card
    from .export import export_scenario
    d = _rows_dir(run_dir) / r["row_id"]
    sim, st, truth = export_scenario(r["scenario"], r["camera"], "frame_yaml", d, seed=r["seed"], fps=r["fps"],
                                     include_face=True, variant=r["variant"],
                                     emulator_overrides=r["emulator_overrides"], emulator_seed=r["emulator_seed"],
                                     full=True)
    card = write_card(sim, d, row=r)
    info = {"row_id": r["row_id"], "frames_dir": str(st["path"]), "truth": str(truth), "frames": st["frames"],
            "card": str(card)}
    (d / "row.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8", newline="\n")
    return info


def run(run_dir, entries: list[str] | None = None, jobs: int = 1, log=print) -> list[dict]:
    """Export every manifest row under <run>/rows/<row_id>/: frame_yaml directory, pipeline / truth / full JSONL and
    the behaviour card (card.md, card.json); then write rows.tsv (write_row_list). jobs: parallel processes."""
    run_dir = Path(run_dir).resolve()
    sel = [x.upper() for x in entries or []]
    rows = [r for r in load_manifest(run_dir)["rows"]
            if not sel or "ALL" in sel or r["entry"] in sel or r["category"] in sel or r["family"] in sel
            or any(r["entry"].startswith(x + "-") for x in sel)]
    done = []
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(jobs) as ex:
            for info in ex.map(_run_row, rows, [str(run_dir)] * len(rows)):
                done.append(info)
                log(f"{info['row_id']}: {info['frames']} frames")
    else:
        for r in rows:
            info = _run_row(r, str(run_dir))
            done.append(info)
            log(f"{info['row_id']}: {info['frames']} frames")
    if entries:   # partial run: list every row of this manifest exported so far (not rows of an earlier one)
        ids = {r["row_id"] for r in load_manifest(run_dir)["rows"]}
        done = [json.loads(p.read_text()) for p in sorted(_rows_dir(run_dir).glob("*/row.json"))
                if p.parent.name in ids]
    write_row_list(run_dir, done)
    return done


def write_row_list(run_dir: Path, done: list[dict]) -> None:
    """<run>/rows.tsv: row_id, labels dir, truth JSONL, behaviour card, fps (one line per exported row)."""
    man = {r["row_id"]: r for r in load_manifest(run_dir)["rows"]}
    with open(run_dir / "rows.tsv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("row_id\tlabels_dir\ttruth\tcard\tfps\n")
        for i in done:
            fh.write(f"{i['row_id']}\t{i['frames_dir']}/labels\t{i['truth']}\t{i['card']}\t"
                     f"{man[i['row_id']]['fps']:g}\n")

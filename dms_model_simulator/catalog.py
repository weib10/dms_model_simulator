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
CATEGORIES_EN = {   # the same in English (the gallery's English page): "name: what it covers"
    "POS": "Behaviour happens: distraction, eyes closed, yawning, phone call, smoking, no seat belt",
    "NEG": "Looks alike but is not: mirror check, drinking, hand by the face, normal blinking",
    "BND": "Duration sweeps: one behaviour from short to long, side by side",
    "CLR": "Stop and repeat: the behaviour stops and comes back after a while",
    "ROB": "Worse detection: missed boxes, dropped frames, latency, more jitter",
    "NAT": "Long everyday driving: long randomly generated drives",
    "REG": "Regression: clips that once caused trouble",
}
FAMILIES_EN = {
    "DIS": "Eyes or head off the road",
    "DRW": "Eyes closed, drowsiness",
    "YWN": "Yawning",
    "PHN": "Phone",
    "SMK": "Smoking",
    "SBT": "Seat belt",
    "ALL": "Mixed or everyday",
}
KINDS = {          # a category in a few plain words (catalog/REVIEW.md, for readers who are not engineers): zh, en
    "POS": ("行為有發生", "Behaviour happens"),
    "NEG": ("像但不是", "Looks like it, is not"),
    "BND": ("門檻邊界掃描", "Sweep across the threshold"),
    "CLR": ("停止後再發生", "Stops, then happens again"),
    "ROB": ("模型輸出有錯時", "With model errors"),
    "NAT": ("長時間自然駕駛", "Long natural driving"),
    "REG": ("之前出過問題", "Once caused trouble"),
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
ENTRY_KEYS = {"id", "number", "title", "title_en", "short", "short_en", "brief", "brief_en", "summary", "summary_en",
              "status", "needs", "source", "tags", "behaviors", "standards", "params", "template", "from_scenario",
              "generator", "variation", "fps", "_path"}
VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
# short name and level table of an entry (the gallery's lists, catalog/REVIEW.md): limits in characters per language
SHORT_MAX = {"zh": 14, "en": 45}
BRIEF_ALL = "all"                                # the brief key for what is the same at every level / has no levels
BRIEF_MAX = {"level": {"zh": 36, "en": 80}, BRIEF_ALL: {"zh": 50, "en": 110}}


class CatalogError(ValueError):
    pass


# ---- behaviour list (catalog/behaviors.yaml) ------------------------------------------------------------
BEHAVIOR_KINDS = ("violation", "normal", "condition")
COMMON = {"often": "常見", "sometimes": "偶爾", "rare": "少見"}
SEPARABLE = {"easy": "好分", "medium": "普通", "hard": "難分"}
BEHAVIOR_ID_RE = re.compile(r"^[VNC][0-9]{2}$")
EXPECT_KEYS = {"label", "min_s", "visible", "max_s", "never"}


def load_behaviors(path=None) -> dict:
    """{id: behaviour} from catalog/behaviors.yaml (V = violation, N = normal, C = condition)."""
    return dict(load_yaml(path or CATALOG_DIR / "behaviors.yaml").get("behaviors") or {})


def is_core(b: dict) -> bool:
    """Must-test behaviour (must_test: true): it must never go wrong (tests/test_behaviors.py). Independent of the
    common / separable estimates, so revising an estimate never drops a test."""
    return bool(b.get("must_test"))


def validate_behaviors(behaviors: dict) -> list[str]:
    """Problems of the behaviour list (empty = valid)."""
    errs = []
    for k, b in behaviors.items():
        if not BEHAVIOR_ID_RE.match(str(k)):
            errs.append(f"{k}: id is not V/N/C + two digits")
        for key in ("name", "name_en", "kind", "family", "common", "separable"):
            if key not in b:
                errs.append(f"{k}: missing {key}")
        if "name_en" in b and (not isinstance(b["name_en"], str) or not b["name_en"].strip()):
            errs.append(f"{k}: name_en must be a non-empty string")
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
            errs.append(f"{k}: must_test behaviour needs `expect`")
        if b.get("kind") == "violation" and ex and "label" not in ex:
            errs.append(f"{k}: violation expect needs `label`")
    return errs


# ---- standards (catalog/standards.yaml, optional local overlay catalog/standards.local.yaml) ----------------------
# The public file holds neutral markers only: {standard id: {requirements: {Rn: {coverage}}}}. Names and texts of a
# standard live in the git-ignored overlay next to it; only internal pages (the gallery) merge it in, validate
# only checks it, and nothing that is committed (index, review, list) reads it: load_standards defaults to the
# public file alone.
LANGS = ("zh", "en")
COVERAGE = ("partial", "gap", "out_of_scope")   # entries related | simulator could, no entry yet | simulator cannot
STANDARD_TEXTS = ("title", "summary", "not_covered")
REQUIREMENT_TEXTS = ("text", "gap")
STANDARD_NOTES = ("premise",)     # optional everywhere: our own premise / reading, which the text does not confirm
REQUIREMENT_NOTES = ("note",)
STANDARD_STRINGS = ("name", "source", "checked")   # name: display label (the id when missing)
REQUIREMENT_STRINGS = ("clause",)
LOCAL_STANDARDS = "standards.local.yaml"


def standards_local_path(path=None) -> Path:
    """The local overlay that belongs to a public standards file: standards.local.yaml in the same folder."""
    return (Path(path) if path else CATALOG_DIR / "standards.yaml").with_name(LOCAL_STANDARDS)


def load_standards_overlay(path=None) -> dict | None:
    """{standard id: names and texts} of the local overlay file as written (path default: catalog/
    standards.local.yaml); None when the file does not exist."""
    p = Path(path) if path else standards_local_path()
    return dict(load_yaml(p).get("standards") or {}) if p.is_file() else None


def load_standards(path=None, local=False) -> dict:
    """{standard id: standard} from catalog/standards.yaml ({} when the file is missing). An entry's `standards`
    ({standard id: [requirement ids]}) says which requirements it relates to; related is not compliant.
    local: True = deep-merge the overlay next to the public file (standards.local.yaml) when it exists; a path =
    that overlay file. Default False: the public markers only. The overlay only adds to standards and requirements
    the public file has, and never changes a `coverage` (validate_standards_overlay reports what it ignores)."""
    p = Path(path) if path else CATALOG_DIR / "standards.yaml"
    out = dict(load_yaml(p).get("standards") or {}) if p.is_file() else {}
    ov = load_standards_overlay(standards_local_path(p) if local is True else local) if local else None
    for sid, s in (ov or {}).items():
        if sid not in out or not isinstance(s, dict) or not isinstance(out[sid], dict):
            continue
        st = deep_merge(out[sid], {k: v for k, v in s.items() if k not in ("requirements", "coverage")})
        reqs = dict(st.get("requirements") or {})
        for rid, r in (s.get("requirements") or {}).items():
            if rid in reqs and isinstance(r, dict) and isinstance(reqs[rid], dict):
                reqs[rid] = deep_merge(reqs[rid], {k: v for k, v in r.items() if k != "coverage"})
        out[sid] = {**st, "requirements": reqs}
    return out


def standard_label(sid: str, s: dict | None) -> str:
    """How a standard is shown to people: its `name` (local overlay) when given, else its id."""
    name = (s or {}).get("name") if isinstance(s, dict) else None
    return str(name) if isinstance(name, str) and name.strip() else str(sid)


def _bilingual(d, where: str) -> list[str]:
    if not isinstance(d, dict) or any(not isinstance(d.get(lg), str) or not d[lg].strip() for lg in LANGS):
        return [f"{where} needs a non-empty text per language {list(LANGS)}"]
    return [f"{where}: unknown language {k!r}" for k in sorted(set(d) - set(LANGS), key=str)]


def _texts_errors(d: dict, texts: tuple, strings: tuple, where: str, required: bool) -> list[str]:
    """Problems of the bilingual texts and plain strings of a standard or requirement: checked when present, and
    missing ones reported when required."""
    errs = []
    for k in texts:
        if k in d or required:
            errs += _bilingual(d.get(k), f"{where}: {k}")
    for k in strings:
        if (k in d or required) and (not isinstance(d.get(k), str) or not d[k].strip()):
            errs.append(f"{where}: missing {k}" if k not in d else f"{where}: {k} must be a non-empty string")
    return errs


def entry_standards_errors(value, standards: dict) -> list[str]:
    """Problems of one entry's `standards` field: known standards, known requirements, no repeats."""
    if not isinstance(value, dict) or not value:
        return ["standards must be a mapping {standard: [requirement ids]} (catalog/standards.yaml)"]
    errs = []
    for sid, reqs in value.items():
        if sid not in standards:
            errs.append(f"standards: {sid!r} is not in catalog/standards.yaml")
            continue
        if not isinstance(reqs, list):
            errs.append(f"standards: {sid} needs a list of requirement ids ([] = related to its topic only)")
            continue
        known = standards[sid].get("requirements") or {}
        errs += [f"standards: {sid} has no requirement {r!r}" for r in reqs if r not in known]
        errs += [f"standards: {sid} lists {r!r} twice" for r in sorted({r for r in reqs if reqs.count(r) > 1}, key=str)]
    return errs


def validate_standards(standards: dict, entries: list[dict] | None = None) -> list[str]:
    """Problems of catalog/standards.yaml (empty = valid): every standard has requirements, each with a valid
    `coverage`. Names and texts (name, title, summary, not_covered, premise, source, checked; clause, text, gap,
    note) are optional
    (the local overlay brings them) but checked when present. With the catalog's entries, also the coverage: a
    `partial` requirement is referenced by at least one entry, a `gap` / `out_of_scope` one by none."""
    errs = []
    for sid, s in standards.items():
        if not isinstance(s, dict):
            errs.append(f"{sid}: not a mapping")
            continue
        errs += _texts_errors(s, STANDARD_TEXTS + STANDARD_NOTES, STANDARD_STRINGS, str(sid), required=False)
        reqs = s.get("requirements")
        if not isinstance(reqs, dict) or not reqs:
            errs.append(f"{sid}: missing requirements")
            continue
        for rid, r in reqs.items():
            if not isinstance(r, dict):
                errs.append(f"{sid} {rid}: not a mapping")
                continue
            errs += _texts_errors(r, REQUIREMENT_TEXTS + REQUIREMENT_NOTES, REQUIREMENT_STRINGS, f"{sid} {rid}",
                                  required=False)
            if r.get("coverage") not in COVERAGE:
                errs.append(f"{sid} {rid}: coverage must be one of {list(COVERAGE)}")
        if entries is None:
            continue
        refs = {rid: [str(e.get("id")) for e in entries if rid in ((e.get("standards") or {}).get(sid) or [])]
                for rid in reqs}
        for rid, r in reqs.items():
            cov = (r or {}).get("coverage") if isinstance(r, dict) else None
            if cov == "partial" and not refs[rid]:
                errs.append(f"{sid} {rid}: coverage partial but no entry lists it")
            elif cov in ("gap", "out_of_scope") and refs[rid]:
                errs.append(f"{sid} {rid}: coverage {cov} but listed by {', '.join(refs[rid])}")
    return errs


def validate_standards_overlay(overlay: dict, standards: dict) -> list[str]:
    """Problems of the local overlay (standards.local.yaml as written) against the public registry (empty = valid).
    It names only standards and requirements the public file has, holds no `coverage` (the public file is its only
    source), and is complete for every standard it names: name, title, summary, not_covered, source, checked, and
    clause, text, gap for every requirement of that standard (an internal page is never half described). `premise`
    (standard) and `note` (requirement) are optional, bilingual when given."""
    errs = []
    s_keys = {*STANDARD_TEXTS, *STANDARD_NOTES, *STANDARD_STRINGS, "requirements"}
    r_keys = {*REQUIREMENT_TEXTS, *REQUIREMENT_NOTES, *REQUIREMENT_STRINGS}
    for sid, s in (overlay or {}).items():
        if sid not in standards:
            errs.append(f"{sid}: not in standards.yaml")
            continue
        if not isinstance(s, dict):
            errs.append(f"{sid}: not a mapping")
            continue
        errs += [f"{sid}: coverage belongs in standards.yaml only" if k == "coverage" else f"{sid}: unknown key {k!r}"
                 for k in sorted(set(s) - s_keys, key=str)]
        errs += _texts_errors(s, STANDARD_TEXTS, STANDARD_STRINGS, str(sid), required=True)
        errs += _texts_errors(s, STANDARD_NOTES, (), str(sid), required=False)      # optional: premise
        reqs = s.get("requirements")
        if not isinstance(reqs, dict):
            errs.append(f"{sid}: missing requirements")
            continue
        known = (standards[sid] or {}).get("requirements") or {}
        errs += [f"{sid} {rid}: not in standards.yaml" for rid in reqs if rid not in known]
        for rid in known:
            r = reqs.get(rid)
            if r is None:
                errs.append(f"{sid} {rid}: missing (every requirement needs clause, text, gap)")
                continue
            if not isinstance(r, dict):
                errs.append(f"{sid} {rid}: not a mapping")
                continue
            errs += [f"{sid} {rid}: coverage belongs in standards.yaml only" if k == "coverage"
                     else f"{sid} {rid}: unknown key {k!r}" for k in sorted(set(r) - r_keys, key=str)]
            errs += _texts_errors(r, REQUIREMENT_TEXTS, REQUIREMENT_STRINGS, f"{sid} {rid}", required=True)
            errs += _texts_errors(r, REQUIREMENT_NOTES, (), f"{sid} {rid}", required=False)   # optional: note
    return errs


def standard_entries(entries: list[dict], sid: str, rid: str | None = None) -> list[dict]:
    """Entries related to a standard (rid None) or to one of its requirements."""
    return [e for e in entries if sid in (e.get("standards") or {})
            and (rid is None or rid in ((e.get("standards") or {}).get(sid) or []))]


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
    nums = [e.get("number") for e in out if e.get("number") is not None]   # a missing number is validate's job
    dup = sorted({n for n in nums if nums.count(n) > 1}, key=str)
    if dup:
        raise CatalogError(f"duplicate entry numbers: {dup}")
    return sorted(out, key=lambda e: str(e.get("id")))


def category(entry: dict) -> str:
    return str(entry["id"]).split("-")[0]


def family(entry: dict) -> str:
    return str(entry["id"]).split("-")[1]


def entry_no(entry: dict) -> str:
    """The entry's fixed number as shown to people: `#045` ("" when it has none, e.g. an entry known only from a
    manifest). The number never changes and is never reused; a new entry takes the largest + 1."""
    n = entry.get("number")
    return f"#{n:03d}" if isinstance(n, int) and not isinstance(n, bool) else ""


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


def short_brief_errors(entry: dict) -> list[str]:
    """Problems of an entry's short name (`short`, `short_en`: a few words) and level table (`brief`, `brief_en`:
    {level: what differs at that level, all: what is the same at every level / what an entry without levels runs}).
    Limits in SHORT_MAX / BRIEF_MAX (characters); uniqueness across the catalog is validate_catalog's."""
    errs = []
    for k, lg in (("short", "zh"), ("short_en", "en")):
        v = entry.get(k)
        if not isinstance(v, str) or not v.strip():
            errs.append(f"missing {k} (the scenario in a few words)" if k not in entry else f"{k} must be a non-empty string")
        elif len(v) > SHORT_MAX[lg]:
            errs.append(f"{k} has {len(v)} characters (at most {SHORT_MAX[lg]})")
    try:
        want = set(LEVELS) if has_levels(entry) else {BRIEF_ALL}
    except CatalogError:
        return errs                                   # a broken levels spec is reported by validate
    keys = {}
    for k, lg in (("brief", "zh"), ("brief_en", "en")):
        b = entry.get(k)
        if not isinstance(b, dict) or not b:
            errs.append(f"missing {k} ({{small, medium, large[, all]}} with levels, else {{all}})" if k not in entry
                        else f"{k} must be a mapping {{level: text}}")
            continue
        keys[k] = set(b)
        ok = set(b) - {BRIEF_ALL} == want - {BRIEF_ALL} and (BRIEF_ALL in b or want == set(LEVELS))
        if not ok:
            errs.append(f"{k} keys {sorted(b, key=str)}: need {', '.join(LEVELS)} (and optionally {BRIEF_ALL})"
                        if want == set(LEVELS) else f"{k} keys {sorted(b, key=str)}: an entry without levels needs "
                                                    f"exactly {BRIEF_ALL}")
        for lv, v in b.items():
            lim = BRIEF_MAX[BRIEF_ALL if lv == BRIEF_ALL else "level"][lg]
            if not isinstance(v, str) or not v.strip():
                errs.append(f"{k}.{lv} must be a non-empty string")
            elif len(v) > lim:
                errs.append(f"{k}.{lv} has {len(v)} characters (at most {lim})")
    if len(keys) == 2 and keys["brief"] != keys["brief_en"]:
        errs.append(f"brief and brief_en have different keys ({sorted(keys['brief'])} vs {sorted(keys['brief_en'])})")
    return errs


def validate_catalog(entries: list[dict]) -> list[str]:
    """Problems across the whole catalog (empty = valid): `short` and `short_en` are unique (ids and numbers are
    checked by load_entries)."""
    errs = []
    for k in ("short", "short_en"):
        seen: dict[str, str] = {}
        for e in entries:
            v = e.get(k)
            if not isinstance(v, str) or not v.strip():
                continue
            key = v.strip()
            if key in seen:
                errs.append(f"{k} {key!r} is used by both {seen[key]} and {e.get('id')}")
            else:
                seen[key] = str(e.get("id"))
    return errs


def validate(entry: dict, profiles: dict, cfgs: dict | None = None, standards: dict | None = None) -> list[str]:
    """Problems of one entry (empty = valid). Builds every grid-mode scenario and checks segment keys, targets,
    hand / phone / cigarette states and frame-aligned durations. standards: catalog/standards.yaml's content
    (default: loaded) for the entry's `standards` field; the coverage across entries is validate_standards'."""
    errs: list[str] = []
    eid = entry.get("id", "?")
    try:
        if not ID_RE.match(str(eid)):
            errs.append(f"id {eid!r} is not <POS|NEG|BND|CLR|ROB|NAT|REG>-<DIS|DRW|YWN|PHN|SMK|SBT|ALL>-<TOPIC>")
        for k in ("title", "status"):
            if k not in entry:
                errs.append(f"missing {k}")
        if "number" not in entry:
            errs.append("missing number (the entry's fixed number; a new entry takes the largest + 1)")
        elif not isinstance(entry["number"], int) or isinstance(entry["number"], bool) or entry["number"] < 1:
            errs.append(f"number must be a whole number >= 1, got {entry['number']!r}")
        errs += [f"unknown key {k!r}" for k in sorted(set(entry) - ENTRY_KEYS, key=str)]
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
        if "standards" in entry:
            errs += entry_standards_errors(entry["standards"], load_standards() if standards is None else standards)
        for k in ("title_en", "summary_en"):         # English of title / summary (the gallery's English page)
            if k in entry and (not isinstance(entry[k], str) or not entry[k].strip()):
                errs.append(f"{k} must be a non-empty string")
        if "title_en" not in entry:
            errs.append("missing title_en (the title in English)")
        if entry.get("summary") and "summary_en" not in entry:
            errs.append("missing summary_en (the summary in English)")
        if "summary_en" in entry and not entry.get("summary"):
            errs.append("summary_en without summary")
        errs += short_brief_errors(entry)
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


def _scripted_yawn_overlaps(sc: dict, beh: dict) -> list[str]:
    """A scripted yawn (yawn: {start: true}) that starts while the previous one still runs restarts it and the mouth
    jumps; reject that."""
    errs, t, busy_until = [], Fraction(0), None
    for i, s in enumerate(sc.get("segments") or []):
        y = s.get("yawn") if isinstance(s.get("yawn"), dict) else {}
        d = s.get("duration")
        if y.get("start"):
            if busy_until is not None and t < busy_until:
                errs.append(f"segment {i}: scripted yawn starts while the previous one still runs "
                            f"(until {float(busy_until):g} s)")
            yd = y.get("duration_s", beh["yawn"]["duration_s"])
            if isinstance(yd, (int, float)) and not isinstance(yd, bool):
                busy_until = t + Fraction(str(yd))
        if isinstance(d, (int, float)) and not isinstance(d, bool):
            t += Fraction(str(d))
    return errs


def _check_scenario(sc: dict, cfgs: dict, fps: float, params: dict) -> list[str]:
    errs = [f"behavior override {p!r} is not in configs/behavior.yaml"
            for p in _unknown_paths(sc.get("behavior"), cfgs["behavior"])]
    errs += [f"cabin override {p!r} is not in configs/cabin.yaml"
             for p in _unknown_paths({k: v for k, v in (sc.get("cabin") or {}).items() if k not in ("targets", "hands")},
                                     cfgs["cabin"])]
    cab = deep_merge(cfgs["cabin"], sc.get("cabin"))
    beh = deep_merge(cfgs["behavior"], sc.get("behavior"))   # segment defaults, as BehaviorGenerator applies them
    errs += _scripted_yawn_overlaps(sc, beh)
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
            need = ((cab.get("held_objects") or {}).get(held) or {}).get("needs_hand")
            if need and s.get("hand", beh["hand"]) != need:
                errs.append(f"segment {i}: held {held!r} needs hand: {need}")
        for key in ("yawn", "closure", "blink"):      # event overrides: keys must exist in configs/behavior.yaml
            if isinstance(s.get(key), dict):
                errs += [f"segment {i}: {key}.{q} is not in configs/behavior.yaml {key}"
                         for q in _unknown_paths(s[key], cfgs["behavior"][key])]
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


# ---- review table (catalog/REVIEW.md, REVIEW.en.md, REVIEW.csv, REVIEW.en.csv) -------------------------------
REVIEW_FILES = {"zh": ("REVIEW.md", "REVIEW.csv"), "en": ("REVIEW.en.md", "REVIEW.en.csv")}
REVIEW_TEXT = {
    "zh": {"title": "情境審閱表",
           "cols": ("編號", "短名", "類別", "行為", *LEVELS, "每級相同／不分級", "標記", "ID", "確認", "備註"),
           "planned": "（未完成）", "sep": "；",      # behaviour names contain 「、」 themselves
           "intro": ("每一列是目錄裡的一條情境，用固定的編號找；目錄的 {n} 條全在這一張表，照編號排。",
                     "small、medium、large 是同一條情境的三個難度：從 small 到 large 越來越接近告警門檻、越難分辨"
                     "（定義見 `catalog/README.md` 開頭的「難度」）。這三欄只寫各級不同的地方，各級相同的、或不分級的情境，寫在「每級相同／不分級」。",
                     "「標記」是情境跟外部法規哪幾項要求有關的代號（只標相關、不表示符合），說明在 `catalog/README.md`「標準」一節。"
                     "`STD-A: R8, R20` 是對到這幾項要求；只寫 `STD-A` 是跟那份法規管的主題有關，但沒有對到特定一項。"
                     "短名後面有「（未完成）」的，模擬器還做不到，只有大綱。",
                     "「確認」「備註」兩欄留給審閱的人填。",
                     "由 `dms-sim catalog review` 產生，不要手改。")},
    "en": {"title": "Scenario review table",
           "cols": ("No.", "Short name", "Kind", "Behaviours", *LEVELS, "Same at every level, or no levels", "Marker",
                    "ID", "OK?", "Comment"),
           "planned": " (planned)", "sep": "; ",
           "intro": ("Each row is one scenario of the catalog, found by its fixed number; all {n} are in this one "
                     "table, by number.",
                     "small, medium and large are three difficulty levels of the same scenario: from small to large "
                     "it gets closer to the alert threshold and harder to tell apart (see `catalog/README.md`). These "
                     "three columns only say what differs between levels; what is the same at every level, or what a "
                     "scenario without levels runs, is in the next column.",
                     "The marker says which requirements of an external regulation a scenario relates to (related, "
                     "never compliant), explained in `catalog/README.md`, section on standards. `STD-A: R8, R20` points "
                     "to those requirements; a bare `STD-A` means related to the topic that regulation covers, without "
                     "pointing to a specific requirement. \"(planned)\" after a "
                     "short name: the simulator cannot run it yet; only its outline exists.",
                     "The last two columns are for the reviewer.",
                     "Generated by `dms-sim catalog review`; do not edit by hand.")},
}


def _review_marker(entry: dict, standards: dict) -> str:
    """`STD-A: R8, R20` (`STD-A` for an empty list): always the standard's public id, never a name."""
    own = entry.get("standards") or {}
    sids = [s for s in standards if s in own] + [s for s in own if s not in standards]
    return "; ".join(f"{s}: {', '.join(map(str, own[s]))}" if own[s] else str(s) for s in sids)


def review_rows(entries: list[dict], behaviors: dict, standards: dict, lang: str = "zh") -> list[list[str]]:
    """The review table as plain cells (header first), one row per entry by number. standards: the public registry
    (load_standards() without the overlay; only its ids are used)."""
    if lang not in REVIEW_TEXT:
        raise CatalogError(f"lang must be one of {list(REVIEW_TEXT)}, got {lang!r}")
    T, en = REVIEW_TEXT[lang], lang == "en"
    out = [list(T["cols"])]
    for e in sorted(entries, key=lambda x: (not isinstance(x.get("number"), int), x.get("number") or 0, str(x["id"]))):
        short = str(e.get("short_en" if en else "short") or e.get("short") or "")
        if e.get("status", "ready") == "planned":
            short += T["planned"]
        names = [(b, behaviors.get(b) or {}) for b in e.get("behaviors") or []]
        beh = T["sep"].join(f"{b} {(d.get('name_en') if en else None) or d.get('name', '')}".strip() for b, d in names)
        brief = e.get("brief_en" if en else "brief") or e.get("brief") or {}
        brief = brief if isinstance(brief, dict) else {}
        out.append([entry_no(e), short, KINDS.get(category(e), ("", ""))[1 if en else 0], beh,
                    *(str(brief.get(lv, "")) for lv in LEVELS), str(brief.get(BRIEF_ALL, "")),
                    _review_marker(e, standards), str(e["id"]), "", ""])
    return out


def review_markdown(entries: list[dict], behaviors: dict, standards: dict, lang: str = "zh") -> str:
    """catalog/REVIEW.md (REVIEW.en.md): a short intro, then every entry in one table (review_rows)."""
    rows = review_rows(entries, behaviors, standards, lang)        # raises on an unknown language
    T = REVIEW_TEXT[lang]

    def cell(s) -> str:
        return " ".join(str(s).split()).replace("|", "\\|")       # one line per row; | would end the cell

    id_col = T["cols"].index("ID")
    L = [f"# {T['title']}", ""]
    for x in T["intro"]:                                          # a blank line each: separate paragraphs
        L += [x.format(n=len(rows) - 1), ""]
    L.append("| " + " | ".join(rows[0]) + " |")
    L.append("|" + "---|" * len(rows[0]))
    for r in rows[1:]:
        L.append("| " + " | ".join(f"`{cell(c)}`" if i == id_col else cell(c) for i, c in enumerate(r)) + " |")
    return "\n".join(L) + "\n"


def review_csv(entries: list[dict], behaviors: dict, standards: dict, lang: str = "zh") -> str:
    """catalog/REVIEW.csv (REVIEW.en.csv): the same rows as the markdown table (written with a BOM for Excel)."""
    import csv
    import io
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(review_rows(entries, behaviors, standards, lang))
    return buf.getvalue()


def write_review(out_dir=None, entries: list[dict] | None = None, behaviors: dict | None = None,
                 standards: dict | None = None) -> list[Path]:
    """Write the four review files into out_dir (default catalog/). standards default: the public registry only;
    these files are committed, so the local overlay is never read here."""
    out = Path(out_dir or CATALOG_DIR)
    out.mkdir(parents=True, exist_ok=True)
    entries = load_entries() if entries is None else entries
    behaviors = load_behaviors() if behaviors is None else behaviors
    standards = load_standards() if standards is None else standards
    paths = []
    for lg, (md, csv_name) in REVIEW_FILES.items():
        p = out / md
        p.write_text(review_markdown(entries, behaviors, standards, lg), encoding="utf-8", newline="\n")
        q = out / csv_name
        q.write_text(review_csv(entries, behaviors, standards, lg), encoding="utf-8-sig", newline="")
        paths += [p, q]
    return paths


# ---- index -------------------------------------------------------------------------------------------
def index_markdown(entries: list[dict]) -> str:
    """catalog/INDEX.md: entries grouped by category, then family."""
    L = ["# 情境目錄索引", "",
         "由 `dms-sim catalog index` 產生，不要手改。id = `<類別>-<行為家族>-<主題>`，說明見 [README.md](README.md)。"
         "編號是每條固定的號碼（條目的 `number`），跟總覽頁一致。", ""]
    ready = sum(e.get("status") == "ready" for e in entries)
    L += [f"共 {len(entries)} 條：ready {ready}、planned {len(entries) - ready}。", ""]
    L += ["| 類別 | 內容 | 條數 |", "|---|---|---|"]
    for c, q in CATEGORIES.items():
        L.append(f"| {c} | {q} | {sum(category(e) == c for e in entries)} |")
    beh = load_behaviors()
    kind_zh = {"violation": "違規", "normal": "正常", "condition": "外觀條件"}
    L += ["", "## 行為清單", "",
          "編號定義在 [behaviors.yaml](behaviors.yaml)。常見度、好分度是估計值；★ = 必測（must_test），一定不能出問題（有測試把關）。", "",
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
        L += ["| 編號 | id | 情境 | 行為 | 標籤 | 難度 | 狀態 |", "|---|---|---|---|---|---|---|"]
        for f in FAMILIES:
            for e in (x for x in es if family(x) == f):
                st = e.get("status", "ready")
                if st == "planned":
                    st = "planned：" + "、".join(map(str, e.get("needs") or []))
                lv = "三段" if has_levels(e) else ""
                L.append(f"| {entry_no(e)} | `{e['id']}` | {e.get('title', '')} | {', '.join(e.get('behaviors') or [])} | "
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


def _layout_info(layout) -> dict | None:
    """{layout: absolute path, layout_sha256} of a layout YAML given to `run` (None = configs/export.yaml's)."""
    if layout is None:
        return None
    p = Path(layout).resolve()
    if not p.is_file():
        raise CatalogError(f"layout {p} not found")
    return {"layout": str(p), "layout_sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def _row_pictures(sim, d: Path, row_id: str, pictures: dict) -> dict:
    """Write the pictures asked for into the row directory and remove the others (left by an earlier run, they
    would show an older output). timeline.png: viz.plot_timeline (truth behaviour, eye openness, injected model
    errors over time). preview.gif: the camera view of every `every`-th frame at `scale` (truth boxes green, model
    boxes red). Returns {"timeline": path, "gif": path} of the files written."""
    from .repro import PICTURE_FILES
    want = [k for k in PICTURE_FILES if pictures.get(k)]
    for k, name in PICTURE_FILES.items():
        if k not in want:
            (d / name).unlink(missing_ok=True)
    if not want:
        return {}
    from . import viz                                   # matplotlib: only when a picture is asked for
    out = {k: str(d / PICTURE_FILES[k]) for k in want}
    if "timeline" in out:
        viz.plt.close(viz.plot_timeline(sim.records, out["timeline"], title=row_id))
    if "gif" in out:
        every = pictures["every"]
        frames = [viz.draw_camera_view(rec, sim.camera, sim.cabin, pictures["scale"]).quantize(
            method=viz.Image.Quantize.FASTOCTREE) for rec in sim.records[::every]]   # 1 byte / pixel while they wait
        viz.save_gif(frames, out["gif"], sim.behavior.fps / every)
    return out


def _run_row(r: dict, run_dir: str, layout: dict | None = None, pipeline: dict | None = None,
             pictures: dict | None = None) -> dict:
    from .behavior_card import write_card
    from .events import road_box
    from .export import export_scenario, load_config
    d = _rows_dir(run_dir) / r["row_id"]
    pipeline = pipeline or {}
    config = None
    if layout is not None or pipeline.get("labels"):
        config = dict(load_config(None))
        if layout is not None:
            config["layout"] = layout["layout"]
        if pipeline.get("labels"):
            config["simulation"] = {**(config.get("simulation") or {}), "labels": pipeline["labels"]}
    sim, st, truth = export_scenario(r["scenario"], r["camera"], "frame_yaml", d, seed=r["seed"], fps=r["fps"],
                                     config=config, include_face=True, variant=r["variant"],
                                     emulator_overrides=r["emulator_overrides"], emulator_seed=r["emulator_seed"],
                                     full=True, pipeline_face=pipeline.get("include_face"))
    card = write_card(sim, d, row=r)
    box = road_box(sim.cabin, sim.behavior.cfg["events"]["on_road_targets"])
    info = {"row_id": r["row_id"], "frames_dir": str(st["path"]), "truth": str(truth), "frames": st["frames"],
            "card": str(card),
            # what counts as looking at the road in this row (vehicle-frame gaze angles, degrees): calibration a
            # post-processing needs, not an answer
            "road_box": None if box is None else {"yaw_deg": [box[0], box[1]], "pitch_deg": [box[2], box[3]]},
            **(layout or {})}
    pics = _row_pictures(sim, d, r["row_id"], pictures or {})
    if pics:
        info["pictures"] = pics
    (d / "row.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8", newline="\n")
    return info


def run(run_dir, entries: list[str] | None = None, jobs: int = 1, log=print, layout=None,
        include_face: bool | None = None, labels: str | None = None, timeline: bool = False, gif: bool = False,
        gif_every: int = 3, gif_scale: float = 0.4) -> list[dict]:
    """Export every manifest row under <run>/rows/<row_id>/: frame_yaml directory, pipeline / truth / full JSONL and
    the behaviour card (card.md, card.json); then write rows.tsv (write_row_list). jobs: parallel processes.
    layout: frame_yaml layout YAML (class ids, per-class thresholds) instead of configs/export.yaml's; its path and
    sha256 go into every row.json, since the layout changes what a post-processing program sees.
    include_face / labels: the pipeline JSONL's face box and label mode (as for `dms-sim run`; None = the defaults
    of configs/export.yaml). Neither changes the truth or the row ids.
    timeline / gif: also write timeline.png / preview.gif per row (_row_pictures; paths under row.json `pictures`).
    They are drawn from the row's own simulation and change no other file. gif_every: draw every Nth frame;
    gif_scale: size relative to the camera image. The GIF is the slow, large one: at the defaults a 13 s row takes
    a few seconds and 1 MB, a 5-minute row over a minute and 20 MB."""
    run_dir = Path(run_dir).resolve()
    lay = _layout_info(layout)
    pipe = {"include_face": include_face, "labels": labels}
    if int(gif_every) < 1 or not float(gif_scale) > 0:
        raise CatalogError(f"gif_every must be >= 1 and gif_scale > 0, got {gif_every!r} and {gif_scale!r}")
    pics = {"timeline": bool(timeline), "gif": bool(gif), "every": int(gif_every), "scale": float(gif_scale)}
    sel = [x.upper() for x in entries or []]
    rows = [r for r in load_manifest(run_dir)["rows"]
            if not sel or "ALL" in sel or r["entry"] in sel or r["category"] in sel or r["family"] in sel
            or any(r["entry"].startswith(x + "-") for x in sel)]
    done = []
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(jobs) as ex:
            for info in ex.map(_run_row, rows, [str(run_dir)] * len(rows), [lay] * len(rows), [pipe] * len(rows),
                               [pics] * len(rows)):
                done.append(info)
                log(f"{info['row_id']}: {info['frames']} frames")
    else:
        for r in rows:
            info = _run_row(r, str(run_dir), lay, pipe, pics)
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

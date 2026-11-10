"""Behaviour list (catalog/behaviors.yaml) and the must-test guarantee: behaviours marked must_test must never go
wrong. For every must-test behaviour there is a ready entry, and every entry that lists it produces the expected
ground truth on every seed and level (see test_core_entries_produce_the_expected_truth). Every NEG entry also stays
below the violation thresholds at its hardest level."""
import numpy as np
import pytest
from dms_model_simulator import catalog as C
from dms_model_simulator.events import label_events
from dms_model_simulator.simulate import Simulation

BEHAVIORS = C.load_behaviors()
ENTRIES = [e for e in C.load_entries() if e.get("status", "ready") == "ready"]
CORE = [k for k, b in BEHAVIORS.items() if C.is_core(b) and b["kind"] != "condition"]
MAX_S_LABEL = {"off_road_s": "eyes_off_road", "closed_s": "eyes_closed"}


def test_behavior_list_is_valid():
    assert C.validate_behaviors(BEHAVIORS) == []
    assert CORE, "no core behaviours"


def test_every_entry_lists_known_behaviors():
    for e in C.load_entries():
        assert e.get("behaviors"), e["id"]
        assert set(e["behaviors"]) <= set(BEHAVIORS), e["id"]


@pytest.mark.parametrize("bid", CORE)
def test_core_behavior_has_a_ready_entry(bid):
    assert any(bid in (e.get("behaviors") or []) for e in ENTRIES), f"{bid} {BEHAVIORS[bid]['name']}: no ready entry"


def _param_sets(entry, level):
    rng = np.random.default_rng(0)
    return C.expand_params(entry, "boundary", rng, levels=[level] if C.has_levels(entry) else None, max_rows=10_000)


GEN_SEEDS = range(10)          # generated (random) drives are checked on more seeds than they list


def _seeds(entry):
    var = {**C.DEFAULT_VARIATION, **(entry.get("variation") or {})}
    seeds = [int(x) for x in var["seeds"]]
    return sorted(set(seeds) | set(GEN_SEEDS)) if "generator" in entry else seeds


def _labels(entry, params, seed=0, sim_out=None):
    sc = C.build_scenario(entry, params, "core-check", seed=seed)
    sim = Simulation(sc, "dash_right", seed=seed, fps=float(entry.get("fps", C.DEFAULT_FPS)))
    if sim_out is not None:
        sim_out.append(sim)
    return label_events(sim.frames, sim.cabin, sim.behavior.cfg)


def _visible_share(entry, params, seed, cls, label):
    """Best share over the entry's cameras of the frames with `label` set in which the truth box of `cls` exists."""
    var = {**C.DEFAULT_VARIATION, **(entry.get("variation") or {})}
    best = 0.0
    for cam in var["cameras"]:
        sim = Simulation(C.build_scenario(entry, params, "vis", seed=seed), cam, seed=seed,
                         fps=float(entry.get("fps", C.DEFAULT_FPS)))
        truths = sim.truths()
        idx = [i for i, t in enumerate(truths) if t["events"].get(label)]
        if idx:
            seen = sum(any(d["class"] == cls for d in truths[i]["detections"]) for i in idx)
            best = max(best, seen / len(idx))
    return best


def _core_entries():
    out = []
    for e in ENTRIES:
        if any(b in CORE for b in e["behaviors"]):
            out.append(pytest.param(e, id=e["id"]))
    return out


@pytest.mark.parametrize("entry", _core_entries())
def test_core_entries_produce_the_expected_truth(entry):
    """Every seed (generated drives: 10 seeds), every level and parameter set:
    - a must-test violation sets its label (a BND sweep: at least one set, sweeps go below thresholds on purpose);
      in POS / CLR entries it lasts at least its min_s, and in POS entries its object is visible on some camera;
    - the normal behaviours stay inside their bounds."""
    listed = [BEHAVIORS[b] for b in entry["behaviors"]]
    allowed = {b["expect"]["label"] for b in listed if b["kind"] == "violation" and b.get("expect")}
    normals = [b for b in listed if b["kind"] == "normal" and b.get("expect")]
    never = set().union(*[set(b["expect"].get("never") or []) for b in normals]) - allowed
    max_s = {}
    for b in normals:
        for k, v in (b["expect"].get("max_s") or {}).items():
            if MAX_S_LABEL[k] not in allowed:
                max_s[k] = max(max_s.get(k, 0.0), float(v))
    # a generated (random) drive may or may not contain a given violation: only its normal bounds are checked
    must_v = [b for b in listed if b["kind"] == "violation" and C.is_core(b)] if "generator" not in entry else []
    sweep = C.category(entry) == "BND"
    timed = C.category(entry) in ("POS", "CLR")
    fired = {b["expect"]["label"]: False for b in must_v}
    sets = [p for lv in (C.LEVELS if C.has_levels(entry) else [None])
            for p in _param_sets(entry, lv or "medium")]
    for params in sets:
        for seed in _seeds(entry):
            ev = _labels(entry, params, seed)
            for b in must_v:
                lab = b["expect"]["label"]
                hit = any(f[lab] for f in ev)
                fired[lab] |= hit
                assert hit or sweep, f"{entry['id']} {params} seed {seed}: never sets {lab} ({b['name']})"
                if hit and timed:
                    for k, lim in (b["expect"].get("min_s") or {}).items():
                        longest = max(f[k] for f in ev)
                        assert longest >= lim - 1e-6, \
                            f"{entry['id']} {params} seed {seed}: {k} only {longest:.2f} s < {lim} s ({b['name']})"
                if hit and C.category(entry) == "POS" and b["expect"].get("visible") and seed == _seeds(entry)[0]:
                    share = _visible_share(entry, params, seed, b["expect"]["visible"], lab)
                    assert share >= 0.5, (f"{entry['id']} {params}: {b['expect']['visible']} visible in only "
                                          f"{share:.0%} of the {lab} frames on its cameras")
            for lab in never:
                hit = [i for i, f in enumerate(ev) if f[lab]]
                assert not hit, f"{entry['id']} {params} seed {seed}: {lab} set at frame {hit[0]}"
            for k, lim in max_s.items():
                worst = max(f[k] for f in ev)
                assert worst <= lim + 1e-6, f"{entry['id']} {params} seed {seed}: {k} reaches {worst:.2f} s > {lim} s"
    missing = [lab for lab, ok in fired.items() if not ok]
    assert not missing, f"{entry['id']}: no parameter set sets {missing}"


# ---- every NEG entry stays on the normal side of the violation thresholds, at its hardest level -------------
NEG_LIMITS = {"off_road_s": 3.0, "closed_s": 1.0}      # Euro NCAP long distraction 3 s; microsleep ~1 s
VATS_WINDOW_S, VATS_LIMIT_S = 30.0, 10.0                # accumulated off road within any 30 s window
UNDECIDED: dict = {}   # known conflicts waiting for a decision (strict xfail: fails once resolved)
NEG_ENTRIES = [pytest.param(e, id=e["id"], marks=[pytest.mark.xfail(reason=UNDECIDED[e["id"]], strict=True)]
                            if e["id"] in UNDECIDED else []) for e in ENTRIES if C.category(e) == "NEG"]


@pytest.mark.parametrize("entry", NEG_ENTRIES)
def test_neg_entries_stay_below_violation_thresholds_at_large(entry):
    listed = [BEHAVIORS[b] for b in entry["behaviors"]]
    allowed = {b["expect"]["label"] for b in listed if b["kind"] == "violation" and b.get("expect")}
    fps = float(entry.get("fps", C.DEFAULT_FPS))
    win = int(round(VATS_WINDOW_S * fps))
    for params in _param_sets(entry, "large"):
      for seed in _seeds(entry):
        ev = _labels(entry, params, seed)
        for k, lim in NEG_LIMITS.items():
            if MAX_S_LABEL[k] in allowed:
                continue
            worst = max(f[k] for f in ev)
            assert worst < lim, f"{entry['id']} {params} seed {seed}: {k} reaches {worst:.2f} s (threshold {lim} s)"
        if "eyes_off_road" not in allowed:
            off = np.array([f["eyes_off_road"] for f in ev], float)
            acc = np.convolve(off, np.ones(win), "valid").max() / fps if len(off) >= win else off.sum() / fps
            assert acc < VATS_LIMIT_S, f"{entry['id']} {params} seed {seed}: {acc:.2f} s off road within {VATS_WINDOW_S:g} s"
        for lab in ("phone_use", "smoking", "seatbelt_off"):
            if lab not in allowed:
                hit = [i for i, f in enumerate(ev) if f[lab]]
                assert not hit, f"{entry['id']} {params}: {lab} set at frame {hit[0]}"

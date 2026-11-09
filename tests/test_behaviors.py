"""Behaviour list (catalog/behaviors.yaml) and the core guarantee: behaviours that are common and easy to tell apart
must never go wrong. For every core behaviour there is a ready entry, and every entry that lists it produces the
expected ground truth: a violation fires its label even at the `small` level; a normal behaviour stays inside its
bounds even at the `large` level."""
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


def _labels(entry, params):
    sc = C.build_scenario(entry, params, "core-check", seed=0)
    sim = Simulation(sc, "dash_right", seed=0, fps=float(entry.get("fps", C.DEFAULT_FPS)))
    return label_events(sim.frames, sim.cabin, sim.behavior.cfg)


def _core_entries():
    out = []
    for e in ENTRIES:
        if any(b in CORE for b in e["behaviors"]):
            out.append(pytest.param(e, id=e["id"]))
    return out


@pytest.mark.parametrize("entry", _core_entries())
def test_core_entries_produce_the_expected_truth(entry):
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
    core_violations = [b for b in listed if b["kind"] == "violation" and C.is_core(b)] if "generator" not in entry else []
    # violations must fire at their mildest level (with levels: every set of `small`; a sweep without levels:
    # at least one set, since a sweep has values below the threshold on purpose); normal behaviours must hold
    # at their hardest level, every set
    levelled = C.has_levels(entry)
    fired = {b["expect"]["label"]: False for b in core_violations}
    for level, check_v, check_n in (("small", True, not levelled), ("large", False, levelled)):
        if levelled is False and level == "large":
            break                                           # no levels: one pass covers both checks
        for params in _param_sets(entry, level):
            ev = _labels(entry, params)
            if check_v:
                for lab in fired:
                    hit = any(f[lab] for f in ev)
                    fired[lab] |= hit
                    assert hit or not levelled, f"{entry['id']} {params}: never sets {lab}"
            if check_n:
                for lab in never:
                    hit = [i for i, f in enumerate(ev) if f[lab]]
                    assert not hit, f"{entry['id']} {params}: {lab} set at frame {hit[0]}"
                for k, lim in max_s.items():
                    worst = max(f[k] for f in ev)
                    assert worst <= lim + 1e-6, f"{entry['id']} {params}: {k} reaches {worst:.2f} s > {lim} s"
    missing = [lab for lab, ok in fired.items() if not ok]
    assert not missing, f"{entry['id']}: no parameter set sets {missing}"


# ---- every NEG entry stays on the normal side of the violation thresholds, at its hardest level -------------
NEG_LIMITS = {"off_road_s": 3.0, "closed_s": 1.0}      # Euro NCAP long distraction 3 s; microsleep ~1 s
VATS_WINDOW_S, VATS_LIMIT_S = 30.0, 10.0                # accumulated off road within any 30 s window
UNDECIDED = {   # known conflicts waiting for a decision (catalog/behaviors.yaml notes); strict: fails once resolved
    "NEG-SMK-CIG-IN-LAP": "N23: the truth counts a cigarette in the hand as smoking (smoking_cigarette_states has hand)",
}
NEG_ENTRIES = [pytest.param(e, id=e["id"], marks=[pytest.mark.xfail(reason=UNDECIDED[e["id"]], strict=True)]
                            if e["id"] in UNDECIDED else []) for e in ENTRIES if C.category(e) == "NEG"]


@pytest.mark.parametrize("entry", NEG_ENTRIES)
def test_neg_entries_stay_below_violation_thresholds_at_large(entry):
    listed = [BEHAVIORS[b] for b in entry["behaviors"]]
    allowed = {b["expect"]["label"] for b in listed if b["kind"] == "violation" and b.get("expect")}
    fps = float(entry.get("fps", C.DEFAULT_FPS))
    win = int(round(VATS_WINDOW_S * fps))
    for params in _param_sets(entry, "large"):
        ev = _labels(entry, params)
        for k, lim in NEG_LIMITS.items():
            if MAX_S_LABEL[k] in allowed:
                continue
            worst = max(f[k] for f in ev)
            assert worst < lim, f"{entry['id']} {params}: {k} reaches {worst:.2f} s (threshold {lim} s)"
        if "eyes_off_road" not in allowed:
            off = np.array([f["eyes_off_road"] for f in ev], float)
            acc = np.convolve(off, np.ones(win), "valid").max() / fps if len(off) >= win else off.sum() / fps
            assert acc < VATS_LIMIT_S, f"{entry['id']} {params}: {acc:.2f} s off road within {VATS_WINDOW_S:g} s"
        for lab in ("phone_use", "smoking", "seatbelt_off"):
            if lab not in allowed:
                hit = [i for i, f in enumerate(ev) if f[lab]]
                assert not hit, f"{entry['id']} {params}: {lab} set at frame {hit[0]}"

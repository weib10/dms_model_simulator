"""Scenario catalog: entries validate, parameter expansion, profiles, expand -> run (frame_yaml + card per row)."""
import copy
import csv
import json
from pathlib import Path
import numpy as np
import pytest
from dms_model_simulator import catalog as C

PR = {"nominal": {}}


def _entry(**kw):
    e = {"id": "BND-PHN-TEST", "title": "t", "summary": "call of call_s seconds", "status": "ready", "behaviors": ["V08"],
         "params": {"call_s": {"grid": [1.0, 2.0, 3.0]}},
         "template": {"segments": [{"target": "road_ahead", "duration": 2.0},
                                   {"target": "road_ahead", "duration": "${call_s}", "hand": "call", "phone": "call"},
                                   {"target": "road_ahead", "duration": 2.0}]},
         "variation": {"cameras": ["dash_right"], "emulator_seeds": [0]}}
    e.update(kw)
    return e


def _short(**kw):
    """A 1 s entry (0.5 s road + 0.5 s call) for tests that run rows."""
    return _entry(params={"call_s": {"grid": [0.5]}},
                  template={"segments": [{"target": "road_ahead", "duration": 0.5},
                                         {"target": "road_ahead", "duration": "${call_s}", "hand": "call",
                                          "phone": "call"}]}, **kw)


# ---- the repo's catalog (depends on catalog/entries, catalog/INDEX.md, catalog/profiles.yaml) -------------------
def test_repo_entries_valid():
    entries = C.load_entries()
    pr = C.load_profiles()
    bad = {e["id"]: C.validate(e, pr) for e in entries}
    assert not {k: v for k, v in bad.items() if v}
    assert len(entries) >= 30
    assert all(e.get("title") for e in entries)


def test_repo_index_current():
    entries = C.load_entries()
    p = C.CATALOG_DIR / "INDEX.md"
    assert p.exists(), "catalog/INDEX.md is missing: run dms-sim catalog index"
    assert p.read_text(encoding="utf-8") == C.index_markdown(entries), "catalog/INDEX.md is stale: run dms-sim catalog index"


def test_repo_profile_stacks_resolve_to_known_keys():
    from dms_model_simulator.config import load_configs, unknown_paths
    pr, em = C.load_profiles(), load_configs()["emulator"]
    assert not {k: unknown_paths(v, em) for k, v in pr.items() if unknown_paths(v, em)}
    worst = C.resolve_profile("worst", pr)
    assert worst["box_jump"]["enabled"] and worst["pipeline"]["latency_frames"] == 2
    assert C.validate(_entry(variation={"profiles": ["jit2+jmp1"]}), pr) == []
    assert C.validate(_entry(variation={"profiles": ["jit2+nope"]}), pr)


# ---- validation ----------------------------------------------------------------------------------------
def test_validate_catches_mistakes():
    assert C.validate(_entry(), PR) == []
    assert any("not a whole number of frames" in x for x in
               C.validate(_entry(params={"call_s": {"fixed": 1.02}}), PR))
    bad = _entry()
    bad["template"] = dict(bad["template"], behavior={"blinkk": {"rate_per_min": 3}})
    assert any("blinkk" in x for x in C.validate(bad, PR))
    assert any("id" in x for x in C.validate(_entry(id="phone-test"), PR))
    assert C.validate(_entry(status="planned", needs=[]), PR)
    assert C.validate(_entry(status="planned", needs=["a capability"]), PR) == []
    assert any("status" in x for x in C.validate(_entry(status="done"), PR))
    assert any("source" in x for x in C.validate(_entry(source="hearsay"), PR))
    assert any("unknown profile" in x for x in C.validate(_entry(variation={"profiles": ["nope"]}), PR))
    assert any("emulator.yaml" in x for x in C.validate(_entry(), {"nominal": {"pipline": {"drop_frame_prob": 0.1}}}))
    assert any("unknown camera" in x for x in C.validate(_entry(variation={"cameras": ["roof"]}), PR))
    assert any("exactly one of template" in x for x in
               C.validate(_entry(from_scenario="scenarios/drowsy.yaml"), PR))


def test_validate_segments():
    seg = lambda **k: {"target": "road_ahead", "duration": 1.0, **k}
    tpl = lambda *s: _entry(params={}, template={"segments": list(s)})
    assert any("unknown target" in x for x in C.validate(tpl(seg(target="moon")), PR))
    assert any("unknown keys" in x for x in C.validate(tpl(seg(glance=1)), PR))
    assert any("unknown hand state" in x for x in C.validate(tpl(seg(hand="waving")), PR))
    assert any("phone" in x for x in C.validate(tpl(seg(phone="pocket")), PR))
    assert any("cigarette" in x for x in C.validate(tpl(seg(cigarette="ear")), PR))
    assert any("head_pose" in x for x in C.validate(tpl(seg(head_pose=[0, 0])), PR))
    assert any("positive number" in x for x in C.validate(tpl(seg(duration=0)), PR))
    assert any("no segments" in x for x in C.validate(tpl(), PR))


def test_validate_rejects_unknown_entry_keys():
    assert any("unknown key 'colour'" in x for x in C.validate(_entry(colour="red"), PR))
    assert any("exactly one of" in x for x in C.validate(_entry(params={"call_s": {"near": 1.0}}), PR))


def test_validate_keys_and_generator():
    assert any("emulator_seed" in x for x in C.validate(_entry(variation={"emulator_seed": [0]}), PR))
    gen = _entry(params={}, generator={"duration_s": 10, "config": {"phone_episodez": {"rate_per_min": 0}}})
    del gen["template"]
    assert any("phone_episodez" in x for x in C.validate(gen, PR))
    gen = _entry(params={}, generator={"config": {}})
    del gen["template"]
    assert any("duration_s" in x for x in C.validate(gen, PR))
    ok = _entry(params={}, generator={"duration_s": 10})
    del ok["template"]
    assert C.validate(ok, PR) == []


def test_validate_entry_file_location(tmp_path):
    e = _entry(_path=str(tmp_path / "POS" / "BND-PHN-TEST.yaml"))
    assert any("file is in POS/" in x for x in C.validate(e, PR))
    assert C.validate(_entry(_path=str(tmp_path / "BND" / "BND-PHN-TEST.yaml")), PR) == []


def test_validate_hand_burst_misses_profile():
    assert C.validate(_entry(), {"nominal": {"detection": {"hand": {"mean_miss_frames": 5}}}}) == []


def test_validate_held_uses_the_scenario_behavior_defaults():
    seg = lambda **k: {"target": "road_ahead", "duration": 2.0, **k}
    ok = _entry(params={}, template={"behavior": {"hand": "at_mouth"}, "segments": [seg(held="cup")]})
    assert C.validate(ok, PR) == []
    bad = _entry(params={}, template={"behavior": {"held": "mug"}, "segments": [seg(hand="at_mouth")]})
    assert any("mug" in e for e in C.validate(bad, PR))
    bad = _entry(params={}, template={"segments": [seg(held="cup")]})              # hand on the wheel
    assert any("needs hand: at_mouth" in e for e in C.validate(bad, PR))


# ---- parameters and templates ----------------------------------------------------------------------------
def test_param_values_and_expand_params():
    rng = np.random.default_rng(0)
    assert C.param_values(2.5, "grid", rng) == [2.5]
    assert C.param_values({"fixed": [1, 2]}, "boundary", rng) == [[1, 2]]
    assert C.param_values({"choice": ["a", "b"]}, "boundary", rng) == ["a"]
    assert C.param_values({"choice": ["a", "b"]}, "grid", rng) == ["a", "b"]
    assert C.param_values({"uniform": [1.0, 2.0]}, "grid", rng, step=0.05) == [1.0, 1.5, 2.0]
    assert C.param_values({"uniform": [1.0, 2.0]}, "boundary", rng, step=0.05) == [1.5]
    for _ in range(20):                                                  # random draws snap to whole frames
        (v,) = C.param_values({"uniform": [1.0, 2.0]}, "random", rng, step=0.05)
        assert 1.0 <= v <= 2.0 and abs(v * 20 - round(v * 20)) < 1e-6
    assert C.param_values({"grid": [1, 2, 3]}, "boundary", rng) == [1, 2, 3]
    assert C.param_values({"grid": [1, 2, 3]}, "random", rng)[0] in (1, 2, 3)
    with pytest.raises(C.CatalogError):
        C.param_values({"grid": [1], "choice": [2]}, "grid", rng)
    e = _entry(params={"call_s": {"grid": [1.0, 2.0]}, "side": {"choice": [1, -1]}})
    assert len(C.expand_params(e, "boundary", rng)) == 2                 # grid kept, choice -> first value
    assert len(C.expand_params(e, "grid", rng)) == 4
    rnd = C.expand_params(e, "random", rng, n=7)                         # 2 x 2 possible sets: repeats are dropped
    assert 1 <= len(rnd) <= 4 and len({json.dumps(p, sort_keys=True) for p in rnd}) == len(rnd)
    with pytest.raises(C.CatalogError):
        C.expand_params(e, "grid", rng, max_rows=3)
    with pytest.raises(C.CatalogError):
        C.expand_params(e, "sweep", rng)
    assert C.expand_params(_entry(params={}), "grid", rng) == [{}]


def test_substitute_build_scenario_and_segment_times():
    assert C.substitute({"a": "${x}", "b": ["${x}", "n_${x}"]}, {"x": 1.5}) == {"a": 1.5, "b": [1.5, "n_1.5"]}
    with pytest.raises(C.CatalogError):
        C.substitute("${y}", {"x": 1})
    with pytest.raises(C.CatalogError):
        C.substitute("n_${y}", {"x": 1})
    e = _entry()
    sc = C.build_scenario(e, {"call_s": 3.0}, "s")
    assert sc["name"] == "s" and sc["description"].startswith("BND-PHN-TEST: t")
    assert sc["segments"][1]["duration"] == 3.0
    assert C.segment_times(sc) == [(0.0, 2.0), (2.0, 5.0), (5.0, 7.0)]
    assert e["template"]["segments"][1]["duration"] == "${call_s}"       # the entry is not changed
    sc = C.build_scenario({"id": "X", "from_scenario": "scenarios/drowsy.yaml"}, {}, "d")
    assert sc["name"] == "d" and len(sc["segments"]) == 5


def test_select():
    es = [_entry(), _entry(id="POS-DRW-A", tags=["sleepy"]), _entry(id="NEG-PHN-B", status="planned", needs=["x"])]
    ids = lambda sel, **k: [e["id"] for e in C.select(es, sel, **k)]
    assert ids(None) == ["BND-PHN-TEST", "POS-DRW-A", "NEG-PHN-B"]
    assert ids(["PHN"]) == ["BND-PHN-TEST", "NEG-PHN-B"]
    assert ids(["pos"]) == ["POS-DRW-A"]
    assert ids(["BND-PHN"]) == ["BND-PHN-TEST"]
    assert ids(["NEG-PHN-B"]) == ["NEG-PHN-B"]
    assert ids(["all"], status="ready") == ["BND-PHN-TEST", "POS-DRW-A"]
    assert ids(None, tag="sleepy") == ["POS-DRW-A"]


def test_load_entries_rejects_duplicate_ids(tmp_path):
    (tmp_path / "BND").mkdir()
    for n in ("a", "b"):
        (tmp_path / "BND" / f"{n}.yaml").write_text("id: BND-PHN-SAME\ntitle: t\nstatus: ready\n", encoding="utf-8")
    with pytest.raises(C.CatalogError, match="duplicate"):
        C.load_entries(tmp_path)


def test_index_markdown_lists_entries():
    es = [_entry(tags=["phone"]), _entry(id="POS-DRW-A", title="閉眼", status="planned", needs=["睡姿"])]
    md = C.index_markdown(es)
    assert "共 2 條：ready 1、planned 1。" in md
    assert "| `BND-PHN-TEST` | t | V08 | phone |  | ready |" in md and "planned：睡姿" in md


# ---- profiles ------------------------------------------------------------------------------------------
def test_resolve_profile_stacks_and_compose(tmp_path):
    pr = {"a": {"x": {"p": 1, "q": 1}}, "b": {"x": {"p": 2}}}
    assert C.resolve_profile("a+b", pr) == {"x": {"p": 2, "q": 1}}          # later parts win
    assert C.resolve_profile("b+a", pr) == {"x": {"p": 1, "q": 1}}
    for bad in ("a+c", "a+", ""):
        with pytest.raises(C.CatalogError):
            C.resolve_profile(bad, pr)
    p = tmp_path / "profiles.yaml"
    p.write_text("profiles:\n  a: {emulator_overrides: {x: {p: 1}}}\n  b: {emulator_overrides: {y: 2}}\n"
                 "  ab: {compose: 'a+b'}\n", encoding="utf-8")
    assert C.load_profiles(p)["ab"] == {"x": {"p": 1}, "y": 2}
    p.write_text("profiles:\n  a+b: {emulator_overrides: {}}\n", encoding="utf-8")
    with pytest.raises(C.CatalogError):
        C.load_profiles(p)


# ---- expand ------------------------------------------------------------------------------------------
def test_expand_writes_scenarios_and_manifest(tmp_path):
    e = _entry(variation={"cameras": ["dash_right", "steering_column"], "emulator_seeds": [0, 1], "variants": 1})
    pr = {"nominal": {}, "drops": {"pipeline": {"drop_frame_prob": 0.5}}}
    man = C.expand([e, _entry(id="NEG-PHN-PLAN", status="planned", needs=["x"])], tmp_path, "boundary",
                   profiles=pr, profile_names=["nominal", "drops"])
    # 3 call lengths x 2 cameras x (nominal + 1 variant) x 2 emulator seeds x 2 profiles
    assert len(man["rows"]) == 3 * 2 * 2 * 2 * 2 and man["skipped_planned"] == ["NEG-PHN-PLAN"]
    assert man["entries"] == ["BND-PHN-TEST"]
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == man
    r = man["rows"][0]
    assert {"row_id", "entry", "category", "family", "params", "scenario", "camera", "variant", "seed",
            "emulator_seed", "profile", "emulator_overrides", "fps", "title", "summary"} <= set(r)
    assert (r["title"], r["summary"]) == ("t", "call of call_s seconds")
    assert (r["category"], r["family"], r["fps"]) == ("BND", "PHN", 20.0)
    assert {x["variant"] for x in man["rows"]} == {None, 0}
    assert all(x["emulator_overrides"] == pr[x["profile"]] for x in man["rows"])
    assert len({x["row_id"] for x in man["rows"]}) == len(man["rows"])
    assert len(list((tmp_path / "scenarios").glob("*.yaml"))) == 3
    from dms_model_simulator.config import load_scenario
    sc = load_scenario(r["scenario"])
    assert sc["name"] == Path(r["scenario"]).stem and sc["segments"][1]["duration"] == r["params"]["call_s"]


def test_expand_rng_per_entry_and_overrides(tmp_path):
    e1 = _entry(params={"call_s": {"uniform": [0.5, 4.0]}})
    e2 = _entry(id="BND-PHN-OTHER", params={"call_s": {"uniform": [0.5, 4.0]}})
    kw = dict(mode="random", n=3, profiles=PR)
    alone = [r["params"] for r in C.expand([e1], tmp_path / "a", **kw)["rows"]]
    both = [r["params"] for r in C.expand([e2, e1], tmp_path / "b", **kw)["rows"] if r["entry"] == e1["id"]]
    assert alone == both
    with pytest.raises(C.CatalogError):
        C.expand([e1], tmp_path / "c", cameras=["nope"], profiles=PR)
    with pytest.raises(C.CatalogError):
        C.expand([e1], tmp_path / "d", profile_names=["nope"], profiles=PR)
    with pytest.raises(C.CatalogError):
        C.expand([e1], tmp_path / "e", max_rows=2, **kw)
    assert not (tmp_path / "e").exists()
    ov = C.expand([e1], tmp_path / "f", cameras=["steering_column"], emulator_seeds=2, profiles=PR)["rows"]
    assert {r["camera"] for r in ov} == {"steering_column"} and {r["emulator_seed"] for r in ov} == {0, 1}


def test_random_mode_rows_are_unique(tmp_path):
    e = _entry(params={"call_s": {"grid": [1.0, 2.0]}})
    man = C.expand([e], tmp_path, "random", n=10, profiles=PR)
    ids = [r["row_id"] for r in man["rows"]]
    assert len(ids) == len(set(ids))
    assert sorted(r["params"]["call_s"] for r in man["rows"]) == [1.0, 2.0]


def test_reexpand_changed_template_gets_new_row_ids(tmp_path):
    e = _entry(params={"call_s": {"fixed": 3.0}})
    old = C.expand([e], tmp_path, "boundary", profiles=PR)
    e2 = copy.deepcopy(e)
    e2["template"]["segments"][0]["duration"] = 4.0                     # same id, params and seed
    new = C.expand([e2], tmp_path, "boundary", profiles=PR)
    assert {r["row_id"] for r in new["rows"]}.isdisjoint(r["row_id"] for r in old["rows"])
    assert C.expand([e], tmp_path / "again", "boundary", profiles=PR)["rows"][0]["row_id"] == old["rows"][0]["row_id"]


# ---- run ---------------------------------------------------------------------------------------------
def _rows_tsv(run_dir):
    with open(run_dir / "rows.tsv", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def test_run_writes_frames_jsonl_and_card_per_row(tmp_path):
    e = _short(variation={"cameras": ["dash_right"], "emulator_seeds": [0, 1]})
    man = C.expand([e], tmp_path, "boundary", profiles={"nominal": {}, "drops": {"pipeline": {"drop_frame_prob": 1.0}}},
                   profile_names=["nominal", "drops"])
    assert len(man["rows"]) == 4
    logged = []
    done = C.run(tmp_path, log=logged.append)
    assert len(done) == 4 and len(logged) == 4
    by_id = {r["row_id"]: r for r in man["rows"]}
    for d in done:
        row = by_id[d["row_id"]]
        rd = tmp_path / "rows" / d["row_id"]
        assert d["frames"] == 20 and Path(d["frames_dir"]).parent == rd
        assert len(list((Path(d["frames_dir"]) / "labels").glob("*.yaml"))) == 20
        stem = Path(row["scenario"]).stem + "__dash_right"
        for suffix in (".pipeline.jsonl", ".truth.jsonl", ".jsonl"):
            assert (rd / f"{stem}{suffix}").exists(), suffix
        assert Path(d["card"]) == rd / "card.md" and (rd / "card.json").exists()
        card = json.loads((rd / "card.json").read_text(encoding="utf-8"))
        assert (card["row_id"], card["entry"], card["profile"]) == (d["row_id"], "BND-PHN-TEST", row["profile"])
        assert card["params"] == {"call_s": 0.5} and card["title"] == "t" and card["frames"] == 20
        assert card["emulator_seed"] == row["emulator_seed"]
        assert [(s["f0"], s["f1"]) for s in card["timeline"]] == [(0, 9), (10, 19)]
        if row["profile"] == "drops":
            assert card["errors"]["dropped"] == [[0, 19]]
        assert d["row_id"] in (rd / "card.md").read_text(encoding="utf-8")
        assert json.loads((rd / "row.json").read_text()) == d
    tsv = _rows_tsv(tmp_path)
    assert list(tsv[0]) == ["row_id", "labels_dir", "truth", "card", "fps"]
    assert {t["row_id"] for t in tsv} == set(by_id)
    for t in tsv:
        d = next(x for x in done if x["row_id"] == t["row_id"])
        assert Path(t["labels_dir"]) == Path(d["frames_dir"]) / "labels" and Path(t["labels_dir"]).is_dir()
        assert Path(t["truth"]) == Path(d["truth"]) and Path(t["truth"]).exists()
        assert Path(t["card"]) == Path(d["card"]) and t["fps"] == "20"


def test_partial_run_lists_rows_of_the_current_manifest(tmp_path):
    e1, e2 = _short(), _short(id="POS-PHN-TEST")
    man = C.expand([e1, e2], tmp_path, "boundary", profiles=PR)
    C.run(tmp_path, entries=["POS"], log=lambda *a: None)
    assert [t["row_id"] for t in _rows_tsv(tmp_path)] == [r["row_id"] for r in man["rows"] if r["entry"] == e2["id"]]
    (tmp_path / "rows" / "OLD-ROW").mkdir()                                # a row of an earlier manifest
    (tmp_path / "rows" / "OLD-ROW" / "row.json").write_text(json.dumps(
        {"row_id": "OLD-ROW", "frames_dir": "x", "truth": "y", "frames": 1, "card": "z"}))
    done = C.run(tmp_path, entries=[e1["id"]], log=lambda *a: None)
    assert {d["row_id"] for d in done} == {r["row_id"] for r in man["rows"]}
    assert len(_rows_tsv(tmp_path)) == 2


def test_run_with_a_layout_uses_its_classes_and_records_it(tmp_path):
    import hashlib
    from conftest import MAPPED_LAYOUT
    e = _short(variation={"cameras": ["dash_right"]})
    C.expand([e], tmp_path, "boundary")
    d = C.run(tmp_path, log=lambda *a: None, layout=MAPPED_LAYOUT)[0]
    assert d["layout"] == str(MAPPED_LAYOUT.resolve())
    assert d["layout_sha256"] == hashlib.sha256(MAPPED_LAYOUT.read_bytes()).hexdigest()
    names = {det["class_name"] for p in (Path(d["frames_dir"]) / "labels").glob("*.yaml")
             for det in (C.load_yaml(p).get("detections") or [])}
    assert "face_box" in names and "face" not in names
    with pytest.raises(C.CatalogError, match="not found"):
        C.run(tmp_path, log=lambda *a: None, layout=tmp_path / "missing.yaml")


def test_run_needs_a_manifest(tmp_path):
    with pytest.raises(C.CatalogError, match="catalog expand"):
        C.run(tmp_path)


def test_cli_catalog_expand_and_run(tmp_path, monkeypatch, capsys):
    from dms_model_simulator import cli
    monkeypatch.setattr(C, "load_entries", lambda root=None: [_short()])
    capsys.readouterr()
    cli.main(["catalog", "list"])
    assert "BND-PHN-TEST" in capsys.readouterr().out
    cli.main(["catalog", "validate", "BND"])
    assert "BND-PHN-TEST: ok" in capsys.readouterr().out
    cli.main(["catalog", "expand", "BND", "--out", str(tmp_path), "--profile", "nominal"])
    assert "1 rows from 1 entries" in capsys.readouterr().out
    cli.main(["catalog", "run", str(tmp_path)])
    out = capsys.readouterr().out
    assert "exported 1 rows" in out and "rows.tsv" in out
    (t,) = _rows_tsv(tmp_path)
    assert Path(t["card"]).name == "card.md" and Path(t["card"]).exists()
    cli.main(["catalog", "index", "--out", str(tmp_path / "INDEX.md")])
    assert "BND-PHN-TEST" in (tmp_path / "INDEX.md").read_text(encoding="utf-8")


# ---- difficulty levels, repeat blocks, parameter overrides --------------------------------------------------
def _lv_entry(**kw):
    """call_s and gap_s follow the row's level; side is an ordinary choice."""
    base = dict(params={"call_s": {"levels": {"small": 0.5, "medium": 1.0, "large": 2.0}},
                        "gap_s": {"levels": {"small": 2.0, "medium": 1.0, "large": 0.5}},
                        "side": {"choice": ["a", "b"]}},
                template={"segments": [{"target": "road_ahead", "duration": "${gap_s}"},
                                       {"target": "road_ahead", "duration": "${call_s}", "hand": "call",
                                        "phone": "call"}]})
    return _entry(**{**base, **kw})


def test_levels_are_one_axis_not_a_product():
    rng = np.random.default_rng(0)
    e = _lv_entry()
    assert C.has_levels(e) and not C.has_levels(_entry())
    ps = C.expand_params(e, "boundary", rng)
    assert [(p["level"], p["call_s"], p["gap_s"]) for p in ps] == [("small", 0.5, 2.0), ("medium", 1.0, 1.0),
                                                                   ("large", 2.0, 0.5)]
    assert len(C.expand_params(e, "grid", rng)) == 3 * 2                  # levels x side choice
    assert [p["level"] for p in C.expand_params(e, "boundary", rng, levels=["large"])] == ["large"]
    rnd = C.expand_params(e, "random", rng, n=30, levels=["small", "large"])
    assert {p["level"] for p in rnd} <= {"small", "large"}
    assert all(p["call_s"] == {"small": 0.5, "large": 2.0}[p["level"]] for p in rnd)
    with pytest.raises(C.CatalogError):
        C.expand_params(e, "boundary", rng, levels=["huge"])
    assert "level" not in C.expand_params(_entry(), "boundary", rng)[0]   # entries without levels: no level key


def test_param_overrides_fix_values_levels_or_not():
    rng = np.random.default_rng(0)
    ps = C.expand_params(_lv_entry(), "boundary", rng, overrides={"call_s": 0.75, "nope": 1})
    assert [p["call_s"] for p in ps] == [0.75] * 3 and [p["gap_s"] for p in ps] == [2.0, 1.0, 0.5]
    ps = C.expand_params(_entry(), "boundary", rng, overrides={"call_s": 1.5})
    assert ps == [{"call_s": 1.5}]


def test_validate_levels_and_reserved_names():
    assert C.validate(_lv_entry(), PR) == []
    bad = _lv_entry(params={"call_s": {"levels": {"small": 0.5, "large": 2.0}}})
    assert any("levels needs exactly" in m for m in C.validate(bad, PR))
    res = _entry(params={"call_s": {"grid": [1.0]}, "level": {"fixed": 1}})
    assert any("reserved" in m for m in C.validate(res, PR))
    off = _lv_entry(params={"call_s": {"levels": {"small": 0.5, "medium": 1.03, "large": 2.0}},
                            "gap_s": {"fixed": 1.0}, "side": {"fixed": "a"}})
    assert any("whole number of frames" in m for m in C.validate(off, PR))   # every level is checked


def test_repeat_blocks_expand_and_validate():
    segs = [{"target": "road_ahead", "duration": 1.0},
            {"repeat": 2, "segments": [{"target": "passenger", "duration": 0.5},
                                       {"repeat": 2, "segments": [{"target": "road_ahead", "duration": 0.25}]}]}]
    out = C.expand_repeats(segs)
    assert [s["target"] for s in out] == ["road_ahead"] + ["passenger", "road_ahead", "road_ahead"] * 2
    assert C.expand_repeats([{"repeat": 0, "segments": [{"target": "x", "duration": 1}]}]) == []
    for bad in ({"repeat": 1.5, "segments": []}, {"repeat": -1, "segments": []}, {"repeat": 2},
                {"repeat": 2, "segments": [], "target": "x"}):
        with pytest.raises(C.CatalogError):
            C.expand_repeats([bad])
    e = _entry(params={"n": {"levels": {"small": 1, "medium": 2, "large": 3}}},
               template={"segments": [{"target": "road_ahead", "duration": 1.0},
                                      {"repeat": "${n}", "segments": [{"target": "passenger", "duration": 0.5},
                                                                      {"target": "road_ahead", "duration": 1.0}]}]})
    assert C.validate(e, PR) == []
    sc = C.build_scenario(e, {"n": 3, "level": "large"}, "s")
    assert len(sc["segments"]) == 1 + 2 * 3 and e["template"]["segments"][1]["repeat"] == "${n}"


def test_expand_rows_carry_level_and_card_shows_it(tmp_path):
    man = C.expand([_lv_entry()], tmp_path, profiles=PR, levels=["small", "large"], overrides={"side": "b"})
    assert {r["level"] for r in man["rows"]} == {"small", "large"}
    assert all(f"__{r['level']}__" in r["row_id"] and r["params"]["side"] == "b" for r in man["rows"])
    assert man["levels"] == ["small", "large"] and man["overrides"] == {"side": "b"}
    from dms_model_simulator.behavior_card import markdown
    c = {"row_id": "x", "scenario": "s", "entry": "E", "title": "t", "summary": "", "params": {"level": "large"},
         "level": "large", "profile": None, "camera": "dash_right", "fps": 20.0, "frames": 1, "duration_s": 0.05,
         "seed": 0, "emulator_seed": 0, "timeline": [],
         "errors": {"frames": 1, "latency_frames": 0, "dropped": [], "missed": {}, "class_flip": {}, "jump": {},
                    "false_positive": {}}}
    assert "- 難度：`large`" in markdown(c)


def test_index_marks_entries_with_levels():
    md = C.index_markdown([_lv_entry(id="NEG-PHN-LV"), _entry()])
    assert "| `NEG-PHN-LV` | t | V08 |  | 三段 | ready |" in md and "| `BND-PHN-TEST` | t | V08 |  |  | ready |" in md


def test_cli_param_overrides(tmp_path, monkeypatch, capsys):
    from dms_model_simulator import cli
    assert cli.parse_param_overrides(["a=1.5", "b=[1, 2]", "c={yaw: 30}", "d=x"]) == \
        {"a": 1.5, "b": [1, 2], "c": {"yaw": 30}, "d": "x"}
    with pytest.raises(SystemExit):
        cli.parse_param_overrides(["novalue"])
    monkeypatch.setattr(C, "load_entries", lambda root=None: [_lv_entry()])
    cli.main(["catalog", "expand", "--out", str(tmp_path / "a"), "--level", "medium", "--param", "call_s=0.25"])
    man = json.loads((tmp_path / "a" / "manifest.json").read_text(encoding="utf-8"))
    assert {r["level"] for r in man["rows"]} == {"medium"} and {r["params"]["call_s"] for r in man["rows"]} == {0.25}
    with pytest.raises(SystemExit):
        cli.main(["catalog", "expand", "--out", str(tmp_path / "b"), "--param", "nope=1"])


def test_validate_event_override_keys_and_scripted_yawn_overlaps():
    seg = lambda **k: {"target": "road_ahead", "duration": 2.0, **k}
    typo = _entry(params={}, template={"segments": [seg(yawn={"strat": True, "peak_opennes": 0.9}),
                                                    seg(closure={"openess": 0.1})]})
    errs = C.validate(typo, PR)
    assert any("yawn.strat" in e for e in errs) and any("closure.openess" in e for e in errs)
    ok = _entry(params={}, template={"segments": [seg(yawn={"start": True, "duration_s": 2.0}),
                                                  seg(yawn={"start": True, "duration_s": 2.0})]})
    assert C.validate(ok, PR) == []                                    # back to back is fine
    overlap = _entry(params={}, template={"segments": [seg(yawn={"start": True, "duration_s": 3.0}),
                                                       seg(yawn={"start": True, "duration_s": 2.0})]})
    assert any("still runs" in e for e in C.validate(overlap, PR))

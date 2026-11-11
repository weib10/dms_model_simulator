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
    e = {"id": "BND-PHN-TEST", "number": 1, "title": "t", "title_en": "t", "short": "t", "short_en": "t",
         "brief": {"all": "b"}, "brief_en": {"all": "b"}, "summary": "call of call_s seconds",
         "summary_en": "call of call_s seconds", "status": "ready", "behaviors": ["V08"],
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
    nums = [e.get("number") for e in entries]
    assert all(isinstance(n, int) and n >= 1 for n in nums) and len(set(nums)) == len(nums)


def test_repo_index_current():
    entries = C.load_entries()
    p = C.CATALOG_DIR / "INDEX.md"
    assert p.exists(), "catalog/INDEX.md is missing: run dms-sim catalog index"
    assert p.read_text(encoding="utf-8") == C.index_markdown(entries), "catalog/INDEX.md is stale: run dms-sim catalog index"


def test_repo_review_current():
    """The four review files are committed: they must match the catalog (line endings aside: git may check them
    out with CRLF). They use the public markers only."""
    entries, beh, std = C.load_entries(), C.load_behaviors(), C.load_standards()
    for lg, (md, csv_name) in C.REVIEW_FILES.items():
        p, q = C.CATALOG_DIR / md, C.CATALOG_DIR / csv_name
        assert p.exists() and q.exists(), "catalog/REVIEW* missing: run dms-sim catalog review"
        assert p.read_text(encoding="utf-8") == C.review_markdown(entries, beh, std, lg), \
            f"catalog/{md} is stale: run dms-sim catalog review"
        raw = q.read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf"), f"catalog/{csv_name} needs a BOM (Excel)"
        assert raw[3:].decode("utf-8").replace("\r\n", "\n") == C.review_csv(entries, beh, std, lg), \
            f"catalog/{csv_name} is stale: run dms-sim catalog review"
    assert C.validate_catalog(entries) == []


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


def test_validate_entry_number():
    assert C.validate(_entry(number=88), PR) == []
    e = _entry()
    del e["number"]
    assert any("missing number" in x for x in C.validate(e, PR))
    for bad in (0, -3, 1.0, "45", True, None):
        assert any("number must be a whole number" in x for x in C.validate(_entry(number=bad), PR)), bad


def test_load_entries_reads_numbers_and_rejects_duplicates(tmp_path):
    (tmp_path / "BND").mkdir()
    for n, num in (("a", 7), ("b", 8)):
        (tmp_path / "BND" / f"{n}.yaml").write_text(f"id: BND-PHN-{n.upper()}\nnumber: {num}\ntitle: t\n", encoding="utf-8")
    assert [e["number"] for e in C.load_entries(tmp_path)] == [7, 8]
    (tmp_path / "BND" / "b.yaml").write_text("id: BND-PHN-B\nnumber: 7\ntitle: t\n", encoding="utf-8")
    with pytest.raises(C.CatalogError, match="duplicate entry numbers"):
        C.load_entries(tmp_path)


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
    es = [_entry(tags=["phone"]), _entry(id="POS-DRW-A", number=45, title="閉眼", status="planned", needs=["睡姿"])]
    md = C.index_markdown(es)
    assert "共 2 條：ready 1、planned 1。" in md
    assert "| #001 | `BND-PHN-TEST` | t | V08 | phone |  | ready |" in md and "| #045 | `POS-DRW-A` | 閉眼 |" in md
    assert "planned：睡姿" in md


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


def _row_files(run_dir, skip=()):
    return {p.relative_to(run_dir).as_posix(): p.read_bytes() for p in sorted(run_dir.rglob("*"))
            if p.is_file() and p.name not in skip}


def test_run_pictures_are_optional_and_change_no_other_file(tmp_path):
    from PIL import Image
    C.expand([_short()], tmp_path, "boundary", profiles=PR)
    (plain,) = C.run(tmp_path, log=lambda *a: None)
    rd = tmp_path / "rows" / plain["row_id"]
    assert "pictures" not in plain and not list(rd.glob("*.png")) and not list(rd.glob("*.gif"))
    before = _row_files(tmp_path)
    (d,) = C.run(tmp_path, log=lambda *a: None, timeline=True, gif=True, gif_every=4, gif_scale=0.25)
    assert d["pictures"] == {"timeline": str(rd / "timeline.png"), "gif": str(rd / "preview.gif")}
    assert json.loads((rd / "row.json").read_text()) == d
    with Image.open(rd / "timeline.png") as im:
        assert im.format == "PNG" and im.width > 400
    with Image.open(rd / "preview.gif") as im:
        assert (im.n_frames, im.size, im.info["duration"]) == (5, (480, 270), 200)   # 20 frames, every 4th, 5 fps
    assert _row_files(tmp_path, skip=("row.json", "timeline.png", "preview.gif")) == \
        {k: v for k, v in before.items() if not k.endswith("row.json")}
    (d,) = C.run(tmp_path, log=lambda *a: None, timeline=True)            # a picture not asked for again is removed
    assert list(d["pictures"]) == ["timeline"] and not (rd / "preview.gif").exists()
    C.run(tmp_path, log=lambda *a: None)
    assert _row_files(tmp_path) == before
    for bad in ({"gif_every": 0}, {"gif_scale": 0.0}):
        with pytest.raises(C.CatalogError, match="gif_every must be"):
            C.run(tmp_path, log=lambda *a: None, gif=True, **bad)


def test_gallery_lists_the_whole_catalog_with_the_pictures_of_the_run(tmp_path):
    from dms_model_simulator import gallery as G
    ran, other = _short(), _short(id="POS-PHN-TEST", number=2, title="not <run>", tags=["x"])
    planned = _short(id="NEG-DIS-TEST", number=3, status="planned", needs=["a thing"])
    man = C.expand([ran], tmp_path, "boundary", profiles=PR)
    (rid,) = [r["row_id"] for r in man["rows"]]
    page = G.gallery_html(tmp_path, entries=[ran, other, planned])           # before `catalog run`
    assert "這一列還沒匯出" in page and "<img" not in page
    C.run(tmp_path, log=lambda *a: None)
    page = G.gallery_html(tmp_path, entries=[ran, other, planned])
    assert "這一列沒有圖" in page and "<img" not in page and f'href="rows/{rid}/card.md"' in page
    C.run(tmp_path, log=lambda *a: None, timeline=True, gif=True)
    out, out_en = G.write_gallery(tmp_path, entries=[ran, other, planned])
    page = out.read_text(encoding="utf-8")
    assert (out, out_en) == (tmp_path / "index.html", tmp_path / "index.en.html") and b"\r" not in out.read_bytes()
    assert f'src="rows/{rid}/timeline.png"' in page and f'src="rows/{rid}/preview.gif"' in page
    assert f'href="scenarios/{Path(man["rows"][0]["scenario"]).name}"' in page
    assert str(tmp_path) not in page                                          # relative links only
    for e in (ran, other, planned):                                           # every entry, run or not: a page,
        assert f'id="e-{e["id"]}"' in page and page.count(f'href="#e/{e["id"]}"') >= 2   # a list line, prev / next
    assert "not &lt;run&gt;" in page and "這次的 run 沒有選這一條" in page and "缺 a thing" in page
    assert "call_s=0.5" in page and "看 road_ahead" in page                   # the row's parameters and script
    # the map: behaviour family x category, a count per cell; one list per family; reading order = family, category
    assert '<td class="n"><a href="#g/PHN/BND"><b>1</b><span class="rg">#1</span></a></td>' in page
    assert '<a href="#g/DIS/NEG"><b>1</b><span class="rg">#3</span></a>' in page
    assert 'id="g-PHN"' in page and 'id="g-PHN-POS"' in page and 'id="g-DIS-NEG"' in page
    assert [page.index(f'id="e-{i}"') for i in ("NEG-DIS-TEST", "POS-PHN-TEST", "BND-PHN-TEST")] == sorted(
        page.index(f'id="e-{i}"') for i in ("NEG-DIS-TEST", "POS-PHN-TEST", "BND-PHN-TEST"))
    assert "目錄有 3 條情境" in page and "這次的 run 跑了 1 條、1 列，1 列有圖" in page
    # home, a page per standard, two families, three entries
    assert page.count('class="view') == 1 + len(C.load_standards()) + 2 + 3


def test_gallery_numbers_position_star_and_sources(tmp_path):
    from dms_model_simulator import gallery as G
    assert G._ranges([48, 45, 46, 47]) == "#45–48" and G._ranges([45, 46, 47, 90]) == "#45–47, 90"
    assert G._ranges([3]) == "#3" and G._ranges([1, 3, 4]) == "#1, 3–4" and G._ranges([]) == ""
    es = [_short(id="POS-PHN-A", number=45), _short(id="POS-PHN-B", number=46, behaviors=["N20"]),
          _short(id="BND-PHN-C", number=90, source="generated", behaviors=["N20"]),
          _short(id="NEG-DIS-D", number=3, behaviors=["N20"])]
    C.expand(es[:1], tmp_path, "boundary", profiles=PR)
    page = G.gallery_html(tmp_path, entries=es)
    sec = lambda i: page[page.index(f'id="{i}"'):].split("</section>")[0]
    # map: codes next to the names, numbers under the counts (contiguous ones merged)
    assert '<span class="code">POS</span></th>' in page and '手機</a><span class="code">PHN</span></th>' in page
    assert '<a href="#g/PHN/POS"><b>2</b><span class="rg">#45–46</span></a>' in page
    assert '<a href="#g/PHN"><b>3</b><span class="rg">#45–46, 90</span></a>' in page
    # family list: the number first; the star only on entries whose main behaviour is must-test
    assert '<a href="#e/POS-PHN-A" data-e="POS-PHN-A"><span class="no">#045</span><span class="t">★ t</span>' in sec("g-PHN")
    assert '<span class="no">#046</span><span class="t">t</span>' in sec("g-PHN")
    n_core = sum(C.is_core(b) for b in C.load_behaviors().values())
    star = f"<code>must_test: true</code> 的 {n_core} 個行為。這是本專案自己訂的清單，不是法規要求。"
    assert star in sec("g-PHN") and "★" not in sec("g-DIS")        # the note only where a star is shown
    assert star in sec("home") and "目錄裡 1 條標了 ★" in sec("home")
    # entry: number by the title, place in the family, previous / next with numbers, source in words
    a, c = sec("e-POS-PHN-A"), sec("e-BND-PHN-C")
    assert '<h1><span class="no">#045</span> t</h1>' in a and "› #045，這一家族第 1／3 條" in a
    assert "› #090，這一家族第 3／3 條" in c and "來源：隨機產生" in c and "來源：手寫劇本" in a
    assert '<a class="prev" href="#e/NEG-DIS-D">← #003 t</a>' in a
    assert '<a class="next" href="#e/POS-PHN-B" style="text-align:right">#046 t →</a>' in a
    # one family column per family (not one per entry); the bar marks families for the script
    assert page.count('<nav class="fam"') == 2 and 'id="s-PHN"' in page and 'id="s-DIS"' in page
    assert all(page.count(f'data-e="{e["id"]}"') == 2 for e in es)    # its list line + its family column
    assert 'data-f=""' in page and '<a href="#g/PHN" data-f="PHN">' in page
    # where the cases come from: counted from the catalog
    assert ("手寫劇本（人照行為逐段寫的劇本）3 條、隨機產生（隨機產生器排出的長段駕駛）1 條、"
            "真錄影整理（從真實錄影整理出的劇本）0 條") in sec("home")
    assert "目前沒有逐條對應到法規或標準條文" in sec("home") and "每條的門檻與時長" in sec("home")
    # search index: [id, label, text, number]
    index = json.loads(page.split("const INDEX=")[1].split(";\ndocument")[0])
    assert ["POS-PHN-A", "#045 t"] == index[1][:2] and index[1][3] == 45 and index[0][3] == 3
    # an entry known only from the manifest has no number: listed without one
    page = G.gallery_html(tmp_path, entries=es[1:])
    orphan = page[page.index('id="e-POS-PHN-A"'):].split("</section>")[0]
    assert "#045" not in orphan and "這一家族第 2／3 條" in orphan and '<span class="rg">#46, 90</span>' in page


# ---- standards (catalog/standards.yaml) and the English page ------------------------------------------------
def _bi(zh, en):
    return {"zh": zh, "en": en}


def escape_html(s: str) -> str:
    from html import escape
    return escape(s)


STD = {"STD-1": {"title": _bi("標準一", "Standard one"), "summary": _bi("摘要一", "Summary one"),
                 "not_covered": _bi("不涵蓋一", "Not covered one"), "source": "https://example.org/std1",
                 "checked": "2026-11-11", "requirements": {
                     "R1": {"clause": "§1", "text": _bi("要求一", "Text one"), "coverage": "partial",
                            "gap": _bi("缺口一", "Gap one")},
                     "R2": {"clause": "§2", "text": _bi("要求二", "Text two"), "coverage": "gap",
                            "gap": _bi("缺口二", "Gap two")},
                     "R3": {"clause": "§3", "text": _bi("要求三", "Text three"), "coverage": "out_of_scope",
                            "gap": _bi("缺口三", "Gap three")}}}}


def test_repo_standards_valid_and_consistent():
    from collections import Counter
    st, entries = C.load_standards(), C.load_entries()
    assert C.validate_standards(st, entries) == []
    reqs = st["STD-A"]["requirements"]
    assert list(reqs) == [f"R{i}" for i in range(1, 22)]
    assert Counter(r["coverage"] for r in reqs.values()) == {"partial": 5, "gap": 3, "out_of_scope": 13}
    related = {e["id"] for e in C.standard_entries(entries, "STD-A")}
    assert {e["id"] for e in entries if C.family(e) in ("DRW", "YWN")} <= related     # the marked families: all
    assert not any(C.family(e) in ("DIS", "PHN", "SMK", "SBT") for e in entries if e["id"] in related)
    overlay = C.load_standards_overlay()                       # the maintainer's local names, when present
    if overlay is not None:
        assert C.validate_standards_overlay(overlay, st) == []


def test_repo_public_standards_hold_markers_only():
    """The public file names no standard and holds no text: {id: {requirements: {Rn: {coverage}}}} and nothing
    else. Names and texts belong in the git-ignored overlay, which only the gallery merges in."""
    raw = C.load_yaml(C.CATALOG_DIR / "standards.yaml")
    assert set(raw) == {"standards"} and raw["standards"]
    for sid, s in raw["standards"].items():
        assert set(s) == {"requirements"}, sid
        assert all(isinstance(r, dict) and set(r) == {"coverage"} for r in s["requirements"].values()), sid
    assert C.load_standards() == raw["standards"]                              # the default ignores the overlay
    assert "catalog/standards.local.yaml" in (C.ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert C.standards_local_path() == C.CATALOG_DIR / "standards.local.yaml"


def _write_yaml(p: Path, d: dict) -> Path:
    import yaml
    p.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def _markers(std: dict) -> dict:
    """The public part of a standards registry: requirement ids and coverage only."""
    return {sid: {"requirements": {rid: {"coverage": r["coverage"]} for rid, r in s["requirements"].items()}}
            for sid, s in std.items()}


def _overlay(std: dict, name: str = "Name One") -> dict:
    """The local part: everything but coverage, plus a display name."""
    return {sid: {"name": name, **{k: v for k, v in s.items() if k != "requirements"},
                  "requirements": {rid: {k: v for k, v in r.items() if k != "coverage"}
                                   for rid, r in s["requirements"].items()}} for sid, s in std.items()}


def test_load_standards_with_the_local_overlay(tmp_path):
    pub = _write_yaml(tmp_path / "standards.yaml", {"standards": _markers(STD)})
    assert C.load_standards(pub) == C.load_standards(pub, local=True) == _markers(STD)     # no overlay file yet
    assert C.load_standards_overlay(C.standards_local_path(pub)) is None
    _write_yaml(C.standards_local_path(pub), {"standards": _overlay(STD)})
    assert C.load_standards(pub) == _markers(STD)                                          # default: public only
    merged = C.load_standards(pub, local=True)
    assert merged == {"STD-1": {**STD["STD-1"], "name": "Name One"}}                     # deep-merged, coverage kept
    assert C.standard_label("STD-1", merged["STD-1"]) == "Name One" and C.standard_label("STD-1", STD["STD-1"]) == "STD-1"
    other = _write_yaml(tmp_path / "elsewhere.yaml", {"standards": {"STD-1": {"name": "Other"}}})
    assert C.load_standards(pub, local=other)["STD-1"] == {**_markers(STD)["STD-1"], "name": "Other"}  # explicit path
    assert C.load_standards(tmp_path / "missing.yaml", local=True) == {}                  # no public file, nothing
    stray = _write_yaml(tmp_path / "stray.yaml", {"standards": {"STD-9": {"name": "x"}, "STD-1": {
        "coverage": "x", "requirements": {"R1": {"coverage": "gap", "clause": "§9"}, "R9": {"clause": "§9"}}}}})
    got = C.load_standards(pub, local=stray)          # only adds to what the public file has; coverage stays its own
    assert set(got) == {"STD-1"} and "coverage" not in got["STD-1"] and set(got["STD-1"]["requirements"]) == {
        "R1", "R2", "R3"} and got["STD-1"]["requirements"]["R1"] == {"coverage": "partial", "clause": "§9"}


def test_validate_standards_public_markers_and_optional_texts():
    one = [_entry(id="POS-DRW-A", standards={"STD-1": ["R1"]})]
    assert C.validate_standards(_markers(STD), one) == []
    bad = _markers(STD)
    bad["STD-1"].update(name=" ", title={"zh": "只有中文"})
    bad["STD-1"]["requirements"]["R1"]["clause"] = ""
    errs = C.validate_standards(bad, one)
    for s in ("STD-1: name must be a non-empty string", "STD-1: title needs a non-empty text per language",
              "STD-1 R1: clause must be a non-empty string"):
        assert any(s in x for x in errs), s
    assert C.validate_standards({"STD-1": {"requirements": {}}}) == ["STD-1: missing requirements"]


def test_validate_standards_overlay():
    pub = _markers(STD)
    assert C.validate_standards_overlay(_overlay(STD), pub) == [] and C.validate_standards_overlay({}, pub) == []
    ov = _overlay(STD)
    ov["STD-9"] = {"name": "x"}                                                  # a standard the public file lacks
    ov["STD-1"]["requirements"]["R9"] = {"clause": "§9"}                         # a requirement it lacks
    ov["STD-1"]["requirements"]["R1"]["coverage"] = "partial"                    # coverage: public file only
    del ov["STD-1"]["name"], ov["STD-1"]["requirements"]["R2"]["gap"], ov["STD-1"]["requirements"]["R3"]
    errs = C.validate_standards_overlay(ov, pub)
    for s in ("STD-9: not in standards.yaml", "STD-1 R9: not in standards.yaml",
              "STD-1 R1: coverage belongs in standards.yaml only", "STD-1: missing name",
              "STD-1 R2: gap needs a non-empty text per language", "STD-1 R3: missing (every requirement"):
        assert any(s in x for x in errs), (s, errs)
    ov = _overlay(STD)
    ov["STD-1"]["coverage"] = "x"
    del ov["STD-1"]["summary"]
    errs = C.validate_standards_overlay(ov, pub)
    assert "STD-1: coverage belongs in standards.yaml only" in errs
    assert any("STD-1: summary needs a non-empty text" in x for x in errs)


def test_validate_entry_standards():
    ok = lambda v, **kw: C.validate(_entry(standards=v, **kw), PR, standards=STD)
    assert ok({"STD-1": ["R1"]}) == [] and ok({"STD-1": []}) == []
    assert any("'NOPE' is not in catalog/standards.yaml" in x for x in ok({"NOPE": []}))
    assert any("has no requirement 'R9'" in x for x in ok({"STD-1": ["R9"]}))
    assert any("needs a list" in x for x in ok({"STD-1": "R1"}))
    assert any("lists 'R1' twice" in x for x in ok({"STD-1": ["R1", "R1"]}))
    assert any("must be a mapping" in x for x in ok(["STD-1"]))
    assert any("not in catalog/standards.yaml" in x for x in ok({"NOPE": []}, status="planned", needs=["x"]))
    assert any("title_en must be a non-empty string" in x for x in ok({"STD-1": []}, title_en=""))
    assert C.validate(_entry(title_en="A call", summary_en="call of call_s seconds"), PR) == []


def test_validate_english_texts_are_required():
    no = lambda *keys, **kw: {k: v for k, v in _entry(**kw).items() if k not in keys}
    assert any("missing title_en" in x for x in C.validate(no("title_en"), PR))
    assert any("missing summary_en" in x for x in C.validate(no("summary_en"), PR))
    assert C.validate(no("summary", "summary_en"), PR) == []                   # no summary, no summary_en
    assert any("summary_en without summary" in x for x in C.validate(no("summary"), PR))
    beh = C.load_behaviors()
    assert all(b.get("name_en") for b in beh.values())
    v = {k: dict(b) for k, b in beh.items()}
    del v["V08"]["name_en"]
    v["V01"]["name_en"] = " "
    errs = C.validate_behaviors(v)
    assert "V08: missing name_en" in errs and "V01: name_en must be a non-empty string" in errs


def test_validate_standards_structure_and_coverage():
    one = [_entry(id="POS-DRW-A", standards={"STD-1": ["R1"]})]
    assert C.validate_standards(STD, one) == [] and C.validate_standards(STD) == []
    assert C.validate_standards(STD, []) == ["STD-1 R1: coverage partial but no entry lists it"]
    two = one + [_entry(id="POS-DRW-B", standards={"STD-1": ["R2", "R3"]})]
    assert C.validate_standards(STD, two) == ["STD-1 R2: coverage gap but listed by POS-DRW-B",
                                              "STD-1 R3: coverage out_of_scope but listed by POS-DRW-B"]
    broken = copy.deepcopy(STD)
    del broken["STD-1"]["summary"]["en"]
    broken["STD-1"]["source"] = ""
    broken["STD-1"]["requirements"]["R2"]["coverage"] = "full"
    del broken["STD-1"]["requirements"]["R3"]["clause"]                    # texts are optional: no error
    errs = C.validate_standards(broken)
    for s in ("STD-1: summary needs a non-empty text per language", "STD-1: source must be a non-empty string",
              "STD-1 R2: coverage must be one of"):
        assert any(s in x for x in errs), s
    assert not any("R3" in x for x in errs)


def test_gallery_standard_page_and_marks(tmp_path):
    from dms_model_simulator import gallery as G
    es = [_short(id="POS-DRW-A", number=1, standards={"STD-1": ["R1"]}),
          _short(id="POS-DRW-B", number=2, standards={"STD-1": []}), _short(id="POS-PHN-C", number=3)]
    C.expand(es[:1], tmp_path, "boundary", profiles=PR)
    page = G.gallery_html(tmp_path, entries=es, standards=STD)
    sec = lambda i, p=page: p[p.index(f'id="{i}"'):].split("</section>")[0]
    s = sec("std-STD-1")
    # the standard's own page: what it is and does not cover, requirements in three groups (the last one folded)
    assert "<h1>標準一</h1>" in s and "摘要一" in s and "不涵蓋一" in s and "目錄有 2 條情境標了跟這個標準的關係" in s
    assert [s.index(x) for x in ("目錄有相關情境", "還沒有情境", "模擬器管不到")] == sorted(
        s.index(x) for x in ("目錄有相關情境", "還沒有情境", "模擬器管不到"))
    assert '<details class="more"><summary><h2>模擬器管不到</h2></summary>' in s and "缺口三" in s
    r1 = s[s.index('id="std-STD-1-R1"'):].split("</tr>")[0]
    assert '<a href="#e/POS-DRW-A" title="t">#001</a>' in r1 and "缺口一" in r1 and "§1" in r1 and "要求一" in r1
    assert "POS-DRW-B" not in r1 and "缺口二" in s
    assert '另外 1 條跟這個標準管的主題有關，沒有對到特定一項：<a href="#e/POS-DRW-B" title="t">#002</a>' in s
    assert '<a href="https://example.org/std1">標準全文</a>（2026-11-11 查證）' in s
    # an entry on the bar; on the home page one line and a count per map cell, no requirement table
    assert '<a class="std" href="#s/STD-1" data-f="s/STD-1">STD-1</a>' in page
    home = sec("home")
    assert '2 條情境標了跟 <a href="#s/STD-1">STD-1</a> 哪幾項要求有關' in home
    assert "R1" not in home and "目前沒有逐條對應" not in home
    assert home.count('<span class="sm" title="2 條跟 STD-1 有關">STD-1 · 2</span>') == 2   # DRW x POS, DRW total
    # marks: family list line, family column, entry title; the entry page says which requirements
    assert '<span class="t">★ t<span class="sm">STD-1</span></span>' in sec("g-DRW")
    assert '<span class="no">#001</span><span class="sm">STD-1</span> ★ t</a>' in page[page.index('id="s-DRW"'):]
    a, b, c = sec("e-POS-DRW-A"), sec("e-POS-DRW-B"), sec("e-POS-PHN-C")
    assert '<h1><span class="no">#001</span> t <a class="sm" href="#s/STD-1">STD-1</a></h1>' in a
    assert '<a href="#s/STD-1">標準一</a>' in a and "<li>R1（§1）：要求一</li>" in a
    assert "跟這個標準管的主題有關，沒有對到特定一項。" in b and "<li>R1" not in b
    assert '<h1><span class="no">#003</span> t</h1>' in c and "stdbox" not in c and 'class="sm"' not in sec("g-PHN")
    # the same in English
    en = G.gallery_html(tmp_path, entries=es, standards=STD, lang="en")
    s = sec("std-STD-1", en)
    assert "<h1>Standard one</h1>" in s and "Summary one" in s and "Scenarios in the catalog" in s
    assert "<h2>Out of the simulator&#x27;s reach</h2>" in s and "Gap one" in s and "(checked 2026-11-11)" in s
    assert "<li>R1 (§1): Text one</li>" in sec("e-POS-DRW-A", en) and "STD-1 · 2" in sec("home", en)


def test_gallery_standard_page_with_markers_only(tmp_path):
    """Without the local overlay a standard is an id with requirement ids: the page says so, lists the ids and the
    related entries, and holds none of the overlay's texts."""
    from dms_model_simulator import gallery as G
    es = [_short(id="POS-DRW-A", number=1, standards={"STD-1": ["R1"]}),
          _short(id="POS-DRW-B", number=2, standards={"STD-1": []}), _short(id="POS-PHN-C", number=3)]
    C.expand(es[:1], tmp_path, "boundary", profiles=PR)
    for lg, intro, ent in (("zh", G.TEXT["zh"]["std_intro"], "STD-1</a>：R1</p>"),
                           ("en", G.TEXT["en"]["std_intro"], "STD-1</a>: R1</p>")):
        page = G.gallery_html(tmp_path, entries=es, standards=_markers(STD), lang=lg)
        sec = lambda i, p=page: p[p.index(f'id="{i}"'):].split("</section>")[0]
        s = sec("std-STD-1")
        assert "<h1>STD-1</h1>" in s and escape_html(intro) in s
        r1 = s[s.index('id="std-STD-1-R1"'):].split("</tr>")[0]
        assert '<a href="#e/POS-DRW-A" title="t">#001</a>' in r1
        assert '<b id="std-STD-1-R2">R2</b>' in s and '<b id="std-STD-1-R3">R3</b>' in s
        assert "#002</a>" in s                                                   # topic only: still listed
        assert ent in sec("e-POS-DRW-A") and '<a class="sm" href="#s/STD-1">STD-1</a>' in sec("e-POS-DRW-A")
        for text in ("標準一", "Standard one", "摘要一", "Summary one", "§1", "要求一", "Text one", "缺口一",
                     "Gap one", "example.org", "2026-11-11", "R1（", "R1 ("):
            assert text not in page, (lg, text)
        assert '<a class="std" href="#s/STD-1" data-f="s/STD-1">STD-1</a>' in page


def test_gallery_shows_the_local_name_of_a_standard(tmp_path):
    from dms_model_simulator import gallery as G
    es = [_short(id="POS-DRW-A", number=1, standards={"STD-1": ["R1"]})]
    C.expand(es, tmp_path, "boundary", profiles=PR)
    pub = _write_yaml(tmp_path / "standards.yaml", {"standards": _markers(STD)})
    _write_yaml(C.standards_local_path(pub), {"standards": _overlay(STD)})
    page = G.gallery_html(tmp_path, entries=es, standards=C.load_standards(pub, local=True))
    assert '<a class="std" href="#s/STD-1" data-f="s/STD-1">Name One</a>' in page    # label = name, route = id
    assert '<a class="sm" href="#s/STD-1">Name One</a>' in page and "Name One · 1" in page
    assert "<h1>標準一</h1>" in page and "<li>R1（§1）：要求一</li>" in page and "example.org/std1" in page
    assert G.TEXT["zh"]["std_intro"] not in page


def test_gallery_two_languages_link_each_other(tmp_path, monkeypatch):
    from dms_model_simulator import gallery as G
    e = _short(id="POS-PHN-A", title="講電話", title_en="On the phone", summary="中文說明", summary_en="English summary",
               short="講電話", short_en="On the phone")
    f = _short(id="POS-PHN-B", number=2, title="只有中文")
    del f["title_en"]                                                      # a page shows the Chinese when English is missing
    C.expand([e], tmp_path, "boundary", profiles=PR)
    C.run(tmp_path, log=lambda *a: None)
    real = C.load_behaviors()
    monkeypatch.setattr(C, "load_behaviors", lambda path=None: {**real, "V08": {**real["V08"], "name_en": "Phone call"}})
    zh, en = G.write_gallery(tmp_path, entries=[e, f], standards={})
    pz, pe = zh.read_text(encoding="utf-8"), en.read_text(encoding="utf-8")
    assert '<html lang="zh-Hant">' in pz and '<html lang="en">' in pe
    # each page links to the other; the script adds the current #... so the link lands on the same place
    assert 'id="lang" href="index.en.html" data-file="index.en.html"' in pz
    assert 'id="lang" href="index.html" data-file="index.html"' in pe
    assert all("lang.href=lang.dataset.file+location.hash" in p for p in (pz, pe))
    # English page: the entry's and the behaviour's English, the script built from the card keys; Chinese if missing
    assert '<h1><span class="no">#001</span> On the phone</h1>' in pe and "English summary" in pe
    assert "V08 Phone call" in pe and "只有中文" in pe
    assert "looks at road_ahead; hand: call; phone: call" in pe and "看 road_ahead" not in pe
    assert '<h1><span class="no">#001</span> 講電話</h1>' in pz and "看 road_ahead" in pz and "English summary" not in pz
    for zh_text in ("情境總覽", "搜尋", "劇本", "這條情境的說明", "相機", "手機", "行為真的發生", "找不到"):
        assert zh_text not in pe, zh_text
    assert "Phone</a>" in pe and "Behaviour happens" in pe and "No match" in pe
    # --lang writes only that page
    en.unlink()
    assert G.write_gallery(tmp_path, entries=[e], lang="zh", standards={}) == [zh] and not en.exists()
    with pytest.raises(ValueError, match="lang must be one of"):
        G.gallery_html(tmp_path, lang="fr")


def test_gallery_texts_have_the_same_keys_in_every_language():
    import re
    from dms_model_simulator import gallery as G

    def shape(v):
        if isinstance(v, dict):
            return {k: shape(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [shape(x) for x in v]
        return sorted(set(re.findall(r"\{(\w*)\}", v)))                  # the placeholders of a text
    assert set(G.TEXT) == set(G.LANGS) == set(G.FILES)
    assert shape(G.TEXT["en"]) == shape(G.TEXT["zh"])
    assert set(C.CATEGORIES_EN) == set(C.CATEGORIES) and set(C.FAMILIES_EN) == set(C.FAMILIES)
    assert all(": " in v for v in C.CATEGORIES_EN.values())


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
    monkeypatch.setattr(C, "load_standards", lambda path=None, local=False: {})   # the repo's need the repo's entries
    monkeypatch.setattr(C, "load_standards_overlay", lambda path=None: None)      # and the maintainer's local file
    capsys.readouterr()
    cli.main(["catalog", "list"])
    assert "#001  BND-PHN-TEST" in capsys.readouterr().out
    cli.main(["catalog", "validate", "BND"])
    assert "BND-PHN-TEST: ok" in capsys.readouterr().out
    monkeypatch.setattr(C, "load_standards", lambda path=None, local=False: STD)  # coverage: against the whole catalog
    with pytest.raises(SystemExit):
        cli.main(["catalog", "validate", "BND"])
    out = capsys.readouterr().out
    assert "STD-1 R1: coverage partial but no entry lists it" in out and "standards.local.yaml" not in out
    monkeypatch.setattr(C, "load_standards", lambda path=None, local=False: _markers(STD))
    monkeypatch.setattr(C, "load_standards_overlay", lambda path=None: {"STD-1": {"name": "x"}})   # incomplete
    with pytest.raises(SystemExit):
        cli.main(["catalog", "validate", "BND"])
    assert "standards.local.yaml: \n  STD-1: title needs" in capsys.readouterr().out
    monkeypatch.setattr(C, "load_standards_overlay", lambda path=None: None)
    monkeypatch.setattr(C, "load_standards", lambda path=None, local=False: {})
    cli.main(["catalog", "expand", "BND", "--out", str(tmp_path), "--profile", "nominal"])
    assert "1 rows from 1 entries" in capsys.readouterr().out
    cli.main(["catalog", "run", str(tmp_path)])
    out = capsys.readouterr().out
    assert "exported 1 rows" in out and "rows.tsv" in out
    (t,) = _rows_tsv(tmp_path)
    assert Path(t["card"]).name == "card.md" and Path(t["card"]).exists()
    assert not (Path(t["card"]).parent / "timeline.png").exists()
    cli.main(["catalog", "run", str(tmp_path), "--timeline"])
    assert "plus timeline.png per row" in capsys.readouterr().out
    assert (Path(t["card"]).parent / "timeline.png").exists()
    cli.main(["catalog", "gallery", str(tmp_path), "--title", "T"])
    out = capsys.readouterr().out
    assert "index.html" in out and "index.en.html" in out and "standards: none, markers only" in out
    assert "<title>T</title>" in (tmp_path / "index.html").read_text("utf-8")
    # the local overlay: used when the file exists, not with --no-local; the page shows the name it brings
    monkeypatch.setattr(C, "load_standards_overlay", lambda path=None: {"STD-1": {"name": "Local name"}})
    monkeypatch.setattr(C, "load_standards", lambda path=None, local=False:
                        {"STD-1": {"name": "Local name", **_markers(STD)["STD-1"]} if local else _markers(STD)["STD-1"]})
    cli.main(["catalog", "gallery", str(tmp_path), "--lang", "zh"])
    assert "standards: STD-1 with local names" in capsys.readouterr().out
    assert ">Local name</a>" in (tmp_path / "index.html").read_text("utf-8")
    cli.main(["catalog", "gallery", str(tmp_path), "--lang", "zh", "--no-local"])
    assert "standards: STD-1, markers only" in capsys.readouterr().out
    assert "Local name" not in (tmp_path / "index.html").read_text("utf-8")
    monkeypatch.setattr(C, "load_standards_overlay", lambda path=None: None)
    monkeypatch.setattr(C, "load_standards", lambda path=None, local=False: {})
    assert '<html lang="en">' in (tmp_path / "index.en.html").read_text("utf-8")
    cli.main(["catalog", "gallery", str(tmp_path), "--lang", "en"])
    assert "index.en.html" in capsys.readouterr().out
    cli.main(["catalog", "index", "--out", str(tmp_path / "INDEX.md")])
    assert "BND-PHN-TEST" in (tmp_path / "INDEX.md").read_text(encoding="utf-8")


# ---- difficulty levels, repeat blocks, parameter overrides --------------------------------------------------
LV_BRIEF = {"small": "1 glance", "medium": "2 glances", "large": "3 glances"}


def _lv_entry(**kw):
    """call_s and gap_s follow the row's level; side is an ordinary choice."""
    base = dict(params={"call_s": {"levels": {"small": 0.5, "medium": 1.0, "large": 2.0}},
                        "gap_s": {"levels": {"small": 2.0, "medium": 1.0, "large": 0.5}},
                        "side": {"choice": ["a", "b"]}},
                template={"segments": [{"target": "road_ahead", "duration": "${gap_s}"},
                                       {"target": "road_ahead", "duration": "${call_s}", "hand": "call",
                                        "phone": "call"}]},
                brief={"small": "call 0.5 s", "medium": "call 1 s", "large": "call 2 s"},
                brief_en={"small": "call 0.5 s", "medium": "call 1 s", "large": "call 2 s"})
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
    e = _entry(params={"n": {"levels": {"small": 1, "medium": 2, "large": 3}}}, brief=LV_BRIEF, brief_en=LV_BRIEF,
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
    md = C.index_markdown([_lv_entry(id="NEG-PHN-LV", number=2), _entry()])
    assert "| #002 | `NEG-PHN-LV` | t | V08 |  | 三段 | ready |" in md
    assert "| #001 | `BND-PHN-TEST` | t | V08 |  |  | ready |" in md


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


# ---- short names, level table, review table ------------------------------------------------------------------
def test_validate_short_and_brief():
    ok = lambda **kw: C.validate(_entry(**kw), PR)
    has = lambda errs, s: any(s in x for x in errs)
    assert ok() == [] and C.validate(_lv_entry(), PR) == []
    no = {k: v for k, v in _entry().items() if k not in ("short", "brief_en")}
    errs = C.validate(no, PR)
    assert has(errs, "missing short") and has(errs, "missing brief_en")
    assert has(ok(short=" "), "short must be a non-empty string")
    assert has(ok(short="字" * 15), "short has 15 characters (at most 14)") and ok(short="字" * 14) == []
    assert has(ok(short_en="x" * 46), "short_en has 46 characters (at most 45)")
    # without levels: exactly `all`; with levels: small, medium, large and optionally all
    assert has(ok(brief={"small": "a"}, brief_en={"small": "a"}), "without levels needs exactly all")
    lv = dict(LV_BRIEF)
    assert C.validate(_lv_entry(brief={**lv, "all": "x"}, brief_en={**lv, "all": "x"}), PR) == []
    errs = C.validate(_lv_entry(brief={"small": "a", "large": "c"}, brief_en={"all": "a"}), PR)
    assert has(errs, "brief keys ['large', 'small']: need small, medium, large") and has(errs, "different keys")
    assert has(ok(brief={"all": "字" * 51}), "brief.all has 51 characters (at most 50)")
    assert ok(brief={"all": "字" * 50}, brief_en={"all": "x" * 110}) == []
    assert has(C.validate(_lv_entry(brief={**lv, "small": "字" * 37}), PR), "brief.small has 37 characters (at most 36)")
    assert has(C.validate(_lv_entry(brief_en={**lv, "large": "x" * 81}), PR), "brief_en.large has 81 characters")
    assert has(ok(brief={"all": ""}), "brief.all must be a non-empty string")
    assert has(ok(brief="x"), "brief must be a mapping")
    assert C.SHORT_MAX == {"zh": 14, "en": 45}
    assert C.BRIEF_MAX == {"level": {"zh": 36, "en": 80}, "all": {"zh": 50, "en": 110}}


def test_validate_catalog_short_names_unique():
    a, b = _entry(id="POS-PHN-A", short="同名"), _entry(id="POS-PHN-B", number=2, short="同名", short_en="u")
    assert C.validate_catalog([a, _entry(id="POS-PHN-C", number=3, short="別名", short_en="v")]) == []
    assert C.validate_catalog([a, b]) == ["short '同名' is used by both POS-PHN-A and POS-PHN-B"]
    c = _entry(id="POS-PHN-C", number=3, short="別名")                    # short_en "t", as a's
    assert C.validate_catalog([a, c]) == ["short_en 't' is used by both POS-PHN-A and POS-PHN-C"]


def _review_fixture():
    beh = {"V08": {"name": "講電話", "name_en": "Phone call"}, "N28": {"name": "手機放車架"}}
    es = [_lv_entry(id="NEG-PHN-LV", number=12, short="有|直線", short_en="With | bar", behaviors=["N28", "V08"],
                    brief={**LV_BRIEF, "all": "每級兩台相機"}, brief_en={**LV_BRIEF, "all": "two cameras"},
                    standards={"STD-1": ["R1", "R2"]}),
          _entry(id="POS-PHN-ONE", number=3, short="不分級", short_en="No levels", brief={"all": "講 1 s"},
                 brief_en={"all": "1 s call"}, standards={"STD-1": []}),
          _entry(id="POS-PHN-PLAN", number=7, short="還沒做", short_en="Not yet", status="planned", needs=["x"])]
    return es, beh


def test_review_table_both_languages():
    es, beh = _review_fixture()
    std = _markers(STD)
    zh, en = C.review_rows(es, beh, std, "zh"), C.review_rows(es, beh, std, "en")
    assert zh[0] == ["編號", "短名", "類別", "行為", "small", "medium", "large", "每級相同／不分級", "標記", "ID", "確認", "備註"]
    assert en[0] == ["No.", "Short name", "Kind", "Behaviours", "small", "medium", "large",
                     "Same at every level, or no levels", "Marker", "ID", "OK?", "Comment"]
    assert [r[0] for r in zh[1:]] == ["#003", "#007", "#012"]                        # by number
    assert zh[1] == ["#003", "不分級", "行為有發生", "V08 講電話", "", "", "", "講 1 s", "STD-1", "POS-PHN-ONE", "", ""]
    assert zh[2][1] == "還沒做（未完成）" and en[2][1] == "Not yet (planned)"
    assert zh[3] == ["#012", "有|直線", "像但不是", "N28 手機放車架；V08 講電話", "1 glance", "2 glances", "3 glances",
                     "每級兩台相機", "STD-1: R1, R2", "NEG-PHN-LV", "", ""]
    assert en[3][2:4] == ["Looks like it, is not", "N28 手機放車架; V08 Phone call"]    # no name_en: the Chinese
    assert en[1][7] == "1 s call" and en[3][1] == "With | bar"
    md = C.review_markdown(es, beh, std, "zh")
    assert md.startswith("# 情境審閱表\n\n") and "目錄的 3 條全在這一張表" in md and "`dms-sim catalog review`" in md
    assert "| #012 | 有\\|直線 | 像但不是 |" in md and "| `NEG-PHN-LV` |  |  |" in md      # | escaped, id in backticks
    assert md.count("\n| #") == 3 and "|---|" + "---|" * 11 in md
    md_en = C.review_markdown(es, beh, std, "en")
    assert md_en.startswith("# Scenario review table\n") and "With \\| bar" in md_en and "all 3 are in" in md_en
    text = C.review_csv(es, beh, std, "en")
    assert list(csv.reader(text.splitlines())) == en and text == C.review_csv(es, beh, std, "en")
    assert set(C.KINDS) == set(C.CATEGORIES)
    with pytest.raises(C.CatalogError, match="lang must be one of"):
        C.review_rows(es, beh, std, "fr")


def test_review_never_shows_the_local_name(tmp_path, monkeypatch):
    """The review files are committed: even with an overlay next to the public file, the marker is the public id."""
    import shutil
    from dms_model_simulator import cli
    es, beh = _review_fixture()
    cat = tmp_path / "cat"
    cat.mkdir()
    _write_yaml(cat / "standards.yaml", {"standards": _markers(STD)})
    _write_yaml(cat / "standards.local.yaml", {"standards": _overlay(STD, name="SECRET-NAME")})
    shutil.copy(C.CATALOG_DIR / "behaviors.yaml", cat / "behaviors.yaml")
    monkeypatch.setattr(C, "CATALOG_DIR", cat)
    monkeypatch.setattr(C, "load_entries", lambda root=None: es)
    assert C.load_standards(local=True)["STD-1"]["name"] == "SECRET-NAME"            # the overlay is there
    cli.main(["catalog", "review", "--out-dir", str(tmp_path / "out")])
    files = sorted((tmp_path / "out").iterdir())
    assert [p.name for p in files] == ["REVIEW.csv", "REVIEW.en.csv", "REVIEW.en.md", "REVIEW.md"]
    for p in files:
        t = p.read_text(encoding="utf-8-sig")
        assert "SECRET" not in t and "標準一" not in t and "STD-1: R1, R2" in t, p.name
    assert (tmp_path / "out" / "REVIEW.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    assert not (tmp_path / "out" / "REVIEW.md").read_bytes().startswith(b"\xef\xbb\xbf")
    # a merged registry passed in by mistake still gives the id
    assert C.review_rows(es, beh, C.load_standards(local=True))[3][8] == "STD-1: R1, R2"


def test_gallery_short_names_and_levels(tmp_path):
    from dms_model_simulator import gallery as G
    es = [_lv_entry(id="POS-PHN-A", number=45, title="很長的標題：講電話", title_en="A long title: on the phone",
                    short="講電話", short_en="On the phone",
                    brief={**LV_BRIEF, "all": "兩台相機"}, brief_en={**LV_BRIEF, "all": "two cameras"}),
          _entry(id="POS-PHN-B", number=46, title="另一個很長的標題", short="第二條", short_en="Second",
                 brief={"all": "講 1 s"}, brief_en={"all": "1 s call"})]
    C.expand(es[1:], tmp_path, "boundary", profiles=PR)
    cases = (("zh", "很長的標題：講電話", "講電話", "第二條", "講 1 s", "兩台相機"),
             ("en", "A long title: on the phone", "On the phone", "Second", "1 s call", "two cameras"))
    for lg, long_a, short_a, short_b, all_b, all_a in cases:
        page = G.gallery_html(tmp_path, entries=es, standards={}, lang=lg)
        sec = lambda i, p=page: p[p.index(f'id="{i}"'):].split("</section>")[0]
        g = sec("g-PHN")
        assert f'<span class="no">#045</span><span class="t">★ {short_a}</span>' in g and long_a not in g
        side = page[page.index('id="s-PHN"'):].split("</nav>")[0]
        assert f'<span class="no">#045</span> ★ {short_a}</a>' in side and f'title="{long_a}"' in side
        a, b = sec("e-POS-PHN-A"), sec("e-POS-PHN-B")
        assert f'<h1><span class="no">#045</span> {short_a}</h1>\n<p class="sub">{long_a}</p>' in a
        assert escape_html(G.TEXT[lg]["lv_lead"]) in a or G.TEXT[lg]["lv_lead"] in a
        assert "<tr><th>small</th><td>1 glance</td></tr><tr><th>medium</th><td>2 glances</td></tr>" in a
        assert f'<tr><th>{G.TEXT[lg]["lv_all"]}</th><td>{all_a}</td></tr>' in a
        assert f'<p class="lv1"><b>{G.TEXT[lg]["lv_one"]}</b>{all_b}</p>' in b and "lvbox" not in b
        assert f"#046 {short_b} →</a>" in a and f"← #045 {short_a}</a>" in b                   # previous / next
        index = json.loads(page.split("const INDEX=")[1].split(";\ndocument")[0])
        assert index[0][1] == f"#045 {short_a}" and index[0][0] == "POS-PHN-A"
        assert all(t in index[0][2] for t in ("pos-phn-a", "講電話", "on the phone", "很長的標題", "a long title"))
    # an entry known only from the manifest has no short name: its title, no subtitle
    page = G.gallery_html(tmp_path, entries=es[:1], standards={})
    orphan = page[page.index('id="e-POS-PHN-B"'):].split("</section>")[0]
    assert 'class="sub"' not in orphan and "<h1> 另一個很長的標題</h1>" in orphan


def test_gallery_premise_and_note(tmp_path):
    from dms_model_simulator import gallery as G
    es = [_short(id="POS-DRW-A", number=1, standards={"STD-1": ["R1", "R2"]})]
    C.expand(es, tmp_path, "boundary", profiles=PR)
    plain = {lg: G.gallery_html(tmp_path, entries=es, standards=STD, lang=lg) for lg in ("zh", "en")}
    std = copy.deepcopy(STD)
    std["STD-1"]["premise"] = _bi("前提一", "Premise one")
    std["STD-1"]["requirements"]["R1"]["note"] = _bi("我們讀成三秒", "We read it as 3 s")
    assert C.validate_standards(std) == []
    assert C.validate_standards_overlay(_overlay(std), _markers(std)) == []           # optional, accepted
    assert C.validate_standards_overlay(_overlay(STD), _markers(STD)) == []           # and not required
    bad = _overlay(std)
    bad["STD-1"]["premise"] = {"zh": "只有中文"}
    bad["STD-1"]["requirements"]["R1"]["note"] = "x"
    errs = C.validate_standards_overlay(bad, _markers(std))
    assert any("STD-1: premise needs a non-empty text" in x for x in errs)
    assert any("STD-1 R1: note needs a non-empty text" in x for x in errs)
    assert any("STD-1: premise needs" in x for x in C.validate_standards({"STD-1": {**std["STD-1"], "premise": {}}}))
    cases = (("zh", "我們的前提（條文沒有確認）", "我們的解讀：", "前提一", "我們讀成三秒", "要求一"),
             ("en", "Our premise (the text does not confirm it)", "Our reading: ", "Premise one",
              "We read it as 3 s", "Text one"))
    for lg, head, lead, prem, note, text in cases:
        page = G.gallery_html(tmp_path, entries=es, standards=std, lang=lg)
        s = page[page.index('id="std-STD-1"'):].split("</section>")[0]
        box = f'<div class="premise"><p class="ph">{head}</p><p>{prem}</p></div>'
        assert box in s and s.index(box) > s.index('<p class="lead">')                # right under the summary
        r1 = s[s.index('id="std-STD-1-R1"'):].split("</tr>")[0]
        n = f'<span class="note"><b>{lead}</b>{note}</span>'
        assert n in r1 and r1.index(n) > r1.index(text)
        e = page[page.index('id="e-POS-DRW-A"'):].split("</section>")[0]
        assert n + "</li>" in e and 'class="note"' not in s[s.index('id="std-STD-1-R2"'):]
        assert 'class="premise"' not in plain[lg] and 'class="note"' not in plain[lg]       # without them: as before

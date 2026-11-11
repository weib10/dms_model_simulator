"""Exams: build (questions / key), answer-sheet checks, grading of state and alert signals, report.

The grader is checked with made-up answer sheets whose score is known in advance: the truth itself (full marks),
all zeros, the truth shifted by k frames, the truth with a hole, alerts placed at known frames."""
import json
from pathlib import Path
import pytest
from dms_model_simulator import catalog as C
from dms_model_simulator import exam as E
from dms_model_simulator.simulate import Simulation

FPS = 20


def _entry(no, id_, behaviors, segments, params=None):
    return {"id": id_, "number": no, "title": id_, "status": "ready", "behaviors": behaviors, "params": params or {},
            "template": {"segments": segments}, "variation": {"cameras": ["dash_right"], "emulator_seeds": [0]}}


ROAD = {"target": "road_ahead", "duration": 1.0}
ENTRIES = [
    _entry(1, "POS-PHN-TEST", ["V08"], [ROAD, {"target": "road_ahead", "duration": 3.0, "hand": "call", "phone": "call"}, ROAD]),
    _entry(2, "NEG-ALL-TEST", ["N01"], [{"target": "road_ahead", "duration": 3.0}]),
    _entry(3, "BND-PHN-TEST", ["V08"], [ROAD, {"target": "road_ahead", "duration": "${call_s}", "hand": "call",
                                               "phone": "call"}, ROAD], {"call_s": {"grid": [0.5, 1.0, 2.0]}}),
]
EXAM = {"name": "T-v1", "title": "test exam", "intent": True,
        "expand": {"profile": ["exact"], "emulator_seeds": 1, "camera": ["dash_right"]},
        "entries": [e["id"] for e in ENTRIES]}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("exam")
    meta = E.build(EXAM, d, entries=ENTRIES, log=lambda *_: None)
    return d, meta


def _truth_flags(d, meta, concept):
    return {r["row_id"]: [bool(t["events"][concept]) for t in E.load_truth(d, r["row_id"])] for r in meta["rows"]}


def _sheet(tmp_path, d, meta, signals, rows):
    return E.grade(d, E.write_answers(tmp_path / "answers", meta, signals, rows))


PHONE = {"phone": {"concept": "phone_use", "kind": "state"}}


# ---- build ---------------------------------------------------------------------------------------------
def test_build_separates_questions_from_the_key(built):
    d, meta = built
    assert meta == json.loads((d / "exam.json").read_text(encoding="utf-8"))
    assert (meta["exam"], meta["rows_total"], meta["cameras"], meta["profiles"]) == ("T-v1", 5, ["dash_right"], ["exact"])
    assert meta["fingerprint_id"] == C.load_manifest(d / "key")["fingerprint_id"]
    by_entry = {}
    for r in meta["rows"]:
        by_entry.setdefault(r["entry"], []).append(r)
        q = d / "questions" / r["row_id"]
        assert sorted(p.name for p in q.iterdir()) == ["pipeline.jsonl", "row.json"]        # no truth, no card
        row = json.loads((q / "row.json").read_text(encoding="utf-8"))
        assert (row["fps"], row["frames"], row["camera"], row["image_size"]) == (FPS, r["frames"], "dash_right", [1920, 1080])
        lo, hi = row["road_box"]["yaw_deg"]
        assert lo < 0 < hi and row["road_box"]["pitch_deg"][0] < 0 < row["road_box"]["pitch_deg"][1]
        assert len((q / "pipeline.jsonl").read_text(encoding="utf-8").splitlines()) == r["frames"]
        assert len(E.load_truth(d, r["row_id"])) == r["frames"]
        assert r["row_id"].startswith(r["pair_id"] + "__e0__")
    pos, neg, bnd = by_entry["POS-PHN-TEST"][0], by_entry["NEG-ALL-TEST"][0], by_entry["BND-PHN-TEST"]
    assert (pos["main_concept"], pos["intent"]) == ("phone_use", "alert")
    assert (neg["main_concept"], neg["intent"]) == (None, "no_alert")
    assert [(r["params"]["call_s"], r["intent"]) for r in bnd] == [(0.5, None), (1.0, None), (2.0, None)]


def test_build_writes_an_answer_key_to_read(built):
    d, meta = built
    text = (d / "key" / "answer_key.md").read_text(encoding="utf-8")
    assert text == E.answer_key_markdown(d) and "# Answer key: T-v1" in text
    parts = {p.split("`", 2)[1]: p for p in text.split("\n## ")[1:]}     # one section per row
    assert list(parts) == [r["row_id"] for r in meta["rows"]]
    truth = _truth_flags(d, meta, "phone_use")
    for r in meta["rows"]:
        runs = E.segments(truth[r["row_id"]])
        line = "| phone_use | " + ", ".join(f"{s / FPS:.2f}-{e / FPS:.2f}" for s, e in runs) + " | none |"
        assert (line in parts[r["row_id"]]) == bool(runs) == ("| phone_use |" in parts[r["row_id"]])
    pos = next(r["row_id"] for r in meta["rows"] if r["entry"] == "POS-PHN-TEST")
    neg = next(r["row_id"] for r in meta["rows"] if r["entry"] == "NEG-ALL-TEST")
    assert "Built to raise an alert of `phone_use`." in parts[pos] and "Built to raise no alert." in parts[neg]
    assert "(5.0 s, 100 frames at 20 fps, camera `dash_right`, error profile `exact`)" in parts[pos]


def test_build_with_pictures_shows_them_in_the_answer_key(built, tmp_path):
    d, meta = built
    assert "timeline.png" not in (d / "key" / "answer_key.md").read_text(encoding="utf-8")
    one = {**EXAM, "entries": ["POS-PHN-TEST"]}
    m = E.build(one, tmp_path / "p", entries=ENTRIES, log=lambda *_: None, timeline=True, gif=True, gif_every=10)
    (rid,) = [r["row_id"] for r in m["rows"]]
    assert m["rows"] == [r for r in meta["rows"] if r["row_id"] == rid]   # the same row as without pictures
    rd = tmp_path / "p" / "key" / "rows" / rid
    assert (rd / "timeline.png").is_file() and (rd / "preview.gif").is_file()
    assert sorted(p.name for p in (tmp_path / "p" / "questions" / rid).iterdir()) == ["pipeline.jsonl", "row.json"]
    text = (tmp_path / "p" / "key" / "answer_key.md").read_text(encoding="utf-8")
    assert f"![timeline](rows/{rid}/timeline.png)" in text and f"[preview.gif](rows/{rid}/preview.gif)" in text
    assert E.load_truth(tmp_path / "p", rid) == E.load_truth(d, rid)


def test_build_refuses_to_overwrite_and_checks_the_definition(built, tmp_path):
    d, _ = built
    with pytest.raises(E.ExamError, match="already holds an exam"):
        E.build(EXAM, d, entries=ENTRIES)
    with pytest.raises(E.ExamError, match="not in the catalog"):
        E.build({**EXAM, "entries": ["POS-PHN-NOPE"]}, tmp_path / "x", entries=ENTRIES)
    with pytest.raises(E.ExamError, match="unknown key 'levels'"):
        E.load_exam({**EXAM, "expand": {"levels": ["small"]}})
    with pytest.raises(E.ExamError, match="duplicate"):
        E.load_exam({**EXAM, "entries": ["A", "A"]})


def test_repo_exams_load_and_name_catalog_entries():
    ids = {e["id"] for e in C.load_entries()}
    names = E.list_exams()
    assert "A-v1" in names and "grading" not in names
    for n in names:
        ex = E.load_exam(n)
        assert ex["name"] == n and set(ex["entries"]) <= ids
    g = E.load_grading()
    assert set(g["default_evidence"]) == set(E.CONCEPTS)


# ---- the exact profile and observability ---------------------------------------------------------------
def test_exact_reports_every_truth_box_and_observable_says_what_is_there():
    pr = C.load_profiles()
    assert pr["exact"]["perfect_detection"] is True
    sc = {"name": "turn", "cabin": {"targets": {"driver_window": [0.30, 1.00, 0.0]}},     # about 73 deg left
          "segments": [ROAD, {"target": "driver_window", "duration": 2.0, "head_pose": [75, 0, 0]},
                       {"target": "phone_lap", "duration": 1.0, "hand": "phone", "phone": "lap"}]}
    exact = Simulation(sc, "dash_right", emulator_overrides=pr["exact"], fps=FPS)
    nominal = Simulation(sc, "dash_right", fps=FPS)
    missed = 0
    for r, rn in zip(exact.run(), nominal.run()):
        t, p, ob = r["truth"], r["pred"], r["truth"]["observable"]
        assert {x["class"] for x in p["detections"]} == {x["class"] for x in t["detections"]}
        assert set(ob) == set(E.OBSERVABLE_CHANNELS)
        assert bool(p["gaze"]["valid"]) == ob["gaze"]
        assert (p["face_dir"] is not None) == ob["face"]          # the face direction needs the face only
        if p["face_dir"] is not None:
            assert p["face_dir"]["angular_error_deg"] < 1e-6
        assert (p["head_pose_camera"] is not None) == ob["head_pose"] == ob["face"]
        assert (p["eye_openness"] is not None) == ob["eyes"] and (p["mouth_openness"] is not None) == ob["mouth"]
        assert ob["phone"] == t["phone_visible"] and ob == rn["truth"]["observable"]
        missed += any(x["class"] == "face" for x in t["detections"]) and not any(
            x["class"] == "face" for x in rn["pred"]["detections"])
    assert missed > 0                                    # the nominal model does lose the turned face
    obs = [r["truth"]["observable"] for r in exact.records]
    assert not obs[5]["head_away"] and obs[5]["gaze"]    # looking ahead
    assert obs[55]["head_away"]                          # head turned 75 deg: the head pose alone shows it
    assert all(not o["phone"] for o in obs[60:])         # phone in the lap: out of this camera's view
    assert all(o["seatbelt_region"] for o in obs)


def test_truth_jsonl_carries_observable(built):
    d, meta = built
    t = E.load_truth(d, meta["rows"][0]["row_id"])
    assert set(t[0]["observable"]) == set(E.OBSERVABLE_CHANNELS) and all(isinstance(v, bool) for v in t[0]["observable"].values())


# ---- grade_state / grade_alert on frame lists ----------------------------------------------------------
def _flags(n, *runs):
    out = [False] * n
    for s, e in runs:
        out[s:e] = [True] * (e - s)
    return out


def test_grade_state_segments_delays_dropouts_and_false_runs():
    n = 100
    t = _flags(n, (20, 40), (60, 80))
    obs = [True] * n
    g = E.grade_state(t, obs, t, 5)
    assert [(s["detected"], s["onset_delay"], s["release_delay"], s["dropouts"]) for s in g["segments"]] == [(True, 0, 0, 0)] * 2
    assert (g["tp"], g["fn"], g["fp"], g["tn"]) == (20, 0, 0, 40) and not g["false_segments"]   # bands: 2 x 2 x 10 frames
    late = E.grade_state(t, obs, _flags(n, (23, 45), (60, 70), (72, 77), (90, 95)), 5)
    a, b = late["segments"]
    assert (a["onset_delay"], a["release_delay"], a["dropouts"]) == (3, 5, 0)
    assert (b["onset_delay"], b["release_delay"], b["dropouts"]) == (0, -3, 1)
    assert late["false_segments"] == [{"start": 90, "end": 95}] and late["fp"] == 5
    blind = E.grade_state(t, obs, [False] * n, 5)
    assert [s["detected"] for s in blind["segments"]] == [False, False] and blind["fn"] == 20 and blind["tp"] == 0
    after = E.grade_state(t, obs, _flags(n, (42, 50)), 5)                # first response within the band after the end
    assert after["segments"][0]["detected"] and after["segments"][0]["onset_delay"] == 22
    assert not E.grade_state(t, obs, _flags(n, (46, 50)), 5)["segments"][0]["detected"]
    stuck = E.grade_state(t, obs, _flags(n, (60, 100)), 5)
    assert stuck["segments"][1]["release_delay"] is None                 # still on when the row ends


def test_grade_state_leaves_out_what_no_detector_could_see():
    n = 100
    t = _flags(n, (20, 60), (70, 90))
    obs = [not (20 <= i < 40 or 70 <= i < 90) for i in range(n)]         # first half of segment 1, all of segment 2
    g = E.grade_state(t, obs, _flags(n, (40, 60)), 5)
    (seg,) = g["segments"]
    assert (seg["first_observable"], seg["onset_delay"], seg["detected"]) == (40, 0, True)
    assert g["unobservable"] == [{"start": 70, "end": 90, "on": False}]
    assert (g["fn"], g["unobs_pos"], g["unobs_on"]) == (0, 40, 0)
    early = E.grade_state(t, obs, _flags(n, (30, 60), (75, 80)), 5)
    assert early["segments"][0]["onset_delay"] == -10 and early["unobs_on"] == 15
    assert early["unobservable"][0]["on"] and not early["false_segments"]
    gap = E.grade_state(_flags(n, (20, 60)), [not 30 <= i < 35 for i in range(n)], _flags(n, (20, 30), (35, 60)), 5)
    assert gap["segments"][0]["dropouts"] == 0                           # the gap had no evidence: not a dropout


def test_grade_alert_classifies_onsets_and_measures_the_delay():
    n = 400
    t = _flags(n, (100, 200))
    obs = [True] * n
    latched = E.grade_alert(t, obs, _flags(n, (160, 200)), FPS, 40, 100)
    pulse = E.grade_alert(t, obs, _flags(n, (160, 161)), FPS, 40, 100)
    for g in (latched, pulse):                                           # only the onset counts
        assert g["onsets"] == [160] and (g["during"], g["after"], g["unprovoked"]) == (1, 0, 0)
        assert g["segments"][0]["alerts"] == [160] and g["segments"][0]["delay"] == 60
    g = E.grade_alert(t, obs, _flags(n, (50, 52), (150, 152), (170, 172), (230, 232), (280, 282), (390, 392)), FPS, 40, 100)
    assert (g["during"], g["after"], g["unprovoked"]) == (3, 1, 2)       # 230 is within the grace, 280 within lookback
    assert g["segments"][0]["alerts"] == [150, 170, 230] and g["truth_frames"] == 100
    hidden = E.grade_alert(t, [i >= 150 for i in range(n)], _flags(n, (190, 195)), FPS, 40, 100)
    assert hidden["segments"][0]["delay"] == 40                          # from the first frame with evidence
    none = E.grade_alert([False] * n, obs, _flags(n, (10, 12)), FPS, 40, 100)
    assert none["truth_frames"] == 0 and none["unprovoked"] == 1 and not none["segments"]


def test_grade_alert_against_the_sheets_own_rule():
    n = 600
    rule = {"type": "continuous", "threshold_s": 2.0}                    # 40 frames; due at start + 39
    t = _flags(n, (0, 20), (100, 141), (200, 300), (400, 500), (520, 580))
    alerts = _flags(n, (10, 11), (240, 241), (410, 411))
    g = E.grade_alert(t, [True] * n, alerts, FPS, 10, 50, rule, rule_tol=5)
    v = [(s.get("verdict"), s.get("latency")) for s in g["segments"]]
    assert v == [("false", None), ("borderline", None), ("hit", 1), ("early", None), ("miss", None)]
    quiet = E.grade_alert(t, [True] * n, [False] * n, FPS, 10, 50, rule, rule_tol=5)
    assert [s["verdict"] for s in quiet["segments"]] == ["silent", "borderline", "miss", "miss", "miss"]
    assert "verdict" not in E.grade_alert(t, [True] * n, alerts, FPS, 10, 50)["segments"][0]


# ---- grading whole answer sheets ------------------------------------------------------------------------
def test_oracle_sheet_gets_full_marks(built, tmp_path):
    d, meta = built
    rep = E.grade(d, E.oracle_answers(d, tmp_path / "oracle"))
    assert rep["rows_answered"] == rep["rows_total"] == 5 and not rep["rows_missing"] and not rep["concepts_not_answered"]
    for c in E.CONCEPTS:
        o = rep["summary"][c]["overall"]
        assert o["detected"] == o["segments"] and o["false_segments"] == 0 and o["fn"] == o["fp"] == 0 and o["dropouts"] == 0
    ph = rep["summary"]["phone_use"]["overall"]
    assert ph["segments"] == 4 and ph["frame_recall"] == 1.0 and ph["onset_delay_s"]["max"] == 0.0
    assert ph["release_delay_s"]["max"] == 0.0
    assert rep["summary"]["phone_use"]["by"]["category"]["NEG"]["segments"] == 0
    assert set(rep["summary"]["phone_use"]["by"]) == set(E.SLICES)
    assert not rep["intent"] or all(i["verdict"] == "not_answered" for i in rep["intent"])   # no alert signals
    path = E.write_report(rep, tmp_path / "report")
    text = path.read_text(encoding="utf-8")
    assert "## Reading the driver state" in text and "## Alerts" not in text and "obvious" not in text
    assert rep["findings"] == [] and "## Where to look" in text and "Nothing: no missed or false segment" in text
    assert json.loads((tmp_path / "report" / "report.json").read_text(encoding="utf-8"))["exam"] == "T-v1"


def test_blind_shifted_and_holed_sheets_score_as_built(built, tmp_path):
    d, meta = built
    truth = _truth_flags(d, meta, "phone_use")
    blind = _sheet(tmp_path / "blind", d, meta, PHONE, {k: {"phone": [False] * len(v)} for k, v in truth.items()})
    o = blind["summary"]["phone"]["overall"]
    assert (o["detected"], o["segments"], o["false_segments"], o["frame_recall"]) == (0, 4, 0, 0.0)
    want = [(r, "phone", "missed", s / FPS, e / FPS) for r, v in truth.items() for s, e in E.segments(v)]
    assert [(f["row_id"], f["signal"], f["what"], f["start_s"], f["end_s"]) for f in blind["findings"]] == want
    r0, _, _, s0, e0 = want[0]
    text = E.report_markdown(blind)
    assert f"| `{r0}` | `phone` | missed | {s0:.2f}-{e0:.2f} |" in text and "key/answer_key.md" in text
    assert "1 more in `report.json`" in E.report_markdown({**blind, "grading": {**blind["grading"], "report_findings_max": 3}})
    k = 4
    shifted = _sheet(tmp_path / "shift", d, meta, PHONE, {r: {"phone": [False] * k + v[:-k]} for r, v in truth.items()})
    o = shifted["summary"]["phone"]["overall"]
    assert o["detected"] == 4 and o["onset_delay_s"]["median"] == o["onset_delay_s"]["max"] == k / FPS
    assert o["release_delay_s"]["median"] == k / FPS and o["false_segments"] == 0
    pos = next(r["row_id"] for r in meta["rows"] if r["entry"] == "POS-PHN-TEST")
    holed = {r: {"phone": list(v)} for r, v in truth.items()}
    start = truth[pos].index(True)
    for i in (start + 20, start + 21, start + 40):                       # two holes in the 3 s call
        holed[pos]["phone"][i] = False
    rep = _sheet(tmp_path / "hole", d, meta, PHONE, holed)
    assert rep["summary"]["phone"]["overall"]["dropouts"] == 2 and rep["summary"]["phone"]["overall"]["fn"] == 3
    assert [(f["row_id"], f["what"]) for f in rep["findings"]] == [(pos, "fell off 2 time(s) during the behaviour")]
    assert rep["concepts_not_answered"] == sorted(set(E.CONCEPTS) - {"phone_use"})


def test_alert_sheet_intent_rule_and_sweep(built, tmp_path):
    d, meta = built
    truth = _truth_flags(d, meta, "phone_use")
    need = int(0.75 * FPS)                                               # toy rule of the test: 0.75 s of phone use

    def alerts(v, extra=()):
        run, out = 0, []
        for x in v:
            run = run + 1 if x else 0
            out.append(run >= need)
        for i in extra:
            out[i] = True
        return out
    sig = {"phone_alert": {"concept": "phone_use", "kind": "alert", "rule": {"type": "continuous", "threshold_s": 0.75}}}
    rep = _sheet(tmp_path / "ok", d, meta, sig, {r: {"phone_alert": alerts(v)} for r, v in truth.items()})
    o = rep["summary"]["phone_alert"]["overall"]
    assert (o["alerts"], o["during"], o["unprovoked"], o["segments_alerted"], o["segments"]) == (3, 3, 0, 3, 4)
    assert (o["rows_without_behaviour"], o["rows_without_behaviour_alerted"]) == (1, 0)
    assert o["delay_s"]["median"] == (need - 1) / FPS
    assert {k: o["rule"][k] for k in ("hit", "miss", "early", "false", "silent")} == \
        {"hit": 3, "miss": 0, "early": 0, "false": 0, "silent": 1}
    assert o["rule"]["latency_s"]["max"] == 0.0
    assert sorted((i["entry"], i["verdict"]) for i in rep["intent"]) == [("NEG-ALL-TEST", "pass"), ("POS-PHN-TEST", "pass")]
    (sw,) = rep["sweeps"]
    assert (sw["entry"], sw["param"], sw["changes"], sw["last_before"], sw["first_after"]) == ("BND-PHN-TEST", "call_s", 1, 0.5, 1.0)
    assert [(p["value"], p["responded"]) for p in sw["points"]] == [(0.5, False), (1.0, True), (2.0, True)]
    text = E.report_markdown(rep)
    assert "## Alerts" in text and "own rule" in text and "0.5 .. 1" in text and "pass 2, fail 0" in text
    neg = next(r["row_id"] for r in meta["rows"] if r["entry"] == "NEG-ALL-TEST")
    pos = next(r["row_id"] for r in meta["rows"] if r["entry"] == "POS-PHN-TEST")
    bad = {r: {"phone_alert": alerts(v)} for r, v in truth.items()}
    bad[neg]["phone_alert"][10] = True                                   # alert where the driver never has a phone
    bad[pos]["phone_alert"] = [False] * len(truth[pos])                  # and none during the 3 s call
    rep = _sheet(tmp_path / "bad", d, meta, sig, bad)
    o = rep["summary"]["phone_alert"]["overall"]
    assert o["rows_without_behaviour_alerted"] == 1 and o["unprovoked"] == 1 and o["rule"]["miss"] == 1
    assert o["rule"]["false_outside_behaviour"] == 1
    (s, e), = E.segments(truth[pos])
    assert {(f["row_id"], f["what"], f["start_s"], f["end_s"]) for f in rep["findings"]} == {
        (neg, "alert without the behaviour", 10 / FPS, None), (pos, "own rule: no alert", s / FPS, e / FPS)}
    assert f"| `{neg}` | `phone_alert` | alert without the behaviour | 0.50 |" in E.report_markdown(rep)
    assert sorted((i["entry"], i["verdict"]) for i in rep["intent"]) == [("NEG-ALL-TEST", "fail"), ("POS-PHN-TEST", "fail")]
    assert "| `" + neg + "` | no alert | phone_alert x1 |" in E.report_markdown(rep)
    other = {"yawn_alert": {"concept": "yawning", "kind": "alert"}}
    rep = _sheet(tmp_path / "other", d, meta, other, {r: {"yawn_alert": [False] * len(v)} for r, v in truth.items()})
    assert sorted((i["entry"], i["verdict"]) for i in rep["intent"]) == [("NEG-ALL-TEST", "pass"), ("POS-PHN-TEST", "not_answered")]


def test_evidence_override_changes_what_counts(built, tmp_path):
    d, meta = built
    truth = _truth_flags(d, meta, "phone_use")
    rows = {r: {"phone": [False] * len(v)} for r, v in truth.items()}
    sig = {"phone": {"concept": "phone_use", "kind": "state", "evidence": ["cigarette"]}}    # never observable here
    rep = _sheet(tmp_path, d, meta, sig, rows)
    o = rep["summary"]["phone"]["overall"]
    assert (o["segments"], o["unobservable_segments"], o["fn"], o["unobs_pos"]) == (0, 4, 0, sum(map(sum, truth.values())))
    assert rep["signals"]["phone"]["evidence"] == ["cigarette"]


# ---- answer-sheet checks -------------------------------------------------------------------------------
def test_answer_sheet_problems_are_refused(built, tmp_path):
    d, meta = built
    truth = _truth_flags(d, meta, "phone_use")
    rows = {r: {"phone": v} for r, v in truth.items()}
    first = meta["rows"][0]["row_id"]

    def sheet(name, signals=PHONE, meta_=meta, rows_=rows):
        return E.write_answers(tmp_path / name, meta_, signals, rows_)
    with pytest.raises(E.ExamError, match="another build"):
        E.grade(d, sheet("fp", meta_={**meta, "fingerprint_id": "0000"}))
    with pytest.raises(E.ExamError, match="exam is 'B'"):
        E.grade(d, sheet("name", meta_={**meta, "exam": "B"}))
    for bad, msg in (({"concept": "sleepy", "kind": "state"}, "concept must be one of"),
                     ({"concept": "phone_use", "kind": "level"}, "kind must be state or alert"),
                     ({"concept": "phone_use", "kind": "state", "evidence": ["radar"]}, "evidence must list"),
                     ({"concept": "phone_use", "kind": "state", "rule": {"type": "continuous", "threshold_s": 1}}, "only an alert"),
                     ({"concept": "phone_use", "kind": "alert", "rule": {"type": "window", "threshold_s": 1}}, "rule must be"),
                     ({"concept": "phone_use", "kind": "alert", "rule": {"type": "continuous", "threshold_s": 0}}, "rule must be"),
                     ({"concept": "phone_use", "kind": "state", "note": "x"}, "unknown keys")):
        with pytest.raises(E.ExamError, match=msg):
            E.grade(d, sheet("sig", signals={"phone": bad}))
    a = sheet("short")
    p = a / f"{first}.jsonl"
    lines = p.read_text(encoding="utf-8").splitlines()
    p.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(E.ExamError, match="one line per frame"):
        E.grade(d, a)
    for edit, msg in ((lambda r: r.update(frame_id=7), "frame_id 7, expected 0"),
                      (lambda r: r["signals"].update(extra=1), "exactly the declared"),
                      (lambda r: r["signals"].update(phone=2), "expected 0 or 1"),
                      (lambda r: r["signals"].update(phone="on"), "expected 0 or 1")):
        rec = json.loads(lines[0])
        edit(rec)
        p.write_text("\n".join([json.dumps(rec)] + lines[1:]) + "\n", encoding="utf-8")
        with pytest.raises(E.ExamError, match=msg):
            E.grade(d, a)
    stray = sheet("stray")
    (stray / "NOT-A-ROW.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(E.ExamError, match="not in this exam"):
        E.grade(d, stray)
    with pytest.raises(E.ExamError, match="answer.yaml not found|not found"):
        E.grade(d, tmp_path / "nowhere")
    part = sheet("part", rows_={first: rows[first]})                     # some rows only: graded, the rest listed
    rep = E.grade(d, part)
    assert rep["rows_answered"] == 1 and len(rep["rows_missing"]) == 4
    with pytest.raises(E.ExamError, match="no answer file"):
        E.grade(d, sheet("none", rows_={}))


# ---- the example adapter and the CLI -------------------------------------------------------------------
def test_example_adapter_answers_in_the_format(built, tmp_path):
    import runpy
    d, meta = built
    mod = runpy.run_path(str(Path(__file__).parent.parent / "examples" / "naive_reader.py"))
    mod["main"](str(d), str(tmp_path / "answers"))
    rep = E.grade(d, tmp_path / "answers")
    assert rep["rows_answered"] == 5 and set(rep["signals"]) == set(mod["SIGNALS"])
    o = rep["summary"]["phone"]["overall"]                               # exact profile: the phone box is the label
    assert o["detected"] == o["segments"] == 4 and o["frame_recall"] == 1.0 and o["false_segments"] == 0
    assert rep["summary"]["looking_away"]["overall"]["false_segments"] == 0


def test_cli_exam_build_oracle_grade(tmp_path, capsys):
    import yaml
    from dms_model_simulator.cli import main
    ex = tmp_path / "mini.yaml"
    ex.write_text(yaml.safe_dump({"name": "mini", "intent": True, "entries": ["POS-PHN-BASIC-CALL"],
                                  "expand": {"level": ["small"], "profile": ["exact"], "emulator_seeds": 1}}), encoding="utf-8")
    main(["exam", "list"])
    assert "A-v1" in capsys.readouterr().out
    main(["exam", "build", str(ex), "--out", str(tmp_path / "x"), "--camera", "steering_column", "--include-face"])
    assert "exam mini" in capsys.readouterr().out
    meta = E.load_built(tmp_path / "x")
    assert meta["cameras"] == ["steering_column"] and meta["pipeline"]["include_face"] is True
    q = tmp_path / "x" / "questions" / meta["rows"][0]["row_id"] / "pipeline.jsonl"
    assert any(o["label"] == "face" for o in json.loads(q.read_text(encoding="utf-8").splitlines()[0])["od"])
    main(["exam", "oracle", str(tmp_path / "x"), "--out", str(tmp_path / "a")])
    main(["exam", "grade", str(tmp_path / "x"), str(tmp_path / "a"), "--out", str(tmp_path / "r")])
    assert "graded 1 of 1 rows, 6 signals" in capsys.readouterr().out
    assert (tmp_path / "r" / "report.md").exists() and (tmp_path / "r" / "report.json").exists()
    with pytest.raises(SystemExit, match="already holds an exam"):
        main(["exam", "build", str(ex), "--out", str(tmp_path / "x")])
    with pytest.raises(SystemExit, match="not found"):
        main(["exam", "build", "no-such-exam", "--out", str(tmp_path / "y")])

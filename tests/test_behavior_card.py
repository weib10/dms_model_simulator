"""Behaviour card: scenario timeline (segment frames / times) and the injected model errors of a run."""
import json
import pytest
from dms_model_simulator.behavior_card import _runs, card, injected_errors, markdown, timeline, write_card
from dms_model_simulator.simulate import Simulation

CAM = "steering_column"
SC = {"name": "card", "description": "road, call, road",
      "segments": [{"target": "road_ahead", "duration": 0.5},
                   {"target": "road_ahead", "duration": 1.0, "hand": "call", "phone": "call"},
                   {"target": "passenger", "duration": 0.75, "head_pose": [30, 0, 0]}]}
CLEAN = {"pipeline": {"drop_frame_prob": 0.0}, "detection": {"false_positives": {"rate_per_frame": 0.0}}}
FORCED_EYE_FLIP = {"eye_state_flip": {"enabled": True, "mean_frames": 1000,
                                      "open_to_closed": {"p_base": 1.0, "p_gaze_down": 0.0, "p_small": 0.0},
                                      "closed_to_open": {"p_base": 1.0}}}


def _sim(ov=None, sc=SC, fps=20, **kw):
    s = Simulation(sc, CAM, seed=0, fps=fps, emulator_overrides=ov, **kw)
    s.run()
    return s


def test_runs():
    assert _runs([]) == [] and _runs([False, False]) == []
    assert _runs([True, True, False, True, False, False, True]) == [(0, 1), (3, 3), (6, 6)]


def test_timeline_matches_the_scenario():
    s = _sim()
    tl = timeline(s)
    assert [(x["index"], x["t0"], x["t1"], x["f0"], x["f1"]) for x in tl] == [
        (0, 0.0, 0.5, 0, 9), (1, 0.5, 1.5, 10, 29), (2, 1.5, 2.25, 30, 44)]
    assert tl[-1]["f1"] == len(s.records) - 1
    assert tl[1]["keys"] == {"target": "road_ahead", "hand": "call", "phone": "call"}       # duration left out
    assert tl[2]["keys"] == {"target": "passenger", "head_pose": [30, 0, 0]}
    assert "call" in tl[1]["text"] and "passenger" in tl[2]["text"] and "30" in tl[2]["text"]
    assert tl[0]["text"] != tl[1]["text"]
    for seg, x in zip(SC["segments"], tl):                               # each listed frame lies inside its segment
        for f in s.frames[x["f0"]:x["f1"] + 1]:
            assert x["t0"] - 1e-9 <= f.t < x["t1"]
            assert f.target_name == seg["target"]


def test_timeline_at_another_fps_and_unaligned_durations():
    sc = {"name": "odd", "segments": [{"target": "road_ahead", "duration": 0.35},
                                      {"target": "road_ahead", "duration": 0.4, "hand": "at_ear"}]}
    s = _sim(sc=sc, fps=30)
    tl = timeline(s)
    # frame k is at t = k / 30: segment 0 holds the frames before 0.35 s (0..10), segment 1 the rest up to 0.75 s
    assert [(x["f0"], x["f1"]) for x in tl] == [(0, 10), (11, 22)]
    assert len(s.records) == 23
    for x in tl:
        assert all(x["t0"] - 1e-9 <= f.t < x["t1"] for f in s.frames[x["f0"]:x["f1"] + 1])


def test_clean_run_has_no_injected_errors():
    s = _sim(CLEAN)
    e = injected_errors(s.records)
    assert e["frames"] == len(s.records) and e["latency_frames"] == 0
    assert e["dropped"] == [] and e["class_flip"] == {} and e["false_positive"] == {}


def test_injected_errors_find_dropped_frames():
    s = _sim({"pipeline": {"drop_frame_prob": 1.0}})
    e = injected_errors(s.records)
    assert e["dropped"] == [(0, len(s.records) - 1)]
    assert e["missed"] == {} and e["class_flip"] == {}                   # a dropped frame is not a missed box


def test_injected_errors_find_missed_boxes():
    s = _sim({**CLEAN, "detection": {**CLEAN["detection"], "phone": {"p_max": 0.0}}})
    e = injected_errors(s.records)
    phone = [i for i, r in enumerate(s.records) if any(d["class"] == "phone" for d in r["truth"]["detections"])]
    assert phone and e["missed"]["phone"] == _runs([i in phone for i in range(len(s.records))])
    assert not any(d["class"] == "phone" for r in s.records for d in r["pred"]["detections"])


def test_injected_errors_find_class_flips():
    s = _sim({**CLEAN, **FORCED_EYE_FLIP})
    e = injected_errors(s.records)
    flags = ["eye" in r["pred"].get("class_flip", []) for r in s.records]
    assert any(flags) and e["class_flip"] == {"eye": _runs(flags)}
    assert "mouth" not in e["class_flip"]


def test_injected_errors_latency():
    s = _sim({**CLEAN, "pipeline": {"drop_frame_prob": 0.0, "latency_frames": 2}})
    e = injected_errors(s.records)
    assert [r["pred"]["source_frame"] for r in s.records[:4]] == [None, None, 0, 1]
    assert e["latency_frames"] == 2
    assert e["dropped"] == [(0, 1)]                                      # the latency fill has no output


def test_card_json_and_markdown(tmp_path):
    s = _sim({**CLEAN, **FORCED_EYE_FLIP}, emulator_seed=7)
    row = {"row_id": "BND-PHN-X__abc__dash_right", "entry": "BND-PHN-X", "title": "call", "summary": "a short call",
           "params": {"call_s": 1.0}, "profile": "eye_flip"}
    p = write_card(s, tmp_path / "c", row=row)
    assert p == tmp_path / "c" / "card.md" and p.exists()
    cj = json.loads((tmp_path / "c" / "card.json").read_text(encoding="utf-8"))
    assert cj == json.loads(json.dumps(card(s, row)))                    # same content as card()
    assert cj["timeline"] == json.loads(json.dumps(timeline(s)))
    assert cj["errors"] == json.loads(json.dumps(injected_errors(s.records)))
    assert (cj["row_id"], cj["entry"], cj["params"], cj["profile"]) == (row["row_id"], "BND-PHN-X", {"call_s": 1.0},
                                                                        "eye_flip")
    assert (cj["scenario"], cj["description"], cj["camera"], cj["fps"]) == ("card", "road, call, road", CAM, 20.0)
    assert (cj["frames"], cj["duration_s"], cj["seed"], cj["emulator_seed"]) == (45, 2.25, 0, 7)
    md = p.read_text(encoding="utf-8")
    assert md == markdown(cj) and md.startswith(f"# 行為卡：{row['row_id']}")
    assert "| 2 | 1.5–2.25 | 30–44 |" in md and "`eye`" in md and "`eye_flip`" in md
    # without a row (dms-sim export / demo): titled by the scenario
    p2 = write_card(_sim(CLEAN), tmp_path / "d")
    md2 = p2.read_text(encoding="utf-8")
    assert md2.startswith("# 行為卡：card") and "BND-PHN-X" not in md2


def test_markdown_caps_the_listed_runs():
    s = _sim({"pipeline": {"drop_frame_prob": 0.5}}, sc={"name": "long", "segments": [
        {"target": "road_ahead", "duration": 10.0}]})
    c = card(s)
    n_runs = len(c["errors"]["dropped"])
    assert n_runs > 12
    assert f"另有 {n_runs - 12} 段較短的未列出" in markdown(c)

"""Reproducibility: run fingerprint, LF output on every OS, and `catalog verify` (compare two runs)."""
import json
from pathlib import Path
import pytest
from dms_model_simulator import catalog as C
from dms_model_simulator import repro

PR = {"nominal": {}}


def _entry():
    return {"id": "BND-PHN-REPRO", "title": "t", "status": "ready", "behaviors": ["V08"],
            "params": {"call_s": {"levels": {"small": 0.5, "medium": 0.5, "large": 0.5}}},
            "template": {"segments": [{"target": "road_ahead", "duration": 0.5},
                                      {"target": "road_ahead", "duration": "${call_s}", "hand": "call", "phone": "call"}]},
            "variation": {"cameras": ["dash_right"], "emulator_seeds": [0]}}


def _run(d: Path, entries=None):
    C.expand(entries or [_entry()], d, profiles=PR, levels=["medium"])
    C.run(d, log=lambda *_: None)
    return d


def test_fingerprint_records_code_configs_and_entries():
    es = [e for e in C.load_entries() if e["id"] == "POS-PHN-BASIC-CALL"]
    fp = repro.fingerprint(es)
    assert {"package", "git_commit", "numpy", "python", "platform", "configs", "catalog", "entries"} <= set(fp)
    assert "emulator.yaml" in fp["configs"] and "behaviors.yaml" in fp["catalog"]
    assert list(fp["entries"]) == ["POS-PHN-BASIC-CALL"]
    assert repro.fingerprint_id(fp) == repro.fingerprint_id(json.loads(json.dumps(fp)))
    other = {**fp, "configs": {**fp["configs"], "emulator.yaml": "x"}}
    diff, _ = repro.compare_fingerprints(fp, other)
    assert diff == [f"configs/emulator.yaml: {fp['configs']['emulator.yaml']} vs x"]
    assert repro.fingerprint_id(other) != repro.fingerprint_id(fp)
    diff, info = repro.compare_fingerprints(fp, {**fp, "platform": "Elsewhere arm64", "python": "0"})
    assert diff == [] and len(info) == 2                                  # informational only


def test_text_hash_ignores_line_endings(tmp_path):
    (tmp_path / "a").write_bytes(b"x\r\ny\r\n")
    (tmp_path / "b").write_bytes(b"x\ny\n")
    assert repro.text_sha(tmp_path / "a") == repro.text_sha(tmp_path / "b")


def test_run_files_use_lf_and_carry_the_fingerprint(tmp_path):
    d = _run(tmp_path / "r")
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert man["fingerprint_id"] == repro.fingerprint_id(man["fingerprint"])
    assert all(r["fingerprint_id"] == man["fingerprint_id"] for r in man["rows"])
    texts = [p for p in d.rglob("*") if p.is_file() and p.suffix in (".jsonl", ".json", ".yaml", ".md", ".tsv")]
    assert texts and not [p.name for p in texts if b"\r" in p.read_bytes()]
    (card,) = d.rglob("card.md")
    assert f"版本指紋：`{man['fingerprint_id']}`" in card.read_text(encoding="utf-8")


def test_verify_same_and_different_runs(tmp_path):
    a, b = _run(tmp_path / "a"), _run(tmp_path / "b")
    res = repro.compare_runs(a, b)
    assert res["same"] and res["rows_compared"] == 1 and res["worst_number_diff"] == 0.0
    # a last-digit float difference (as between operating systems) is within the tolerance
    (truth,) = b.rglob("*.truth.jsonl")
    lines = truth.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["head_pose"]["yaw"] += 1e-13
    truth.write_text("\n".join([json.dumps(rec)] + lines[1:]) + "\n", encoding="utf-8", newline="\n")
    res = repro.compare_runs(a, b)
    assert res["same"] and 0 < res["worst_number_diff"] < 1e-9
    # a real change is reported
    rec["head_pose"]["yaw"] += 0.5
    rec["od"] = rec["od"][1:]
    truth.write_text("\n".join([json.dumps(rec)] + lines[1:]) + "\n", encoding="utf-8", newline="\n")
    res = repro.compare_runs(a, b)
    assert not res["same"] and len(res["rows_differ"]) == 1
    (probs,) = res["rows_differ"].values()
    assert any("head_pose/yaw" in p for p in probs) and any("items" in p for p in probs)


def test_verify_ignores_row_pictures(tmp_path):
    a, b = _run(tmp_path / "a"), _run(tmp_path / "b")
    C.run(b, log=lambda *_: None, timeline=True, gif=True)
    assert list(b.rglob("timeline.png")) and list(b.rglob("preview.gif"))
    assert repro.compare_runs(a, b)["same"]


def test_verify_reports_a_changed_entry(tmp_path):
    a = _run(tmp_path / "a")
    e = _entry()
    e["_path"] = str(tmp_path / "e.yaml")
    Path(e["_path"]).write_text("changed", encoding="utf-8")
    b = _run(tmp_path / "b", [e])
    res = repro.compare_runs(a, b)
    assert not res["same"] and any(s.startswith("entries/BND-PHN-REPRO") for s in res["fingerprint_diff"])


def test_cli_verify(tmp_path, capsys):
    from dms_model_simulator import cli
    a, b = _run(tmp_path / "a"), _run(tmp_path / "b")
    cli.main(["catalog", "verify", str(a), str(b)])
    assert "SAME TEST, SAME OUTPUT" in capsys.readouterr().out
    (pipe,) = b.rglob("*.pipeline.jsonl")
    pipe.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit):
        cli.main(["catalog", "verify", str(a), str(b)])
    assert "DIFFERENT" in capsys.readouterr().out

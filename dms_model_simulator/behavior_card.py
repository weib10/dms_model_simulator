"""Behaviour card of one simulated run: what the driver does, when, and which model errors were injected.

The card is written for whoever checks a post-processing program on the exported model output: they run their
program on the frames, then compare its alerts with the timeline here. Two parts:

- timeline: one line per scenario segment, t0-t1 s (frames f0-f1), in plain words built from the segment's keys;
- injected model errors: per kind and class, the runs of frames where the simulated model output differs from
  the truth on purpose (dropped frames, missed boxes, open / closed class flips, box jumps, false positives) and
  the latency.

`write_card(sim, out_dir)` writes card.md (readable) and card.json (same content, machine readable).
"""
from __future__ import annotations
import json
from pathlib import Path

SEGMENT_WORDS = {   # segment key -> how it reads on the card
    "target": "看 {}",
    "head_pose": "頭 yaw/pitch/roll = {}°",
    "eye_openness": "眼睛開合度 {}",
    "closure": "閉眼 {}",
    "yawn": "打哈欠",
    "blink": "眨眼 {}",
    "mouth_openness": "嘴巴開合度 {}",
    "hand": "手：{}",
    "phone": "手機：{}",
    "cigarette": "菸：{}",
    "seatbelt": "安全帶：{}",
    "held": "嘴邊拿著：{}",
}
DEFAULT_STATE = {"hand": "wheel", "phone": "none", "cigarette": "none", "seatbelt": "on", "held": "none"}
MAX_RUNS_LISTED = 12


def _runs(flags: list[bool]) -> list[tuple[int, int]]:
    """[(first, last)] frame indices of each run of True."""
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def _seg_text(seg: dict) -> str:
    parts = []
    for k, fmt in SEGMENT_WORDS.items():
        if k not in seg or k == "duration":
            continue
        v = seg[k]
        if k in DEFAULT_STATE and str(v) == DEFAULT_STATE[k]:
            continue
        if k == "yawn":
            if v:
                parts.append(fmt)
            continue
        if isinstance(v, dict):
            v = ", ".join(f"{a}={b}" for a, b in v.items())
        elif isinstance(v, list):
            v = "[" + ", ".join(str(x) for x in v) + "]"
        parts.append(fmt.format(v))
    return "；".join(parts) or "（預設狀態）"


def timeline(sim) -> list[dict]:
    """Per segment: {"index", "t0", "t1", "f0", "f1" (last frame), "keys" (the segment as written), "text"}."""
    out = []
    n = len(sim.frames)
    for i, (seg, s) in enumerate(zip(sim.scenario["segments"], sim.behavior.segments)):
        f0, f1 = int(s["start_frame"]), min(int(s["end_frame"]), n) - 1
        if f1 < f0:
            continue
        keys = {k: v for k, v in seg.items() if k != "duration"}
        out.append({"index": i, "t0": round(float(s["start"]), 3), "t1": round(float(s["end"]), 3),
                    "f0": f0, "f1": f1, "keys": keys, "text": _seg_text(seg)})
    return out


def injected_errors(records: list[dict]) -> dict:
    """Runs of frames per injected error kind: {"latency_frames", "dropped", "missed": {class: runs},
    "class_flip": {eye|mouth: runs}, "jump": {class: runs}, "false_positive": {class: runs}}; each run is
    [first frame, last frame]. A missed box = the class is in the truth detections but not in the model output
    of a frame that was not dropped (with latency, the output of an earlier frame is compared, as written)."""
    n = len(records)
    dropped = [bool(r["pred"].get("dropped")) for r in records]
    lat = max((r["frame"] - r["pred"]["source_frame"] for r in records
               if r["pred"].get("source_frame") is not None), default=0)
    truth_cls = sorted({d["class"] for r in records for d in r["truth"]["detections"]})
    missed, jump, fp = {}, {}, {}
    for c in truth_cls:
        flags = [not dropped[i] and any(d["class"] == c for d in r["truth"]["detections"])
                 and not any(d["class"] == c and not d.get("false_positive") for d in r["pred"]["detections"])
                 for i, r in enumerate(records)]
        if any(flags):
            missed[c] = _runs(flags)
    pred_cls = sorted({d["class"] for r in records for d in r["pred"]["detections"]})
    for c in pred_cls:
        jflags = [any(d["class"] == c and d.get("jump") for d in r["pred"]["detections"]) for r in records]
        if any(jflags):
            jump[c] = _runs(jflags)
        fflags = [any(d["class"] == c and d.get("false_positive") for d in r["pred"]["detections"]) for r in records]
        if any(fflags):
            fp[c] = _runs(fflags)
    flip = {}
    for key in ("eye", "mouth"):
        flags = [key in (r["pred"].get("class_flip") or []) for r in records]
        if any(flags):
            flip[key] = _runs(flags)
    return {"frames": n, "latency_frames": int(lat), "dropped": _runs(dropped), "missed": missed,
            "class_flip": flip, "jump": jump, "false_positive": fp}


def card(sim, row: dict | None = None) -> dict:
    """Card content of a finished Simulation (sim.run() called). row: optional catalog manifest row (entry, title,
    summary, params, profile)."""
    row = row or {}
    fps = float(sim.behavior.fps)
    return {
        "row_id": row.get("row_id"), "entry": row.get("entry"), "title": row.get("title", ""),
        "summary": row.get("summary", ""), "params": row.get("params", {}), "level": row.get("level"),
        "profile": row.get("profile"), "fingerprint_id": row.get("fingerprint_id"),
        "scenario": sim.scenario["name"], "description": sim.scenario.get("description", ""),
        "camera": sim.camera.name, "fps": fps, "frames": len(sim.records),
        "duration_s": round(len(sim.records) / fps, 3), "seed": sim.seed, "emulator_seed": sim.emulator_seed,
        "timeline": timeline(sim), "errors": injected_errors(sim.records),
    }


def _t(f: int, fps: float) -> str:
    return f"{f / fps:.2f}"


def _run_list(runs: list, fps: float) -> str:
    total = sum(b - a + 1 for a, b in runs)
    shown = sorted(runs, key=lambda r: r[0] - r[1])[:MAX_RUNS_LISTED]       # longest first
    txt = ", ".join(f"{_t(a, fps)}–{_t(b + 1, fps)} s（{b - a + 1} 幀）" for a, b in sorted(shown))
    more = f"，另有 {len(runs) - len(shown)} 段較短的未列出" if len(runs) > len(shown) else ""
    return f"共 {len(runs)} 段、{total} 幀：{txt}{more}"


def markdown(c: dict) -> str:
    fps = c["fps"]
    L = [f"# 行為卡：{c['row_id'] or c['scenario']}", ""]
    if c.get("entry"):
        L += [f"- 情境：`{c['entry']}` {c['title']}", f"- 說明：{c['summary']}" if c.get("summary") else None,
              f"- 難度：`{c['level']}`" if c.get("level") else None,
              f"- 參數：`{json.dumps(c['params'], ensure_ascii=False)}`"]
    if c.get("fingerprint_id"):
        L.append(f"- 版本指紋：`{c['fingerprint_id']}`（程式、設定、條目的雜湊；同一個指紋 + 同一個 row id = 同一個測試，見 manifest.json）")
    if c.get("profile"):
        L.append(f"- 誤差等級：`{c['profile']}`")
    L += [f"- 相機：`{c['camera']}`，{fps:g} fps，{c['frames']} 幀（{c['duration_s']:g} s）",
          f"- 種子：行為 {c['seed']}、模型誤差 "
          + ("（由行為種子推出）" if c["emulator_seed"] is None else str(c["emulator_seed"])), ""]
    L += ["## 行為時間軸", "", "時間是幀的時間 t = frame / fps；一段的結束時間不含在段內。", "",
          "| 段 | 時間 (s) | 幀 | 駕駛在做什麼 |", "|---|---|---|---|"]
    for s in c["timeline"]:
        L.append(f"| {s['index']} | {s['t0']:g}–{s['t1']:g} | {s['f0']}–{s['f1']} | {s['text']} |")
    e = c["errors"]
    L += ["", "## 注入的模型誤差", "",
          "模型輸出跟真值不同的地方，都是模擬器故意加的。後處理在這些時段的反應，要分開看。", ""]
    L.append(f"- 延遲：{e['latency_frames']} 幀" + ("（輸出比真值晚，開頭補空幀）" if e["latency_frames"] else ""))
    L.append(f"- 掉幀（整幀沒有輸出）：{_run_list(e['dropped'], fps) if e['dropped'] else '無'}")
    for title, key in (("漏框", "missed"), ("開閉類別判反", "class_flip"), ("框跳到別處", "jump"), ("誤報框", "false_positive")):
        if not e[key]:
            L.append(f"- {title}：無")
            continue
        L.append(f"- {title}：")
        for cls, runs in e[key].items():
            L.append(f"  - `{cls}`：{_run_list(runs, fps)}")
    L += ["", "## 檔案", "",
          "- `*.frames/labels/<frame>.yaml`：每幀的模型輸出（檔名 = frame index，`timestamp_msec` = round(t × 1000)）",
          "- `*.pipeline.jsonl`：同一份模型輸出的 JSONL；`*.truth.jsonl`：每幀真值；`*.jsonl`：真值與模型輸出並排，"
          "含誤差標記（`dropped`、`jump`、`false_positive`、`class_flip`）",
          "- `card.json`：這份卡片的機器可讀版"]
    return "\n".join(x for x in L if x is not None) + "\n"


def write_card(sim, out_dir, row: dict | None = None) -> Path:
    """Write <out_dir>/card.md and card.json; returns the card.md path."""
    c = card(sim, row)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "card.json").write_text(json.dumps(c, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    p = out / "card.md"
    p.write_text(markdown(c), encoding="utf-8", newline="\n")
    return p

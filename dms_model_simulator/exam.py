"""Exams for a post-processing program: build a question package from catalog entries, grade an answer sheet.

    dms-sim exam build A-v1 --out runs/A          # questions/ (model output) + key/ (truth, behaviour cards)
    <your adapter>  runs/A/questions -> answers/   # run your post-processing, write the answer sheet (docs/EXAM.md)
    dms-sim exam grade runs/A answers --out report

The answer sheet is in the exam's vocabulary, not the program's: the adapter declares named signals, each tied to
one of the six behaviour labels of the truth (CONCEPTS) and marked `state` (on while the driver does it) or `alert`
(raised once the program decides to), and gives every signal a 0 / 1 per frame. The grader lines these up with the
truth. It holds no alert threshold and no alert logic: what should raise an alert is either measured (how long
after the behaviour started did the alert come), read off the exam's design (rows built to be obvious), or checked
against a rule the answer sheet itself declares.

Settings of the grader (how wide the tolerance bands are, which evidence a label needs): exams/grading.yaml.
"""
from __future__ import annotations
import json
import math
import shutil
from pathlib import Path
from . import catalog as C
from .config import ROOT, load_yaml
from .emulator import OBSERVABLE_CHANNELS
from .metrics import segments

EXAMS_DIR = ROOT / "exams"
GRADING = EXAMS_DIR / "grading.yaml"
CONCEPTS = ("eyes_off_road", "eyes_closed", "phone_use", "yawning", "smoking", "seatbelt_off")
KINDS = ("state", "alert")
RULES = ("continuous",)
EXAM_KEYS = {"name", "title", "intent", "expand", "entries"}
EXPAND_KEYS = {"mode", "n", "seed", "level", "profile", "emulator_seeds", "camera", "variants", "max_rows"}
SLICES = ("category", "family", "level", "profile")


class ExamError(ValueError):
    pass


# ---- exam definitions (exams/<name>.yaml) -------------------------------------------------------------
def exam_path(name_or_path) -> Path:
    p = Path(name_or_path)
    if not p.is_file():
        p = EXAMS_DIR / f"{name_or_path}.yaml"
    if not p.is_file():
        raise ExamError(f"exam {name_or_path!r} not found (a YAML path, or one of: {', '.join(list_exams()) or 'none'})")
    return p


def list_exams() -> list[str]:
    return sorted(p.stem for p in EXAMS_DIR.glob("*.yaml") if p.name != GRADING.name)


def load_exam(name_or_path) -> dict:
    """An exam definition: {name, title, intent, expand: `catalog expand` arguments, entries: [catalog entry ids]}.
    The entry list is frozen in the file: entries added to the catalog later do not join an existing exam."""
    exam = load_yaml(exam_path(name_or_path)) if not isinstance(name_or_path, dict) else dict(name_or_path)
    errs = [f"unknown key {k!r}" for k in sorted(set(exam) - EXAM_KEYS)]
    errs += [f"missing {k}" for k in ("name", "entries") if not exam.get(k)]
    errs += [f"expand: unknown key {k!r}" for k in sorted(set(exam.get("expand") or {}) - EXPAND_KEYS)]
    if len(set(exam.get("entries") or [])) != len(exam.get("entries") or []):
        errs.append("entries: duplicate ids")
    if errs:
        raise ExamError(f"exam {exam.get('name', name_or_path)!r}: " + "; ".join(errs))
    return exam


def _row_intent(exam: dict, row: dict, behaviors: list[str], known: dict) -> tuple[str | None, str | None]:
    """(main concept, intent) of a row. Main concept: the truth label of the entry's first behaviour when that is
    a violation. Intent (only in exams marked `intent: true`, whose rows are built to be far from any reasonable
    threshold): "alert" for a POS row with a main concept, "no_alert" for a NEG row without any violation
    behaviour, else None. It is the exam author's design, not a specification."""
    kinds = [known.get(b, {}).get("kind") for b in behaviors]
    first = known.get(behaviors[0], {}) if behaviors else {}
    concept = (first.get("expect") or {}).get("label") if first.get("kind") == "violation" else None
    intent = None
    if exam.get("intent"):
        if row["category"] == "POS" and concept:
            intent = "alert"
        elif row["category"] == "NEG" and "violation" not in kinds:
            intent = "no_alert"
    return concept, intent


# ---- build --------------------------------------------------------------------------------------------
def build(exam, out_dir, jobs: int = 1, cameras: list[str] | None = None, layout=None,
          include_face: bool | None = None, labels: str | None = None, seed: int | None = None,
          force: bool = False, log=print, entries: list[dict] | None = None) -> dict:
    """Simulate an exam and lay it out for whoever sits it. Writes under out_dir:

        exam.json                      what the exam is: name, fingerprint, one record per row
        questions/<row_id>/            what the post-processing may read
            pipeline.jsonl             model output, one record per frame (docs/SPEC.md)
            labels/                    the same frames as frame_yaml, only with `layout`
            row.json                   fps, frame count, camera, road_box (what counts as looking at the road)
        key/                           the answer side: a catalog run (manifest, truth, full JSONL, behaviour cards)

    exam: a name in exams/, a YAML path or a definition dict. cameras / seed override the definition's
    `expand.camera` / `expand.seed` (a post-processing is built for one camera position; a private seed with
    `mode: random` gives a held-out version). layout, include_face, labels only change how the model output is
    written (as for `catalog run`): the truth and the row ids stay the same. entries: the catalog entries to pick
    from (default: the repo's catalog). Returns the content of exam.json."""
    exam = load_exam(exam)
    out = Path(out_dir)
    if (out / "exam.json").exists() or (out / "key").exists():
        if not force:
            raise ExamError(f"{out} already holds an exam (use --force to rebuild)")
        for sub in ("questions", "key"):
            shutil.rmtree(out / sub, ignore_errors=True)
        (out / "exam.json").unlink(missing_ok=True)
    by_id = {e["id"]: e for e in (entries if entries is not None else C.load_entries())}
    missing = [i for i in exam["entries"] if i not in by_id]
    if missing:
        raise ExamError(f"exam {exam['name']}: entries not in the catalog: {missing}")
    ex = dict(exam.get("expand") or {})
    key = out / "key"
    man = C.expand([by_id[i] for i in exam["entries"]], key, ex.get("mode", "boundary"), int(ex.get("n", 5)),
                   int(seed if seed is not None else ex.get("seed", 0)), cameras=cameras or ex.get("camera"),
                   variants=ex.get("variants"), emulator_seeds=ex.get("emulator_seeds"),
                   profile_names=ex.get("profile"), levels=ex.get("level"), max_rows=int(ex.get("max_rows", 2000)))
    if man["skipped_planned"]:
        raise ExamError(f"exam {exam['name']}: planned entries cannot be simulated: {man['skipped_planned']}")
    done = {d["row_id"]: d for d in C.run(key, jobs=jobs, log=log, layout=layout, include_face=include_face,
                                          labels=labels)}
    known = C.load_behaviors()
    rows = []
    for r in man["rows"]:
        info = done[r["row_id"]]
        q = out / "questions" / r["row_id"]
        q.mkdir(parents=True, exist_ok=True)
        rd = key / "rows" / r["row_id"]
        shutil.copyfile(_one(rd, "*.pipeline.jsonl"), q / "pipeline.jsonl")
        if layout is not None:
            shutil.copytree(Path(info["frames_dir"]) / "labels", q / "labels")
        cam = json.loads(_one(rd, "*__camera.json").read_text(encoding="utf-8"))
        qrow = {"row_id": r["row_id"], "fps": r["fps"], "frames": info["frames"], "camera": r["camera"],
                "image_size": [cam.get("width"), cam.get("height")], "road_box": info["road_box"]}
        _write_json(q / "row.json", qrow)
        behaviors = list(by_id[r["entry"]].get("behaviors") or [])
        concept, intent = _row_intent(exam, r, behaviors, known)
        rows.append({"row_id": r["row_id"], "entry": r["entry"], "category": r["category"], "family": r["family"],
                     "behaviors": behaviors, "main_concept": concept, "intent": intent, "level": r["level"],
                     "params": r["params"], "camera": r["camera"], "variant": r["variant"], "seed": r["seed"],
                     "emulator_seed": r["emulator_seed"], "profile": r["profile"],
                     # same driver, same camera: rows that differ only in the model errors share it
                     "pair_id": r["row_id"].rsplit("__", 2)[0],
                     "fps": r["fps"], "frames": info["frames"], "title": r["title"]})
    meta = {"exam": exam["name"], "title": exam.get("title", ""), "intent": bool(exam.get("intent")),
            "fingerprint_id": man["fingerprint_id"], "expand": {"mode": man["mode"], "seed": man["seed"]},
            "cameras": sorted({r["camera"] for r in rows}), "profiles": sorted({r["profile"] for r in rows}),
            "pipeline": {"include_face": include_face, "labels": labels,
                         "layout_sha256": next(iter(done.values()), {}).get("layout_sha256")},
            "rows_total": len(rows), "frames_total": sum(r["frames"] for r in rows), "rows": rows}
    _write_json(out / "exam.json", meta)
    (key / "answer_key.md").write_text(answer_key_markdown(out), encoding="utf-8", newline="\n")
    return meta


def _spans(flags: list[bool], fps: float) -> str:
    """Runs of true frames in seconds, "8.00-12.00, 15.50-15.60": from the time of a run's first frame to the time
    of the first frame after it."""
    return ", ".join(f"{s / fps:.2f}-{e / fps:.2f}" for s, e in segments(flags))


def answer_key_markdown(exam_dir, grading=None) -> str:
    """The truth of a built exam as a page to read (key/answer_key.md): per row, when each behaviour label is on
    and which part of that the camera has no evidence of (by the default evidence of the grader settings)."""
    exam = load_built(exam_dir)
    ev = load_grading(grading)["default_evidence"]
    L = [f"# Answer key: {exam['exam']}", "",
         f"Exam build `{exam['fingerprint_id']}`, {exam['rows_total']} rows. What the driver does in every row, read "
         "from the truth. Times are seconds from the start of the row; a span runs from its first frame to the first "
         "frame after it. `eyes_closed` is on in every blink too.", "",
         "- Frame by frame: `rows/<row_id>/*.truth.jsonl`, fields `events` (the labels) and `observable` (what the "
         "camera can show).",
         "- The script of a row and the model errors injected into it: `rows/<row_id>/card.md`.",
         "- The same truth in the answer-sheet format, to compare line by line with your own: "
         "`dms-sim exam oracle <exam dir> --out <dir>`.", "",
         "`no evidence` = the label is on but the camera cannot show it (a phone below the image, a mouth behind a "
         "hand): the grader does not count those frames against a signal."]
    for r in exam["rows"]:
        truth = load_truth(exam_dir, r["row_id"])
        fps = float(r["fps"])
        want = {"alert": f" Built to raise an alert of `{r['main_concept']}`.", "no_alert": " Built to raise no alert."}
        L += ["", f"## `{r['row_id']}`", "",
              f"{r['title']} ({r['frames'] / fps:.1f} s, {r['frames']} frames at {fps:g} fps, camera `{r['camera']}`, "
              f"error profile `{r['profile']}`).{want.get(r['intent'], '')}", ""]
        lines = []
        for c in CONCEPTS:
            t = [bool(x["events"][c]) for x in truth]
            if any(t):
                hidden = [t[i] and not any(x["observable"][ch] for ch in ev[c]) for i, x in enumerate(truth)]
                lines.append(f"| {c} | {_spans(t, fps)} | {_spans(hidden, fps) or 'none'} |")
        L += (["| behaviour | on (s) | of that, no evidence (s) |", "|---|---|---|"] + lines) if lines \
            else ["No behaviour label is on in this row."]
    return "\n".join(L) + "\n"


def _one(d: Path, pattern: str) -> Path:
    found = sorted(d.glob(pattern))
    if len(found) != 1:
        raise ExamError(f"{d}: expected one {pattern}, found {len(found)}")
    return found[0]


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(x) for x in fh if x.strip()]


def load_built(exam_dir) -> dict:
    p = Path(exam_dir) / "exam.json"
    if not p.is_file():
        raise ExamError(f"{p} not found (run `dms-sim exam build` first)")
    return json.loads(p.read_text(encoding="utf-8"))


def load_truth(exam_dir, row_id: str) -> list[dict]:
    """Truth records of one row (key/rows/<row_id>/*.truth.jsonl)."""
    return _read_jsonl(_one(Path(exam_dir) / "key" / "rows" / row_id, "*.truth.jsonl"))


# ---- answer sheets ------------------------------------------------------------------------------------
def load_grading(path=None) -> dict:
    g = load_yaml(path or GRADING)
    bad = {c: [x for x in ev if x not in OBSERVABLE_CHANNELS] for c, ev in (g.get("default_evidence") or {}).items()}
    if set(g.get("default_evidence") or {}) != set(CONCEPTS) or any(bad.values()):
        raise ExamError(f"grading: default_evidence needs every concept {CONCEPTS} with channels of {OBSERVABLE_CHANNELS}")
    return g


def load_answer_header(answers_dir, exam: dict, grading: dict) -> dict:
    """answers/answer.yaml checked against the exam: {name: {concept, kind, evidence, rule}} (evidence filled
    in from the grading defaults). Raises ExamError listing every problem."""
    p = Path(answers_dir) / "answer.yaml"
    if not p.is_file():
        raise ExamError(f"{p} not found (the answer sheet's declaration, see docs/EXAM.md)")
    head = load_yaml(p)
    errs = []
    if head.get("exam") != exam["exam"]:
        errs.append(f"exam is {head.get('exam')!r}, this exam is {exam['exam']!r}")
    if str(head.get("fingerprint_id")) != str(exam["fingerprint_id"]):
        errs.append(f"fingerprint_id is {head.get('fingerprint_id')!r}, this exam's is {exam['fingerprint_id']!r} "
                    f"(the answers were made for another build of the exam)")
    sig = head.get("signals")
    if not isinstance(sig, dict) or not sig:
        errs.append("signals: needs at least one signal")
        sig = {}
    out = {}
    for name, s in sig.items():
        s = s if isinstance(s, dict) else {}
        if s.get("concept") not in CONCEPTS:
            errs.append(f"signal {name!r}: concept must be one of {', '.join(CONCEPTS)}")
            continue
        if s.get("kind") not in KINDS:
            errs.append(f"signal {name!r}: kind must be state or alert")
        extra = sorted(set(s) - {"concept", "kind", "evidence", "rule"})
        if extra:
            errs.append(f"signal {name!r}: unknown keys {extra}")
        ev = s.get("evidence") or grading["default_evidence"][s["concept"]]
        if not isinstance(ev, list) or not ev or any(x not in OBSERVABLE_CHANNELS for x in ev):
            errs.append(f"signal {name!r}: evidence must list channels of {', '.join(OBSERVABLE_CHANNELS)}")
        rule = s.get("rule")
        if rule is not None:
            if s.get("kind") != "alert":
                errs.append(f"signal {name!r}: only an alert can carry a rule")
            thr = rule.get("threshold_s") if isinstance(rule, dict) else None
            if not isinstance(rule, dict) or rule.get("type") not in RULES or isinstance(thr, bool) \
                    or not isinstance(thr, (int, float)) or thr <= 0 or set(rule) - {"type", "threshold_s"}:
                errs.append(f"signal {name!r}: rule must be {{type: continuous, threshold_s: <seconds > 0>}}")
        out[str(name)] = {"concept": s["concept"], "kind": s.get("kind"), "evidence": ev, "rule": rule}
    if errs:
        raise ExamError(f"{p}: " + "; ".join(errs))
    return out


def load_answer_row(answers_dir, row_id: str, frames: int, signals: dict) -> dict[str, list[bool]] | None:
    """{signal: [on per frame]} of one row, None when the row was not answered. The sheet must have one line per
    frame of the question, in order, with every declared signal as 0 / 1 (or true / false)."""
    p = Path(answers_dir) / f"{row_id}.jsonl"
    if not p.is_file():
        return None
    recs = _read_jsonl(p)
    if len(recs) != frames:
        raise ExamError(f"{p}: {len(recs)} lines, the question has {frames} frames (one line per frame, also for "
                        f"frames without model output)")
    out = {s: [] for s in signals}
    for i, r in enumerate(recs):
        if r.get("frame_id") != i:
            raise ExamError(f"{p}: line {i + 1} has frame_id {r.get('frame_id')!r}, expected {i}")
        vals = r.get("signals")
        if not isinstance(vals, dict) or set(vals) != set(signals):
            raise ExamError(f"{p}: line {i + 1}: signals must be exactly the declared ones {sorted(signals)}")
        for s in signals:
            v = vals[s]
            if v not in (0, 1, True, False) or isinstance(v, float):
                raise ExamError(f"{p}: line {i + 1}: signal {s!r} is {v!r}, expected 0 or 1")
            out[s].append(bool(v))
    return out


def write_answers(answers_dir, exam: dict, signals: dict, rows: dict[str, dict[str, list]]) -> Path:
    """Write an answer sheet: answer.yaml (signals: {name: {concept, kind[, evidence][, rule]}}) and one
    <row_id>.jsonl per row of `rows` ({row_id: {signal: [0 / 1 per frame]}}). For adapters written in Python."""
    import yaml
    out = Path(answers_dir)
    out.mkdir(parents=True, exist_ok=True)
    head = {"exam": exam["exam"], "fingerprint_id": exam["fingerprint_id"], "signals": signals}
    (out / "answer.yaml").write_text(yaml.safe_dump(head, sort_keys=False, allow_unicode=True), encoding="utf-8",
                                     newline="\n")
    for row_id, sig in rows.items():
        n = len(next(iter(sig.values())))
        with open(out / f"{row_id}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            for i in range(n):
                fh.write(json.dumps({"frame_id": i, "signals": {k: int(bool(v[i])) for k, v in sig.items()}}) + "\n")
    return out


def oracle_answers(exam_dir, answers_dir) -> Path:
    """The answer sheet of a perfect reader: one state signal per concept, copied from the truth. Grading it must
    give full marks, so it checks an installation; it also shows the format."""
    exam = load_built(exam_dir)
    signals = {c: {"concept": c, "kind": "state"} for c in CONCEPTS}
    rows = {}
    for r in exam["rows"]:
        truth = load_truth(exam_dir, r["row_id"])
        rows[r["row_id"]] = {c: [bool(t["events"][c]) for t in truth] for c in CONCEPTS}
    return write_answers(answers_dir, exam, signals, rows)


# ---- grading one row ----------------------------------------------------------------------------------
def _frames(seconds: float, fps: float) -> int:
    return int(round(float(seconds) * float(fps)))


def grade_state(t: list[bool], obs: list[bool], a: list[bool], band: int) -> dict:
    """A state signal against its truth label, frame lists of one row. obs: the label's evidence is observable.
    band: frames around a truth segment's start / end that the per-frame counts leave out (and how long after
    its end a first response still counts).

    Per truth segment with at least one observable frame: detected (the signal is on somewhere from the segment's
    start to `band` frames after its end), onset_delay (frames from the first observable frame to the first on
    frame; negative = on before the evidence was there), release_delay (frames the signal stays on after the
    end; negative = it went off that many frames early; None = still on at the end of the row), dropouts (times
    it fell off between its first and last on frame in the segment, counting only gaps with an observable frame).
    Segments without an observable frame are listed apart (`unobservable`), not counted as misses.
    false_segments: runs of the signal with no truth within `band` frames. Per-frame counts tp / fn (truth on,
    observable) and fp / tn (truth off), all outside the bands; unobs_pos / unobs_on: truth-on frames without
    evidence, and how many of them the signal was on."""
    n = len(t)
    segs, unobservable = [], []
    edge = [False] * n
    for s, e in segments(t):
        for b in (s, e):
            for i in range(max(0, b - band), min(n, b + band)):
                edge[i] = True
        ob = [i for i in range(s, e) if obs[i]]
        if not ob:
            unobservable.append({"start": s, "end": e, "on": any(a[s:e])})
            continue
        f0 = ob[0]
        on = [i for i in range(s, min(n, e + band)) if a[i]]
        seg = {"start": s, "end": e, "first_observable": f0, "detected": bool(on), "onset_delay": None,
               "release_delay": None, "dropouts": 0}
        if on:
            seg["onset_delay"] = on[0] - f0
            inside = [i for i in on if i < e]
            if inside:
                gaps = segments([not a[i] for i in range(inside[0], inside[-1] + 1)])
                seg["dropouts"] = sum(any(obs[inside[0] + k] for k in range(gs, ge)) for gs, ge in gaps)
            if a[e - 1] or (e < n and a[e]):
                j = e
                while j < n and a[j]:
                    j += 1
                seg["release_delay"] = None if j == n and e < n else j - e
            else:
                seg["release_delay"] = (inside[-1] + 1 - e) if inside else 0
        segs.append(seg)
    false_segments = [{"start": s, "end": e} for s, e in segments(a)
                      if not any(t[max(0, s - band):min(n, e + band)])]
    c = {"tp": 0, "fn": 0, "fp": 0, "tn": 0, "unobs_pos": 0, "unobs_on": 0}
    for i in range(n):
        if t[i] and not obs[i]:
            c["unobs_pos"] += 1
            c["unobs_on"] += a[i]
        elif not edge[i]:
            c[("tp" if a[i] else "fn") if t[i] else ("fp" if a[i] else "tn")] += 1
    return {"frames": n, "segments": segs, "unobservable": unobservable, "false_segments": false_segments, **c}


def grade_alert(t: list[bool], obs: list[bool], a: list[bool], fps: float, grace: int, lookback: int,
                rule: dict | None = None, rule_tol: int = 1) -> dict:
    """An alert signal against the truth label it belongs to, frame lists of one row. Only the frames where the
    signal turns on count (`onsets`), so a latched alert and a one-frame pulse grade alike.

    Every onset is one of: during (inside a truth segment or up to `grace` frames after it: it belongs to that
    segment), after (not during, but the label was on in the `lookback` frames before) or unprovoked (the label
    was not). Per truth segment with an observable frame: alerts (onsets that belong to it) and delay (frames
    from its first observable frame to the first of them): the delay is this program's effective threshold.

    rule {type: continuous, threshold_s}: the answer sheet's own rule, "alert once the behaviour has lasted
    threshold_s". Each observable segment then gets a verdict: hit / miss / early (alert more than rule_tol
    frames before the rule allows) for segments at least rule_tol frames longer than the threshold, false /
    silent for segments at least rule_tol frames shorter, borderline (not judged) in between; `latency` is
    frames after the frame the rule is met. Onsets outside every segment are false under any rule."""
    n = len(t)
    onsets = [i for i in range(n) if a[i] and (i == 0 or not a[i - 1])]
    segs = []
    for s, e in segments(t):
        ob = [i for i in range(s, e) if obs[i]]
        segs.append({"start": s, "end": e, "first_observable": ob[0] if ob else None, "alerts": [], "delay": None})
    kinds = {"during": 0, "after": 0, "unprovoked": 0}
    unprovoked_at = []
    for k in onsets:
        owner = next((sg for sg in reversed(segs) if sg["start"] <= k < sg["end"] + grace), None)
        if owner is not None:
            owner["alerts"].append(k)
            kinds["during"] += 1
        elif any(t[max(0, k - lookback):k]):
            kinds["after"] += 1
        else:
            kinds["unprovoked"] += 1
            unprovoked_at.append(k)
    need = math.ceil(float(rule["threshold_s"]) * fps - 1e-9) if rule else None
    for sg in segs:
        f0 = sg["first_observable"]
        if f0 is None:
            continue
        if sg["alerts"]:
            sg["delay"] = sg["alerts"][0] - f0
        if rule:
            length, due = sg["end"] - f0, f0 + need - 1     # due: the frame at which the run has lasted threshold_s
            if length >= need + rule_tol:
                first = sg["alerts"][0] if sg["alerts"] else None
                sg["verdict"] = "miss" if first is None else "early" if first < due - rule_tol else "hit"
                if sg["verdict"] == "hit":
                    sg["latency"] = first - due
            elif length <= need - rule_tol:
                sg["verdict"] = "false" if sg["alerts"] else "silent"
            else:
                sg["verdict"] = "borderline"
    return {"frames": n, "truth_frames": sum(t), "onsets": onsets, **kinds, "unprovoked_at": unprovoked_at,
            "segments": segs}


# ---- grading an exam ----------------------------------------------------------------------------------
def _stats(xs: list[float]) -> dict | None:
    if not xs:
        return None
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(math.ceil(p * len(xs))) - 1)]
    return {"n": len(xs), "min": xs[0], "median": q(0.5), "p90": q(0.9), "max": xs[-1]}


def _ratio(a: int, b: int) -> float | None:
    return a / b if b else None


def _sum_state(items: list[tuple[dict, dict]]) -> dict:
    """Totals of grade_state results; items: [(exam row, result)]. Delays in seconds."""
    seg = [(r["fps"], s) for r, g in items for s in g["segments"]]
    det = [(f, s) for f, s in seg if s["detected"]]
    tot = {k: sum(g[k] for _, g in items) for k in ("tp", "fn", "fp", "tn", "unobs_pos", "unobs_on")}
    minutes = sum(g["frames"] / r["fps"] for r, g in items) / 60.0
    nfalse = sum(len(g["false_segments"]) for _, g in items)
    return {"rows": len(items), "segments": len(seg), "detected": len(det),
            "detected_rate": _ratio(len(det), len(seg)),
            "onset_delay_s": _stats([s["onset_delay"] / f for f, s in det]),
            "release_delay_s": _stats([s["release_delay"] / f for f, s in det if s["release_delay"] is not None]),
            "still_on_at_row_end": sum(s["release_delay"] is None for _, s in det),
            "dropouts": sum(s["dropouts"] for _, s in det),
            "unobservable_segments": sum(len(g["unobservable"]) for _, g in items),
            "false_segments": nfalse, "false_segments_per_min": nfalse / minutes if minutes else None,
            "frame_recall": _ratio(tot["tp"], tot["tp"] + tot["fn"]),
            "frame_precision": _ratio(tot["tp"], tot["tp"] + tot["fp"]),
            "frame_false_rate": _ratio(tot["fp"], tot["fp"] + tot["tn"]), **tot}


def _sum_alert(items: list[tuple[dict, dict]], has_rule: bool) -> dict:
    seg = [(r["fps"], s) for r, g in items for s in g["segments"] if s["first_observable"] is not None]
    fired = [(f, s) for f, s in seg if s["alerts"]]
    hours = sum(g["frames"] / r["fps"] for r, g in items) / 3600.0
    n_on = sum(len(g["onsets"]) for _, g in items)
    out = {"rows": len(items), "alerts": n_on, "alerts_per_hour": n_on / hours if hours else None,
           **{k: sum(g[k] for _, g in items) for k in ("during", "after", "unprovoked")},
           # the hard fact: the label is never on in the row, yet the alert came
           "rows_without_behaviour": sum(g["truth_frames"] == 0 for _, g in items),
           "rows_without_behaviour_alerted": sum(g["truth_frames"] == 0 and bool(g["onsets"]) for _, g in items),
           "segments": len(seg), "segments_alerted": len(fired),
           "segments_alerted_more_than_once": sum(len(s["alerts"]) > 1 for _, s in fired),
           "delay_s": _stats([s["delay"] / f for f, s in fired])}
    if has_rule:
        verdicts = [s.get("verdict") for _, s in seg]
        out["rule"] = {v: verdicts.count(v) for v in ("hit", "miss", "early", "false", "silent", "borderline")}
        out["rule"]["false_outside_behaviour"] = out["after"] + out["unprovoked"]
        out["rule"]["latency_s"] = _stats([s["latency"] / f for f, s in seg if "latency" in s])
    return out


def _intent(row: dict, signals: dict, graded: dict) -> dict | None:
    """Verdict of a row the exam was built to make obvious (row["intent"]), from its alert signals."""
    if not row.get("intent"):
        return None
    alerts = {k: s for k, s in signals.items() if s["kind"] == "alert"}
    out = {"row_id": row["row_id"], "entry": row["entry"], "intent": row["intent"], "concept": row["main_concept"]}
    if row["intent"] == "no_alert":
        got = {k: len(graded[k]["onsets"]) for k in alerts if graded[k]["onsets"]}
        return {**out, "verdict": "not_answered" if not alerts else "fail" if got else "pass", "alerts": got}
    mine = [k for k, s in alerts.items() if s["concept"] == row["main_concept"]]
    if not mine:
        return {**out, "verdict": "not_answered", "alerts": {}}
    segs = [s for k in mine for s in graded[k]["segments"] if s["first_observable"] is not None]
    got = {k: graded[k]["during"] for k in mine if graded[k]["during"]}
    return {**out, "verdict": "not_judged" if not segs else "pass" if got else "fail", "alerts": got}


def _sweeps(rows: list[dict], signals: dict, graded: dict) -> list[dict]:
    """Parameter sweeps (BND entries): rows of one entry that differ in a single numeric parameter, per signal the
    response along that parameter (alert: any onset; state: any segment detected). `changes` counts how often
    the response flips along the sorted values: 1 = one clean switching point (between `last_before` and
    `first_after`), more = the response is not monotone."""
    groups: dict = {}
    for r in rows:
        if r["category"] == "BND" and r["row_id"] in graded:
            key = (r["entry"], r["level"], r["camera"], r["variant"], r["seed"], r["emulator_seed"], r["profile"])
            groups.setdefault(key, []).append(r)
    out = []
    for key, rs in groups.items():
        names = sorted({k for r in rs for k in r["params"] if k != "level"})
        vary = [k for k in names if len({json.dumps(r["params"].get(k)) for r in rs}) > 1]
        if len(vary) != 1 or not all(isinstance(r["params"].get(vary[0]), (int, float)) for r in rs):
            continue
        rs = sorted(rs, key=lambda r: r["params"][vary[0]])
        for name, s in signals.items():
            pts = []
            for r in rs:
                g = graded[r["row_id"]][name]
                if s["kind"] == "alert":
                    seg = next((x for x in g["segments"] if x["alerts"]), None)
                    pts.append({"value": r["params"][vary[0]], "responded": bool(g["onsets"]),
                                "delay_s": None if seg is None or seg["delay"] is None else seg["delay"] / r["fps"]})
                else:
                    seg = next((x for x in g["segments"] if x["detected"]), None)
                    pts.append({"value": r["params"][vary[0]], "responded": seg is not None,
                                "delay_s": None if seg is None else seg["onset_delay"] / r["fps"]})
            flips = [i for i in range(1, len(pts)) if pts[i]["responded"] != pts[i - 1]["responded"]]
            out.append({"entry": key[0], "level": key[1], "camera": key[2], "profile": key[6], "signal": name,
                        "param": vary[0], "points": pts, "changes": len(flips),
                        "last_before": pts[flips[0] - 1]["value"] if len(flips) == 1 else None,
                        "first_after": pts[flips[0]]["value"] if len(flips) == 1 else None})
    return out


RULE_FINDINGS = {"miss": "own rule: no alert", "early": "own rule: alert too early",
                 "false": "own rule: alert in a segment shorter than the threshold"}


def _findings(rows_out: list[dict], signals: dict) -> list[dict]:
    """Every case the summaries count against a signal, with its row and time: {row_id, signal, what, start_s,
    end_s (None for a single frame)}. State signals: a truth segment missed, a signal that fell off inside one, a
    run with no behaviour near it. Alerts: an onset with none of its behaviour before it, and the miss / early /
    false verdicts of the sheet's own rule. Delays are not listed: they are measurements."""
    out = []
    for r in rows_out:
        fps = r["fps"]
        for name, g in r["signals"].items():
            def add(what, s, e=None):
                out.append({"row_id": r["row_id"], "signal": name, "what": what, "start_s": s / fps,
                            "end_s": None if e is None else e / fps})
            if signals[name]["kind"] == "state":
                for sg in g["segments"]:
                    if not sg["detected"]:
                        add("missed", sg["start"], sg["end"])
                    elif sg["dropouts"]:
                        add(f"fell off {sg['dropouts']} time(s) during the behaviour", sg["start"], sg["end"])
                for sg in g["false_segments"]:
                    add("on without the behaviour", sg["start"], sg["end"])
            else:
                for k in g["unprovoked_at"]:
                    add("alert without the behaviour", k)
                for sg in g["segments"]:
                    if sg.get("verdict") in RULE_FINDINGS:
                        add(RULE_FINDINGS[sg["verdict"]], sg["start"], sg["end"])
    return out


def grade(exam_dir, answers_dir, grading=None) -> dict:
    """Grade an answer sheet against a built exam. Returns the report (write_report writes it): the declared
    signals, per-signal summaries (overall and by category / family / level / profile), the verdicts of the
    designed-obvious rows, the parameter sweeps, the findings (every miss and false response with its row and
    time) and every row's details. Rows without an answer file are listed in `rows_missing` and left out; a
    malformed answer file raises ExamError."""
    exam = load_built(exam_dir)
    g = load_grading(grading)
    signals = load_answer_header(answers_dir, exam, g)
    known = {r["row_id"] for r in exam["rows"]}
    stray = sorted(p.stem for p in Path(answers_dir).glob("*.jsonl") if p.stem not in known)
    if stray:
        raise ExamError(f"{answers_dir}: answer files for rows that are not in this exam: {stray[:5]}")
    graded, missing, rows_out, intents = {}, [], [], []
    for r in exam["rows"]:
        ans = load_answer_row(answers_dir, r["row_id"], r["frames"], signals)
        if ans is None:
            missing.append(r["row_id"])
            continue
        truth = load_truth(exam_dir, r["row_id"])
        fps = float(r["fps"])
        res = {}
        for name, s in signals.items():
            t = [bool(x["events"][s["concept"]]) for x in truth]
            obs = [any(x["observable"][ch] for ch in s["evidence"]) for x in truth]
            if s["kind"] == "state":
                res[name] = grade_state(t, obs, ans[name], _frames(g["edge_band_s"], fps))
            else:
                res[name] = grade_alert(t, obs, ans[name], fps, _frames(g["alert_grace_s"], fps),
                                        _frames(g["lookback_s"], fps), s["rule"],
                                        max(1, _frames(g["rule_tolerance_s"], fps)))
        graded[r["row_id"]] = res
        rows_out.append({**{k: r[k] for k in ("row_id", "entry", "category", "family", "level", "profile", "camera",
                                              "params", "fps", "frames", "intent", "main_concept")}, "signals": res})
        it = _intent(r, signals, res)
        if it:
            intents.append(it)
    if not graded:
        raise ExamError(f"{answers_dir}: no answer file matches a row of exam {exam['exam']} (<row_id>.jsonl)")
    by_row = {r["row_id"]: r for r in exam["rows"]}
    summary = {}
    for name, s in signals.items():
        items = [(by_row[rid], res[name]) for rid, res in graded.items()]
        total = (lambda it: _sum_state(it)) if s["kind"] == "state" else (lambda it: _sum_alert(it, bool(s["rule"])))
        summary[name] = {"overall": total(items),
                         "by": {k: {v: total([x for x in items if str(x[0][k]) == v])
                                    for v in sorted({str(x[0][k]) for x in items})} for k in SLICES}}
    return {"exam": exam["exam"], "title": exam.get("title", ""), "fingerprint_id": exam["fingerprint_id"],
            "grading": g, "signals": signals,
            "concepts_not_answered": sorted(set(CONCEPTS) - {s["concept"] for s in signals.values()}),
            "rows_total": len(exam["rows"]), "rows_answered": len(graded), "rows_missing": missing,
            "summary": summary, "intent": intents, "sweeps": _sweeps(exam["rows"], signals, graded),
            "findings": _findings(rows_out, signals), "rows": rows_out}


# ---- report -------------------------------------------------------------------------------------------
def _f(v, digits: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def _pct(v) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def _dist(st: dict | None) -> str:
    return "n/a" if not st else f"{st['median']:.2f} / {st['p90']:.2f} / {st['max']:.2f}"


def report_markdown(rep: dict) -> str:
    sig = rep["signals"]
    states = [k for k, s in sig.items() if s["kind"] == "state"]
    alerts = [k for k, s in sig.items() if s["kind"] == "alert"]
    g = rep["grading"]
    L = [f"# Exam report: {rep['exam']}", ""]
    if rep["title"]:
        L += [rep["title"], ""]
    L += [f"- Exam build: `{rep['fingerprint_id']}`",
          f"- Rows answered: {rep['rows_answered']} of {rep['rows_total']}"
          + (f" (missing: {', '.join(rep['rows_missing'][:5])}{' ...' if len(rep['rows_missing']) > 5 else ''})"
             if rep["rows_missing"] else ""),
          f"- Concepts without a signal (not answered): {', '.join(rep['concepts_not_answered']) or 'none'}",
          f"- Grader settings: edge band {g['edge_band_s']} s, alert grace {g['alert_grace_s']} s, lookback "
          f"{g['lookback_s']} s, rule tolerance {g['rule_tolerance_s']} s", "",
          "The grader has no alert thresholds. Delays are measurements, not errors. Truth frames whose evidence no "
          "detector could see (a phone in the lap, a mouth behind a hand) are not counted against a signal.", "",
          "## Signals", "", "| signal | concept | kind | evidence | own rule |", "|---|---|---|---|---|"]
    for k, s in sig.items():
        rule = f"{s['rule']['type']} >= {s['rule']['threshold_s']} s" if s["rule"] else ""
        L.append(f"| `{k}` | {s['concept']} | {s['kind']} | {', '.join(s['evidence'])} | {rule} |")
    if states:
        L += ["", "## Reading the driver state", "",
              "Per truth segment that has evidence: was the signal on, how late, did it flicker. Delays in seconds as "
              "median / p90 / max. Frame recall and precision leave out the edge band around segment starts and ends.",
              "", "| signal | segments | detected | onset delay | release delay | dropouts | false segments (per min) "
              "| frame recall | frame precision | no-evidence frames (signal on) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for k in states:
            o = rep["summary"][k]["overall"]
            L.append(f"| `{k}` | {o['segments']} | {o['detected']} ({_pct(o['detected_rate'])}) | "
                     f"{_dist(o['onset_delay_s'])} | {_dist(o['release_delay_s'])} | {o['dropouts']} | "
                     f"{o['false_segments']} ({_f(o['false_segments_per_min'])}) | {_pct(o['frame_recall'])} | "
                     f"{_pct(o['frame_precision'])} | {o['unobs_pos']} ({o['unobs_on']}) |")
        for k in states:
            L += ["", f"### `{k}` by row category", "",
                  "| category | rows | segments | detected | onset delay | dropouts | false segments | frame recall |",
                  "|---|---|---|---|---|---|---|---|"]
            for cat, o in rep["summary"][k]["by"]["category"].items():
                L.append(f"| {cat} | {o['rows']} | {o['segments']} | {o['detected']} ({_pct(o['detected_rate'])}) | "
                         f"{_dist(o['onset_delay_s'])} | {o['dropouts']} | {o['false_segments']} | "
                         f"{_pct(o['frame_recall'])} |")
    if alerts:
        L += ["", "## Alerts", "",
              "An alert counts where the signal turns on. `during` = inside a truth segment of its concept (or within "
              "the grace time after it), `after` = the behaviour was there within the lookback, `unprovoked` = it was "
              "not. `no-behaviour rows alerted` is the hard fact: the behaviour never occurs in the row, yet the alert "
              "came. Delay = seconds from the first frame with evidence to the alert (median / p90 / max): the "
              "program's effective threshold.", "",
              "| signal | alerts | during | after | unprovoked | no-behaviour rows alerted | segments alerted | "
              "more than once | delay |", "|---|---|---|---|---|---|---|---|---|"]
        for k in alerts:
            o = rep["summary"][k]["overall"]
            L.append(f"| `{k}` | {o['alerts']} | {o['during']} | {o['after']} | {o['unprovoked']} | "
                     f"{o['rows_without_behaviour_alerted']} of {o['rows_without_behaviour']} | "
                     f"{o['segments_alerted']} of {o['segments']} | {o['segments_alerted_more_than_once']} | "
                     f"{_dist(o['delay_s'])} |")
        ruled = [k for k in alerts if sig[k]["rule"]]
        if ruled:
            L += ["", "### Against the answer sheet's own rule", "",
                  "Segments clearly longer than the declared threshold should be hit, clearly shorter ones should stay "
                  "silent; segments within the rule tolerance of it are borderline and not judged.", "",
                  "| signal | rule | hit | miss | early | false (short segment) | false (outside behaviour) | silent "
                  "| borderline | latency after the rule is met |", "|---|---|---|---|---|---|---|---|---|---|"]
            for k in ruled:
                r = rep["summary"][k]["overall"]["rule"]
                L.append(f"| `{k}` | {sig[k]['rule']['type']} >= {sig[k]['rule']['threshold_s']} s | {r['hit']} | "
                         f"{r['miss']} | {r['early']} | {r['false']} | {r['false_outside_behaviour']} | {r['silent']} | "
                         f"{r['borderline']} | {_dist(r['latency_s'])} |")
    if rep["intent"] and alerts:
        c = {v: sum(i["verdict"] == v for i in rep["intent"]) for v in ("pass", "fail", "not_judged", "not_answered")}
        L += ["", "## Rows built to be obvious", "",
              "The exam author's design, not a specification: POS rows should raise an alert of their main behaviour, "
              "NEG rows no alert at all. `not judged` = the behaviour has no evidence in this camera; `not answered` = "
              "no alert signal for it.", "",
              f"pass {c['pass']}, fail {c['fail']}, not judged {c['not_judged']}, not answered {c['not_answered']}"]
        fails = [i for i in rep["intent"] if i["verdict"] == "fail"]
        if fails:
            L += ["", "| row | designed | got |", "|---|---|---|"]
            for i in fails:
                got = ", ".join(f"{k} x{v}" for k, v in i["alerts"].items()) or "no alert"
                want = f"alert ({i['concept']})" if i["intent"] == "alert" else "no alert"
                L.append(f"| `{i['row_id']}` | {want} | {got} |")
    if rep["sweeps"]:
        L += ["", "## Parameter sweeps", "",
              "Rows of one entry that differ in one parameter. `switch` = between which two values the response "
              "changes; `not monotone` = it flips more than once along the sweep.", "",
              "| entry | level | profile | signal | parameter | response along the values | switch |", "|---|---|---|---|---|---|---|"]
        for s in rep["sweeps"]:
            pts = " ".join(f"{p['value']:g}:{'Y' if p['responded'] else '-'}" for p in s["points"])
            sw = (f"{s['last_before']:g} .. {s['first_after']:g}" if s["changes"] == 1
                  else "none" if s["changes"] == 0 else "not monotone")
            L.append(f"| {s['entry']} | {s['level'] or ''} | {s['profile']} | `{s['signal']}` | {s['param']} | {pts} | {sw} |")
    found, cap = rep["findings"], int(g["report_findings_max"])
    L += ["", "## Where to look", "",
          "Every case the tables above count against a signal, with its row and time (delays are not listed: they are "
          "measurements). What the driver did in a row: `key/answer_key.md` in the exam directory; frame by frame: "
          "`key/rows/<row_id>/*.truth.jsonl`; the script and the injected model errors: `key/rows/<row_id>/card.md`.", ""]
    if found:
        L += ["| row | signal | what | when (s) |", "|---|---|---|---|"]
        for f in found[:cap]:
            when = f"{f['start_s']:.2f}" + ("" if f["end_s"] is None else f"-{f['end_s']:.2f}")
            L.append(f"| `{f['row_id']}` | `{f['signal']}` | {f['what']} | {when} |")
        if len(found) > cap:
            L += ["", f"{len(found) - cap} more in `report.json` (`findings`)."]
    else:
        L.append("Nothing: no missed or false segment, no alert without its behaviour, no verdict against an own rule.")
    L += ["", "Per-row details (every segment, every alert frame): `report.json`.", ""]
    return "\n".join(L)


def write_report(rep: dict, out_dir) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "report.json", rep)
    p = out / "report.md"
    p.write_text(report_markdown(rep), encoding="utf-8", newline="\n")
    return p

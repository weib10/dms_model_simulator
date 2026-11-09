"""Replay pipeline-format JSONL (or a dms.bin file) to a post-processing program (see docs/SPEC.md "Replay API")."""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Iterator
from .schema import FIELDS


def _iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def iter_pipeline(path: str | Path, realtime: bool = False) -> Iterator[dict]:
    """Yield pipeline records in file order. realtime=True sleeps so records are yielded at their
    timestamp_ms spacing (measured from the first record, so sleeps do not accumulate drift). Records with a
    missing / null timestamp_ms are yielded without waiting."""
    t0_wall = t0_rec = None
    for rec in _iter_jsonl(path):
        if realtime and rec.get(FIELDS["timestamp_ms"]) is not None:
            ts = float(rec[FIELDS["timestamp_ms"]]) / 1000.0
            if t0_wall is None:
                t0_wall, t0_rec = time.monotonic(), ts
            delay = (ts - t0_rec) - (time.monotonic() - t0_wall)
            if delay > 0:
                time.sleep(delay)
        yield rec


def iter_dms_bin(path: str | Path, layout=None, realtime: bool = False, strict: bool = True) -> Iterator[dict]:
    """Yield frames of a dms.bin file (see binlog.read_dms_bin) in file order. Each frame carries the raw
    `header` / `objects` plus pipeline-style frame_id, timestamp_ms and od. realtime=True paces frames by
    timestamp_ms exactly like iter_pipeline. layout: DmsBinLayout, YAML path or None (configs/binlog.yaml).
    strict=False: a truncated last frame ends the stream with a warning instead of raising (see read_dms_bin)."""
    from .binlog import read_dms_bin
    t0_wall = t0_rec = None
    for fr in read_dms_bin(path, layout, strict=strict):
        if realtime and fr[FIELDS["timestamp_ms"]] is not None:
            ts = fr[FIELDS["timestamp_ms"]] / 1000.0
            if t0_wall is None:
                t0_wall, t0_rec = time.monotonic(), ts
            delay = (ts - t0_rec) - (time.monotonic() - t0_wall)
            if delay > 0:
                time.sleep(delay)
        yield fr


def iter_pairs(pipeline_path: str | Path, truth_path: str | Path, realtime: bool = False) -> Iterator[tuple[dict, dict]]:
    """Yield (pred, truth) pairs aligned by frame_id, in pipeline order. Pipeline frames without a
    truth record are skipped."""
    key = FIELDS["frame_id"]
    truth = {r[key]: r for r in _iter_jsonl(truth_path)}
    for pred in iter_pipeline(pipeline_path, realtime=realtime):
        t = truth.get(pred[key])
        if t is not None:
            yield pred, t

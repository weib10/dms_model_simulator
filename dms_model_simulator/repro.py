"""Reproducibility: a fingerprint of everything that decides a run's output, and a comparison of two runs.

A catalog run (catalog.expand) stores the fingerprint in manifest.json: package version, git commit, numpy
version and a hash of every config file, of catalog/profiles.yaml and catalog/behaviors.yaml and of each selected
entry. Same fingerprint + same row ids = the same test. Outputs are then identical on the same machine; across
operating systems the maths libraries can differ in the last digit of a float (measured: < 1e-12 px), so
compare_runs checks structure exactly (detections present, classes, flags, labels) and numbers with a tolerance.
"""
from __future__ import annotations
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import numpy as np
from .config import CONFIG_DIR, ROOT

# fingerprint keys that change the output; the others are recorded for information only
DECISIVE = ("package", "git_commit", "numpy", "configs", "catalog", "entries")
# optional pictures of a row (`catalog run --timeline / --gif`): views of the output, not output
PICTURE_FILES = {"timeline": "timeline.png", "gif": "preview.gif"}
# not compared: row.json holds absolute paths of this machine; a run with pictures is the same test as one without
SKIP_FILES = {"row.json", *PICTURE_FILES.values()}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def text_sha(path) -> str:
    """Hash of a text file with line endings normalised (a checkout with CRLF hashes like one with LF)."""
    return _sha(Path(path).read_bytes().replace(b"\r\n", b"\n"))


def _git(*args) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def fingerprint(entries: list[dict] | None = None) -> dict:
    """What decides the output of a run: code version, configs, catalog files and the selected entries."""
    from . import __version__
    from .catalog import CATALOG_DIR
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no", "--", "dms_model_simulator", "configs",
                      "catalog")) if commit else None
    return {
        "package": __version__,
        "git_commit": commit,
        "git_dirty": dirty,                    # uncommitted changes to code / configs / catalog
        "numpy": np.__version__,
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "configs": {p.name: text_sha(p) for p in sorted(CONFIG_DIR.glob("*.yaml"))},
        "catalog": {n: text_sha(CATALOG_DIR / n) for n in ("profiles.yaml", "behaviors.yaml")
                    if (CATALOG_DIR / n).exists()},
        "entries": {e["id"]: text_sha(e["_path"]) for e in entries or [] if e.get("_path")},
    }


def fingerprint_id(fp: dict) -> str:
    """Short id of the decisive part of a fingerprint (printed on behaviour cards)."""
    return _sha(json.dumps({k: fp.get(k) for k in DECISIVE}, sort_keys=True).encode())


def compare_fingerprints(a: dict | None, b: dict | None) -> tuple[list[str], list[str]]:
    """(differences that change the output, informational differences)."""
    if not a or not b:
        return ["a run has no fingerprint (made before fingerprints were recorded)"], []
    diff, info = [], []
    for k in DECISIVE:
        x, y = a.get(k), b.get(k)
        if isinstance(x, dict) or isinstance(y, dict):
            x, y = x or {}, y or {}
            for name in sorted(set(x) | set(y)):
                if x.get(name) != y.get(name):
                    diff.append(f"{k}/{name}: {x.get(name)} vs {y.get(name)}")
        elif x != y:
            diff.append(f"{k}: {x} vs {y}")
    for k in ("git_dirty", "python", "platform"):
        if a.get(k) != b.get(k):
            (diff if k == "git_dirty" and (a.get(k) or b.get(k)) else info).append(f"{k}: {a.get(k)} vs {b.get(k)}")
    return diff, info


class _Cmp:
    """Walks two parsed documents: structure must match exactly, numbers within tol."""

    def __init__(self, tol: float):
        self.tol, self.worst, self.problems = tol, 0.0, []

    def walk(self, x, y, path: str) -> None:
        if len(self.problems) > 20:
            return
        num = (int, float)
        if isinstance(x, num) and isinstance(y, num) and not isinstance(x, bool) and not isinstance(y, bool):
            d = abs(float(x) - float(y))
            self.worst = max(self.worst, d)
            if d > self.tol:
                self.problems.append(f"{path}: {x} vs {y}")
        elif isinstance(x, dict) and isinstance(y, dict):
            if set(x) != set(y):
                self.problems.append(f"{path}: keys {sorted(set(x) ^ set(y))}")
            for k in sorted(set(x) & set(y), key=str):
                self.walk(x[k], y[k], f"{path}/{k}")
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                self.problems.append(f"{path}: {len(x)} vs {len(y)} items")
            for i, (p, q) in enumerate(zip(x, y)):
                self.walk(p, q, f"{path}[{i}]")
        elif x != y:
            self.problems.append(f"{path}: {x!r} vs {y!r}")


def _load(p: Path):
    if p.suffix == ".jsonl":
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    if p.suffix == ".yaml":
        import yaml
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    return None


def compare_row(a: Path, b: Path, tol: float) -> tuple[list[str], float]:
    """(problems, largest number difference) between two exported row directories."""
    fa = {p.relative_to(a).as_posix() for p in a.rglob("*") if p.is_file() and p.name not in SKIP_FILES}
    fb = {p.relative_to(b).as_posix() for p in b.rglob("*") if p.is_file() and p.name not in SKIP_FILES}
    problems = [f"only in A: {x}" for x in sorted(fa - fb)] + [f"only in B: {x}" for x in sorted(fb - fa)]
    worst = 0.0
    for rel in sorted(fa & fb):
        pa, pb = a / rel, b / rel
        ba, bb = pa.read_bytes(), pb.read_bytes()
        if ba == bb or ba.replace(b"\r\n", b"\n") == bb.replace(b"\r\n", b"\n"):
            continue                                     # identical text: no need to parse
        if pa.suffix in (".jsonl", ".json", ".yaml"):
            c = _Cmp(tol)
            c.walk(_load(pa), _load(pb), rel)
            problems += c.problems
            worst = max(worst, c.worst)
        elif pa.suffix in (".md", ".tsv", ".txt"):
            if pa.read_bytes().replace(b"\r\n", b"\n") != pb.read_bytes().replace(b"\r\n", b"\n"):
                problems.append(f"{rel}: text differs")
        elif pa.read_bytes() != pb.read_bytes():
            problems.append(f"{rel}: bytes differ")
    return problems, worst


def compare_runs(a, b, tol: float = 1e-9) -> dict:
    """Compare two catalog run directories (manifest.json + rows/). `same` is True when the fingerprints agree
    on everything that decides the output, the row ids are the same and every row matches (see compare_row)."""
    from .catalog import load_manifest
    a, b = Path(a), Path(b)
    ma, mb = load_manifest(a), load_manifest(b)
    fp_diff, fp_info = compare_fingerprints(ma.get("fingerprint"), mb.get("fingerprint"))
    ra, rb = {r["row_id"] for r in ma["rows"]}, {r["row_id"] for r in mb["rows"]}
    rows, worst, not_run = {}, 0.0, []
    for rid in sorted(ra & rb):
        da, db = a / "rows" / rid, b / "rows" / rid
        if not da.is_dir() or not db.is_dir():
            not_run.append(rid)
            continue
        problems, w = compare_row(da, db, tol)
        worst = max(worst, w)
        if problems:
            rows[rid] = problems
    same = not fp_diff and ra == rb and not rows and not not_run
    return {"same": same, "fingerprint_diff": fp_diff, "fingerprint_info": fp_info,
            "only_a": sorted(ra - rb), "only_b": sorted(rb - ra), "not_exported": not_run,
            "rows_compared": len(ra & rb) - len(not_run), "rows_differ": rows, "worst_number_diff": worst,
            "tol": tol}

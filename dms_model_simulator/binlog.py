"""dms.bin writer / reader: per-frame model detections in a packed binary log.

The byte layout is data-driven: endianness, field order and types, arrays and nested records, class ids and
the simulator -> output class map, per-class conf thresholds, bbox format, units and file magic all come from a
YAML layout (default `configs/binlog.yaml`). Only model outputs are written: header fields other than
timestamp / frame_index / width / height / num_objects are written as their `default` (0) and read back raw.
"""
from __future__ import annotations
import collections
import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
from .config import CONFIG_DIR, load_yaml
from .schema import FIELDS, LABELS, mapped_detections, to_pipeline

DEFAULT_LAYOUT = CONFIG_DIR / "binlog.yaml"

TYPE_CODES = {"u8": "B", "i8": "b", "u16": "H", "i16": "h", "u32": "I", "i32": "i",
              "u64": "Q", "i64": "q", "f32": "f", "f64": "d"}
ENDIAN = {"little": "<", "big": ">"}
TS_PER_S = {"s": 1.0, "ms": 1e3, "us": 1e6}
BOX_FIELDS = {"xywh": ("x", "y", "w", "h"), "ltwh": ("x", "y", "w", "h"), "xyxy": ("x1", "y1", "x2", "y2")}
_LABEL_TO_CLASS = {v: k for k, v in LABELS.items()}


@dataclass
class _Leaf:
    name: tuple            # C field names from the record root, e.g. ("pose", "yaw")
    role: str              # dotted role path, e.g. "pose.yaw"
    kind: str
    count: int | None      # None = scalar, N = fixed array
    default: Any


class _Record:
    """One packed struct (frame header or object): ordered fields -> struct format. Fields may be fixed arrays
    (`count: N`) or nested records (`type: record`, `fields: [...]`); `role` (default: the name) is what the
    writer fills in, and values are passed / looked up by dotted role path."""

    def __init__(self, fields: list[dict], endian: str):
        self.fields, self.endian = fields, endian
        self.leaves: list[_Leaf] = []
        self.struct = struct.Struct(endian + self._build(fields, (), ""))
        self.by_role = {lf.role: lf for lf in self.leaves}
        self.kinds = {lf.role: lf.kind for lf in self.leaves}

    def _build(self, fields, names: tuple, roles: str) -> str:
        fmt = ""
        for f in fields:
            t = f["type"]
            if t == "pad":
                fmt += f"{int(f['size'])}x"
                continue
            name = f["name"]
            role = roles + str(f.get("role", name))
            if t == "record":
                fmt += self._build(list(f["fields"]), names + (name,), role + ".")
                continue
            if t not in TYPE_CODES:
                raise ValueError(f"unknown field type {t!r} for field {name!r}")
            count = f.get("count")
            count = None if count is None else int(count)
            if count is not None and count < 1:
                raise ValueError(f"field {name!r}: count must be >= 1")
            fmt += f"{count}{TYPE_CODES[t]}" if count else TYPE_CODES[t]
            self.leaves.append(_Leaf(names + (name,), role, t, count, f.get("default", 0)))
        return fmt

    @property
    def size(self) -> int:
        return self.struct.size

    @property
    def roles(self) -> list[str]:
        return [lf.role for lf in self.leaves]

    @staticmethod
    def _conv(kind: str, v):
        return float(v) if kind.startswith("f") else int(round(v))

    def pack(self, values: dict) -> bytes:
        """values: {role path: value}; an array leaf takes a sequence (padded with its default)."""
        out = []
        for lf in self.leaves:
            v = values.get(lf.role, lf.default)
            if lf.count is None:
                out.append(self._conv(lf.kind, v))
                continue
            seq = list(v) if isinstance(v, (list, tuple)) else [v]
            if len(seq) > lf.count:
                raise ValueError(f"dms.bin field {'.'.join(lf.name)!r} holds {lf.count} values, got {len(seq)}")
            seq += [lf.default] * (lf.count - len(seq))
            out.extend(self._conv(lf.kind, x) for x in seq)
        try:
            return self.struct.pack(*out)
        except struct.error as e:                     # name the offending field (e.g. u32 timestamp overflow)
            i = 0
            for lf in self.leaves:
                for v in out[i:i + (lf.count or 1)]:
                    try:
                        struct.pack(self.endian + TYPE_CODES[lf.kind], v)
                    except struct.error as e1:
                        raise ValueError(f"dms.bin field {'.'.join(lf.name)!r} ({lf.kind}) cannot hold value "
                                         f"{v!r}: {e1}") from e
                i += lf.count or 1
            raise

    def unpack(self, buf: bytes) -> dict:
        """Raw fields by C name, nested records as dicts, arrays as lists."""
        flat = self.struct.unpack(buf)
        out: dict = {}
        i = 0
        for lf in self.leaves:
            n = lf.count or 1
            v = list(flat[i:i + n]) if lf.count else flat[i]
            i += n
            d = out
            for k in lf.name[:-1]:
                d = d.setdefault(k, {})
            d[lf.name[-1]] = v
        return out

    def get(self, raw: dict, role: str, default=None):
        """Value of a role in an unpacked record."""
        lf = self.by_role.get(role)
        if lf is None:
            return default
        d = raw
        for k in lf.name:
            d = d[k]
        return d


@dataclass
class DmsBinLayout:
    spec: dict
    source: str | None = None
    header: _Record = field(init=False)
    obj: _Record = field(init=False)

    def __post_init__(self):
        s = self.spec
        self.endian = ENDIAN[s.get("endian", "little")]
        self.header = _Record(list(s["header_fields"]), self.endian)
        self.obj = _Record(list(s["object_fields"]), self.endian)
        if "num_objects" not in self.header.roles:
            raise ValueError("header_fields must contain num_objects")
        self.bbox_format = s.get("bbox_format", "xywh")
        if self.bbox_format not in BOX_FIELDS:
            raise ValueError(f"bbox_format must be one of {list(BOX_FIELDS)}")
        missing = [n for n in BOX_FIELDS[self.bbox_format] if n not in self.obj.roles]
        if missing:
            raise ValueError(f"object_fields missing box fields {missing} for bbox_format {self.bbox_format}")
        self.conf_scale = float(s.get("conf_scale", 1.0))
        if not self.conf_scale > 0:
            raise ValueError(f"conf_scale must be > 0, got {s.get('conf_scale')!r}")
        self.ts_scale = TS_PER_S[s.get("timestamp_unit", "ms")]
        self.ts_ms_factor = 1000.0 / self.ts_scale           # stored timestamp -> ms
        ts_kind = self.header.kinds.get("timestamp")
        self.ts_ms_is_int = ts_kind is not None and not ts_kind.startswith("f") and self.ts_ms_factor == 1.0
        self.class_ids = {str(k): int(v) for k, v in (s.get("class_ids") or {}).items()}
        names = {}
        for k, v in self.class_ids.items():
            names.setdefault(v, k)
        names.update({int(k): str(v) for k, v in (s.get("class_names") or {}).items()})
        self.class_names = names
        # simulator class -> output class (optional; without it labels are looked up in class_ids directly)
        self.class_map = s.get("class_map")
        if self.class_map is not None:
            for sim_cls, rule in self.class_map.items():
                outs = [rule] if rule is None or isinstance(rule, str) else [rule["below"], rule["otherwise"]]
                for o in outs:
                    if o is not None and o not in self.class_ids:
                        raise ValueError(f"class_map {sim_cls!r} -> {o!r}: not in class_ids")
                if isinstance(rule, dict) and isinstance(rule["threshold"], str) and rule["threshold"] not in s:
                    raise ValueError(f"class_map {sim_cls!r}: threshold key {rule['threshold']!r} not in the layout")
        self.conf_threshold = {str(k): float(v) for k, v in (s.get("conf_threshold") or {}).items()}
        fh = s.get("file_header") or {}
        magic = fh.get("magic") or ""
        if isinstance(magic, str) and magic.lower().startswith("0x"):
            self.magic = bytes.fromhex(magic[2:])
        else:
            self.magic = str(magic).encode("ascii") if not isinstance(magic, bytes) else magic
        self.version = fh.get("version")
        self.version_struct = struct.Struct(self.endian + TYPE_CODES[fh.get("version_type", "u32")])

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "DmsBinLayout":
        p = Path(path) if path else DEFAULT_LAYOUT
        return cls(load_yaml(p), str(p))

    @classmethod
    def coerce(cls, layout: "DmsBinLayout | str | Path | dict | None") -> "DmsBinLayout":
        if isinstance(layout, DmsBinLayout):
            return layout
        if isinstance(layout, dict):
            return cls(layout)
        return cls.from_yaml(layout)

    # --- file header ---
    def file_header_bytes(self) -> bytes:
        b = self.magic
        if self.version is not None:
            b += self.version_struct.pack(int(self.version))
        return b

    # --- boxes ---
    def encode_box(self, xyxy) -> dict:
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        if self.bbox_format in ("xywh", "ltwh"):
            return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    def timestamp_ms(self, stored) -> int | float:
        """Stored header timestamp -> ms: int for an integer field in ms, else a float rounded to 1e-6 ms."""
        if self.ts_ms_is_int:
            return int(stored)
        return round(float(stored) * self.ts_ms_factor, 6)

    def decode_box(self, o: dict) -> list[float]:
        if self.bbox_format in ("xywh", "ltwh"):
            return [o["x"], o["y"], o["x"] + o["w"], o["y"] + o["h"]]
        return [o["x1"], o["y1"], o["x2"], o["y2"]]


def _size(camera: Any) -> tuple[int, int]:
    if isinstance(camera, dict):
        return int(camera["width"]), int(camera["height"])
    return int(camera.width), int(camera.height)


def _frame_objects(L: DmsBinLayout, rec: dict, cam_name, include_face: bool, skipped, below) -> list[dict]:
    """Detections of one frame to write, under the layout's class map / class_ids and conf thresholds."""
    if L.class_map is not None:
        dets, sk, bl = mapped_detections(rec, L.spec, include_face=include_face)
    else:                                    # simulator labels looked up in class_ids
        dets, sk, bl = [], collections.Counter(), collections.Counter()
        for d in to_pipeline(rec, cam_name, include_face=include_face)[FIELDS["od"]]:
            label = d[FIELDS["label"]]
            src = _LABEL_TO_CLASS.get(label, label)
            cls = label if label in L.class_ids else src
            if cls not in L.class_ids:
                sk[cls] += 1
                continue
            conf = float(d[FIELDS["score"]])
            if cls in L.conf_threshold and conf < L.conf_threshold[cls]:
                bl[cls] += 1
                continue
            dets.append({"class": cls, "source_class": src, "bbox": d[FIELDS["bbox"]], "conf": conf})
    skipped.update(sk); below.update(bl)
    return dets


def encode_frames(records, camera, layout=None, include_face: bool = True,
                  stats: dict | None = None) -> Iterator[dict]:
    """Per-frame values of the dms.bin writer, in record order: the one place that applies the layout's class map
    and conf thresholds (write_dms_bin packs them; export.export_frame_yaml prints them).
    Yields {"frame_index", "t" (s), "objects": [{"class", "class_id", "source_class", "bbox" [x1,y1,x2,y2], "conf"}]}.
    stats (optional dict) receives the Counters "skipped" (simulator class not in the layout) and "below_threshold"
    (under conf_threshold), filled as frames are consumed."""
    L = DmsBinLayout.coerce(layout)
    cam_name = camera.get("name") if isinstance(camera, dict) else getattr(camera, "name", None)
    skipped: collections.Counter = collections.Counter()
    below: collections.Counter = collections.Counter()
    if stats is not None:
        stats["skipped"], stats["below_threshold"] = skipped, below

    def frames():
        for rec in records:
            dets = _frame_objects(L, rec, cam_name, include_face, skipped, below)
            yield {"frame_index": int(rec["frame"]), "t": float(rec["t"]),
                   "objects": [dict(d, class_id=L.class_ids[d["class"]]) for d in dets]}
    return frames()


def write_dms_bin(records, camera, layout=None, path="dms.bin", include_face: bool = True) -> dict:
    """Write simulation records to a dms.bin file, one frame record per simulation record (a dropped model frame
    is a zero-object record). Returns stats {"path", "frames", "objects", "skipped": {simulator class: n} (not in
    the layout's class map / class_ids), "below_threshold": {class: n} (under conf_threshold)}.
    include_face=False omits the face record."""
    L = DmsBinLayout.coerce(layout)
    width, height = _size(camera)
    counts: dict = {}
    frames = encode_frames(records, camera, L, include_face, counts)
    n_frames = n_obj = 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(L.file_header_bytes())
        for fr in frames:
            objs = []
            for d in fr["objects"]:
                vals = {"class_id": d["class_id"], "score": d["conf"] * L.conf_scale}
                vals.update(L.encode_box(d["bbox"]))
                objs.append(L.obj.pack(vals))
            hdr = {"timestamp": fr["t"] * L.ts_scale, "frame_index": fr["frame_index"],
                   "width": width, "height": height, "num_objects": len(objs)}
            fh.write(L.header.pack(hdr))
            fh.write(b"".join(objs))
            n_frames += 1
            n_obj += len(objs)
    return {"path": path, "frames": n_frames, "objects": n_obj, "skipped": dict(counts["skipped"]),
            "below_threshold": dict(counts["below_threshold"])}


def read_dms_bin(path, layout=None, strict: bool = True) -> Iterator[dict]:
    """Parse a dms.bin file. Yields one dict per frame:
    {"header": raw header fields by C name (arrays as lists, nested records as dicts),
     "objects": [raw object fields + "class", "bbox_xyxy", "conf"],
     "frame_id", "timestamp_ms", "od": pipeline-style list [{label, bbox [x1,y1,x2,y2], score}]}.
    timestamp_ms is an int for an integer timestamp field in ms, else a float rounded to 1e-6 ms.
    A truncated frame (header or object records cut off) raises ValueError; with strict=False the frames before
    it are yielded and the stream ends with a warnings.warn instead (e.g. a file still being written)."""
    L = DmsBinLayout.coerce(layout)
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:len(L.magic)] != L.magic:
        raise ValueError(f"bad magic: expected {L.magic!r}, got {data[:len(L.magic)]!r}")
    pos = len(L.file_header_bytes())
    if L.version is not None:
        if len(data) < pos:
            raise ValueError("file shorter than its file header")
        (ver,) = L.version_struct.unpack_from(data, len(L.magic))
        if ver != int(L.version):
            raise ValueError(f"version mismatch: file {ver}, layout {L.version}")
    hs, os_ = L.header.size, L.obj.size
    H, O = L.header, L.obj
    n_frames = 0

    def truncated(msg: str):
        msg = f"{msg}; {n_frames} complete frames before it"
        if strict:
            raise ValueError(msg)
        warnings.warn(f"{path}: {msg}", stacklevel=3)

    while pos < len(data):
        if pos + hs > len(data):
            truncated(f"truncated frame header at byte {pos} ({len(data) - pos} of {hs} bytes)")
            return
        h = H.unpack(data[pos:pos + hs]); pos += hs
        n = int(H.get(h, "num_objects"))
        if pos + n * os_ > len(data):
            truncated(f"truncated object records at byte {pos} (frame needs {n} x {os_} bytes, {len(data) - pos} left)")
            return
        objs = []
        for _ in range(n):
            raw = O.unpack(data[pos:pos + os_]); pos += os_
            cid = int(O.get(raw, "class_id", -1))
            o = dict(raw)
            o["class"] = L.class_names.get(cid, f"id{cid}")
            o["bbox_xyxy"] = L.decode_box({r: O.get(raw, r) for r in BOX_FIELDS[L.bbox_format]})
            sc = O.get(raw, "score")
            o["conf"] = float(sc) / L.conf_scale if sc is not None else None
            objs.append(o)
        ts = H.get(h, "timestamp")
        fi = H.get(h, "frame_index")
        n_frames += 1
        yield {
            "header": h, "objects": objs,
            FIELDS["frame_id"]: int(fi) if fi is not None else None,
            FIELDS["timestamp_ms"]: L.timestamp_ms(ts) if ts is not None else None,
            FIELDS["od"]: [{FIELDS["label"]: o["class"], FIELDS["bbox"]: o["bbox_xyxy"], FIELDS["score"]: o["conf"]}
                           for o in objs],
        }


def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def format_frame(fr: dict, layout: DmsBinLayout | None = None) -> str:
    """One frame as human-readable text (used by `dms-sim inspect-bin`). Scalar header fields on the first line;
    arrays and nested records on their own lines."""
    head, extra = [], []
    for k, v in fr["header"].items():
        if isinstance(v, dict):
            extra.append(f"    {k}: " + "  ".join(f"{a}={_fmt(b)}" for a, b in v.items()))
        elif isinstance(v, list):
            extra.append(f"    {k}: [" + ", ".join(_fmt(x) for x in v) + "]")
        else:
            head.append(f"{k}={v}")
    lines = [f"[frame {fr[FIELDS['frame_id']]}] " + "  ".join(head)] + extra
    for o in fr["objects"]:
        raw = "  ".join(f"{k}={_fmt(v)}" for k, v in o.items() if k not in ("class", "bbox_xyxy"))
        lines.append(f"    {o['class']:<12} {raw}")
    return "\n".join(lines)

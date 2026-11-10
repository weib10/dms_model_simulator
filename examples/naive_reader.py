"""The simplest adapter that sits an exam: read each question, write the answer sheet (docs/EXAM.md).

    python examples/naive_reader.py runs/A answers
    dms-sim exam grade runs/A answers --out report

It is an example of the answer-sheet format and a baseline to compare a real program with. It is NOT a reference
post-processing: it has no tracking, no smoothing and raises no alerts. It reads one frame at a time and declares
one `state` signal per behaviour label:

    looking_away   gaze direction outside the row's road box; kept from the last frame when the gaze is not valid
    eyes_shut      mean eye openness below EYE_SHUT
    phone          a phone box is in the frame
    mouth_wide     mouth openness above MOUTH_WIDE
    cigarette      a cigarette box is in the frame
    no_belt        the frame has boxes but no seatbelt box

Standard library only, on purpose: an adapter needs nothing from this package. answer.yaml is written as JSON,
which is valid YAML.
"""
import json
import sys
from pathlib import Path

EYE_SHUT = 0.2        # the simulator's own boundary between an open and a closed eye (configs/behavior.yaml)
MOUTH_WIDE = 0.5      # made up for the example

SIGNALS = {
    "looking_away": {"concept": "eyes_off_road", "kind": "state"},
    "eyes_shut": {"concept": "eyes_closed", "kind": "state"},
    "phone": {"concept": "phone_use", "kind": "state"},
    "mouth_wide": {"concept": "yawning", "kind": "state"},
    "cigarette": {"concept": "smoking", "kind": "state"},
    "no_belt": {"concept": "seatbelt_off", "kind": "state"},
}


def read_row(question_dir: Path) -> list[dict]:
    """One answer line per frame of questions/<row_id>/pipeline.jsonl."""
    row = json.loads((question_dir / "row.json").read_text(encoding="utf-8"))
    (yaw_lo, yaw_hi), (pitch_lo, pitch_hi) = row["road_box"]["yaw_deg"], row["road_box"]["pitch_deg"]
    out, away, no_belt = [], False, False
    with open(question_dir / "pipeline.jsonl", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            labels = {d["label"] for d in rec["od"]}
            gaze = rec["gaze"]
            if gaze and gaze["valid"]:
                away = not (yaw_lo <= gaze["yaw_vehicle"] <= yaw_hi and pitch_lo <= gaze["pitch_vehicle"] <= pitch_hi)
            if labels:                                   # a frame without any box says nothing about the belt
                no_belt = "seatbelt" not in labels
            eyes = [v for v in (rec["eye_openness"] or {}).values() if v is not None]
            mouth = rec["mouth_openness"]
            out.append({"frame_id": rec["frame_id"], "signals": {
                "looking_away": int(away),
                "eyes_shut": int(bool(eyes) and sum(eyes) / len(eyes) < EYE_SHUT),
                "phone": int("phone" in labels),
                "mouth_wide": int(mouth is not None and mouth > MOUTH_WIDE),
                "cigarette": int("cigarette" in labels),
                "no_belt": int(no_belt),
            }})
    return out


def main(exam_dir: str, answers_dir: str) -> None:
    exam = json.loads((Path(exam_dir) / "exam.json").read_text(encoding="utf-8"))
    out = Path(answers_dir)
    out.mkdir(parents=True, exist_ok=True)
    head = {"exam": exam["exam"], "fingerprint_id": exam["fingerprint_id"], "signals": SIGNALS}
    (out / "answer.yaml").write_text(json.dumps(head, indent=1) + "\n", encoding="utf-8")
    for row in exam["rows"]:
        lines = read_row(Path(exam_dir) / "questions" / row["row_id"])
        with open(out / f"{row['row_id']}.jsonl", "w", encoding="utf-8", newline="\n") as fh:
            fh.writelines(json.dumps(x) + "\n" for x in lines)
    print(f"answered {len(exam['rows'])} rows of exam {exam['exam']} -> {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python naive_reader.py <exam dir> <answers dir>")
    main(sys.argv[1], sys.argv[2])

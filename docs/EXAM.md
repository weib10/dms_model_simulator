# Exams for a post-processing program

An exam is a fixed set of simulated drives. You run your post-processing on them, hand in an **answer sheet**, and
the grader compares it with what the driver really did.

- The exam does not know your program. You write a small **adapter** that turns your program's output into the
  answer sheet; how your program reads the model output is up to you.
- The grader holds **no alert thresholds and no alert logic**. It measures (how late, how often, where), reads the
  exam's design (rows built to be obvious) and, if you declare a rule of your own, checks you against that.
- Everything is reproducible: the same exam definition on the same version gives the same questions and the same
  truth to anyone.

```bash
dms-sim exam list
dms-sim exam build A-v1 --out runs/A --jobs 4         # questions/ + key/
python examples/naive_reader.py runs/A answers        # an adapter: yours goes here
dms-sim exam grade runs/A answers --out report        # report.md + report.json
```

`dms-sim exam oracle runs/A --out answers` writes the truth itself as an answer sheet: it must get full marks, so
it checks an installation and shows the format.

## What `build` writes

```
runs/A/
  exam.json                 name, fingerprint_id, one record per row
  questions/<row_id>/       what your program may read
    pipeline.jsonl          model output, one record per frame (field contract: SPEC.md)
    labels/                 the same frames as one YAML per frame, only with --layout
    row.json                fps, frames, camera, image_size, road_box
  key/                      the answer side: truth, full JSONL with error flags, behaviour cards
```

- Feed your program `questions/` only. `key/` is not secret (anyone can rebuild it), which is why an exam meant to
  catch tuning to the questions is built with `mode: random` and a private `--seed`.
- `--camera` builds the exam for another camera position, `--layout` / `--labels mapped` / `--include-face` write the
  model output with your class ids, thresholds and face box. None of them changes the truth.
- `road_box` is calibration, not an answer: the range of vehicle-frame gaze yaw / pitch (degrees) that counts as
  looking at the road in this row. A few rows widen it (a traffic light above the windscreen, a junction scan).
- `fingerprint_id` identifies the build (code version, configs, entries). An answer sheet names the build it was
  made for.

## The answer sheet

```
answers/
  answer.yaml               the declaration
  <row_id>.jsonl            one file per row, one line per frame
```

`answer.yaml`:

```yaml
exam: A-v1
fingerprint_id: 0123456789abcdef        # copy it from exam.json
signals:
  head_away: {concept: eyes_off_road, kind: state}
  distract:  {concept: eyes_off_road, kind: alert}
  drowsy:    {concept: eyes_closed,   kind: alert, rule: {type: continuous, threshold_s: 1.5}}
```

- **Signal names are yours.** Declare as many as your program has.
- **`concept`** is one of the six behaviour labels of the truth: `eyes_off_road`, `eyes_closed`, `phone_use`,
  `yawning`, `smoking`, `seatbelt_off`. Several signals may share a concept (a long-distraction alert and a
  cumulative one). A concept without a signal is reported as not answered, not as wrong.
- **`kind: state`** follows the behaviour: on while the driver does it. **`kind: alert`** is raised when your program
  decides to. A program that only raises alerts declares alerts only.
- **`evidence`** (optional) lists the channels this signal can read, see [Observability](#observability).
- **`rule`** (optional, alerts only) is your own specification, see [Your own rule](#your-own-rule).

`<row_id>.jsonl`:

```json
{"frame_id": 0, "signals": {"head_away": 0, "distract": 0, "drowsy": 0}}
```

- One line for every frame of the question, in order, also for frames without model output. Every declared signal
  in every line, as 0 or 1.
- Alerts count where the signal turns from 0 to 1, so a latched alert and a one-frame pulse grade alike. A program
  that emits an event list writes a 1 on the frame of each event.
- You may answer some rows only; the rest are listed as missing. A malformed file is refused, not guessed at.

JSON is valid YAML, so `answer.yaml` can be written with a JSON library (the example adapter does).

## Observability

The truth says what the driver does, whether or not a camera can see it. A phone in the lap, a mouth behind a hand,
the eyes of a head turned away: no detector could report them, and a post-processing must not be blamed for them.
Every truth frame therefore carries `observable`, one flag per evidence channel:

| channel | true when |
|---|---|
| `face`, `head_pose` | the face box is in the image |
| `head_away` | the head pose is there and the head itself points outside the road box |
| `gaze` | the gaze model gives a direction (eyes open enough, an eye visible, not too far from the camera) |
| `eyes`, `mouth`, `hand`, `phone`, `cigarette` | that box is in the truth |
| `seatbelt_region` | the place where a worn belt is boxed is in the image |

The flags describe a detector that misses nothing, which is exactly the `exact` error profile: there the model
output holds what is observable and nothing else.

A truth frame counts for a signal when at least one channel of its evidence is observable. Defaults
(`exams/grading.yaml`):

| concept | evidence |
|---|---|
| `eyes_off_road` | `gaze` or `head_away` |
| `eyes_closed` | `eyes` |
| `phone_use` | `phone` |
| `yawning` | `mouth` |
| `smoking` | `cigarette` |
| `seatbelt_off` | `seatbelt_region` |

A program that reads boxes only can say so (`evidence: [head_away]`): an eyes-only glance then no longer counts
against it. Frames without evidence where the signal was on anyway are reported apart.

## What the report measures

No single score. Every table states how many rows or segments it rests on; `report.json` has every segment and
every alert frame of every row.

**State signals**, per truth segment that has evidence:

- detected or not; onset delay (from the first frame with evidence) and release delay; dropouts (the signal fell
  off inside the segment);
- false segments: the signal on with no truth nearby;
- per-frame recall and precision, leaving out a short band around the start and end of each segment.

Segments are the main measure. Per-frame recall punishes what cannot be known yet: a yawn's truth label starts
when the mouth begins to open, about a second before it is wide.

**Alert signals**, without any specification:

- every alert is `during` a behaviour of its concept (or shortly after it), `after` one, or `unprovoked`;
- rows where the behaviour never occurs and the alert came anyway: wrong under any threshold;
- delay from the first frame with evidence to the alert: your program's effective threshold;
- behaviours that drew more than one alert;
- **rows built to be obvious**, in exams marked `intent: true`: POS rows should raise an alert of their main
  behaviour, NEG rows none. This is the exam author's design, not a specification; a program that by design needs
  three yawns before it speaks will fail the single-yawn row and knows why;
- **parameter sweeps** (BND entries): along a duration or angle sweep, between which two values the response
  switches, and whether it switches more than once.

### Your own rule

`rule: {type: continuous, threshold_s: N}` declares "this alert comes once the behaviour has lasted N seconds".
The grader then judges each truth segment by your number: hit, miss or early for segments clearly longer than N,
false or silent for clearly shorter ones, not judged within the tolerance; and the latency after the rule is met.
The repository ships no thresholds of its own. Only this rule shape exists so far.

## Exam definitions

`exams/<name>.yaml`:

```yaml
name: A-v1
title: "Must-test behaviours, easiest level, no model errors"
intent: true                    # rows are obvious by design (see above)
expand:                         # arguments of `dms-sim catalog expand`
  level: [small]
  profile: [exact]
  emulator_seeds: 1
  camera: [dash_right]
entries: [POS-PHN-BASIC-CALL, NEG-DIS-MIRROR-CHECK, ...]
```

The entry list is frozen in the file. A changed list is a new version of the exam.

## Limits

- The grader's times (`exams/grading.yaml`: edge band, alert grace, lookback, rule tolerance) are provisional.
- `head_away` uses the road box on the head's facing direction. A head turned a little, with the eyes doing the
  rest and the gaze not valid, has no evidence under the defaults.
- `seatbelt_off` is graded on the absence of a box; the truth has no notion of a belt worn wrongly.
- A layout's score thresholds can drop boxes the truth marks observable; that is your model configuration, and the
  grader does not correct for it.
- Comparing two reports (the same rows with and without a model error, or two versions of a program) is not built
  yet; rows that differ only in model errors share `pair_id` in `exam.json`.

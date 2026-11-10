# Exams for a post-processing program

An exam is a fixed set of simulated drives. You run your post-processing on them, hand in an **answer sheet**, and
the grader compares it with what the driver really did.

- The exam does not know your program. You write a small **adapter** that turns your program's output into the
  answer sheet; how your program reads the model output is up to you.
- The grader holds **no alert thresholds and no alert logic**. It measures (how late, how often, where), reads the
  exam's design (rows built to be obvious) and, if you declare a rule of your own, checks you against that.
- Everything is reproducible: the same exam definition on the same version gives the same questions and the same
  truth to anyone.

Contents: [A first run](#a-first-run) · [What `build` writes](#what-build-writes) ·
[Finding the answer](#finding-the-answer) · [The answer sheet](#the-answer-sheet) · [Observability](#observability) ·
[What the report measures](#what-the-report-measures) · [Exam definitions](#exam-definitions) · [Limits](#limits)

## A first run

**1. Build the exam and check the installation.** `oracle` writes the truth itself as an answer sheet; grading it
must give full marks.

```bash
dms-sim exam list                                                   # the exams in exams/
dms-sim exam build A-v1 --out runs/A --jobs 4                       # questions/ + key/, about a minute
dms-sim exam oracle runs/A --out runs/A-oracle
dms-sim exam grade runs/A runs/A-oracle --out runs/A-oracle-report
```

`runs/A-oracle-report/report.md` shows every segment detected, every delay `0.00`, and `Nothing` under
*Where to look*.

**2. See a real answer sheet.** The example adapter reads one frame at a time and raises no alerts:

```bash
python examples/naive_reader.py runs/A runs/A-naive
dms-sim exam grade runs/A runs/A-naive --out runs/A-naive-report
```

Its report is not perfect, and says where. *Where to look* has one line, a glance `looking_away` missed in
`NEG-ALL-HABITS-COMBO` at 6.60-7.20 s (the driver looks down at the instruments while rubbing an eye; the reader
only follows the gaze), and the table above it shows `mouth_wide` coming on a second late (the yawn label starts
when the mouth begins to open).

**3. Write your adapter.** For every row in `runs/A/questions/`, run your program on `pipeline.jsonl` (or on
`labels/`, see `--layout`), and write one line per frame saying which of your signals are on
([The answer sheet](#the-answer-sheet)). `examples/naive_reader.py` is 80 lines to copy from.

**4. Grade and read the report** from the top: the signals you declared, how each follows the driver
(*Reading the driver state*, *Alerts*), then *Where to look*: every miss and every false response with its row and
time. Look that row up in the answer key ([Finding the answer](#finding-the-answer)).

## What `build` writes

```
runs/A/
  exam.json                 name, fingerprint_id, one record per row
  questions/<row_id>/       what your program may read
    pipeline.jsonl          model output, one record per frame (field contract: SPEC.md)
    labels/                 the same frames as one YAML per frame, only with --layout
    row.json                fps, frames, camera, image_size, road_box
  key/                      the answer side
    answer_key.md           every row: when each behaviour is on, in seconds
    rows/<row_id>/          truth per frame, full JSONL with error flags, behaviour card
      timeline.png          with --timeline: gaze zone, eye openness, behaviour labels, injected errors over time
      preview.gif           with --gif: the camera view, truth boxes green, model boxes red
```

- Feed your program `questions/` only. `key/` is not secret (anyone can rebuild it), which is why an exam meant to
  catch tuning to the questions is built with `mode: random` and a private `--seed`.
- `--camera` builds the exam for another camera position, `--layout` / `--labels mapped` / `--include-face` write the
  model output with your class ids, thresholds and face box. None of them changes the truth.
- `--timeline` / `--gif` add a picture of every row to `key/` and `answer_key.md` shows them under the row: the
  quickest way to explain a row to someone who will not read JSONL. They change nothing else (`--gif-every`,
  `--gif-scale` set the GIF's frame step and size; with both, A-v1 builds in about a minute at `--jobs 4` and its
  GIFs take ~40 MB).
- `road_box` is calibration, not an answer: the range of vehicle-frame gaze yaw / pitch (degrees) that counts as
  looking at the road in this row. A few rows widen it (a traffic light above the windscreen, a junction scan).
- `fingerprint_id` identifies the build (code version, configs, entries). An answer sheet names the build it was
  made for.

## Finding the answer

The answer to a row is what the driver does in it, frame by frame. Four views of the same truth:

| you want | look at |
|---|---|
| all rows at a glance | `key/answer_key.md`: per row, the spans in seconds where each behaviour label is on, and which part of that the camera has no evidence of |
| one row's script | `key/rows/<row_id>/card.md`: what the driver was told to do when, and the model errors injected where |
| one row as a picture | `key/rows/<row_id>/timeline.png` and `preview.gif`, when built with `--timeline` / `--gif` |
| every frame | `key/rows/<row_id>/*.truth.jsonl`: `events` holds the six labels, `observable` what the camera can show (fields: [SPEC.md](SPEC.md)) |
| the truth in your own format | `dms-sim exam oracle runs/A --out DIR`: `DIR/<row_id>.jsonl` has the same lines as your answer file, to compare one by one |

An entry of `answer_key.md`:

```
## `POS-DRW-BASIC-CLOSURE__small__d52e4a06__dash_right__vn__e0__exact`

<entry title> (22.0 s, 440 frames at 20 fps, camera `dash_right`, error profile `exact`). Built to raise an alert of `eyes_closed`.

| behaviour | on (s) | of that, no evidence (s) |
|---|---|---|
| eyes_closed | 2.45-2.55, 3.95-4.05, 7.20-7.30, 8.00-12.00, 14.80-14.90, 21.80-21.90 | none |
```

A span runs from its first frame to the first frame after it: `8.00-12.00` at 20 fps is frames 160 to 239. The
short spans are blinks: `eyes_closed` says the eyes are shut, not for how long.

From a line of the report to the answer: *Where to look* names the row, the signal and the time; open that row in
`answer_key.md`, then its `card.md` if the reason is not plain.

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

The last section, **Where to look**, lists every case the tables count against a signal, with its row and time: a
behaviour a state signal missed or fell off during, a signal on with no behaviour near, an alert with none of its
behaviour before it, and the verdicts against your own rule. Delays are not listed: they are measurements, and
whether 1.5 s is late is yours to say.

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

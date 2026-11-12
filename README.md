# dms_model_simulator

Simulates the frame-by-frame output of a driver monitoring system (DMS) object-detection model for a scripted
driver in a car cabin. The output includes the errors a real model makes:

- missed boxes (also in bursts)
- box jitter and jumps
- open / closed eye and mouth classes reported wrong
- false positives
- dropped frames and latency

**Purpose: tune post-processing.** Feed the simulated output to your alert logic (tracking, time windows,
hysteresis) and see how it copes with realistic model errors.

- The simulator contains no alert logic and no alert thresholds.
- **Exams** grade a post-processing program: you hand in what it reported per frame, the grader compares it with
  what the driver did ([Exams](#exams), [`docs/EXAM.md`](docs/EXAM.md)).
- Each run comes with a **behaviour card**: what the driver did when, and which model errors were injected where.
  Compare your alerts with it.
- It does not try to match one particular model: error levels are tunable and graded (see [Error profiles](#error-profiles)).

## Contents

- [Requirements](#requirements)
- [Quick start](#quick-start)
- [What it produces](#what-it-produces)
- [Checking a post-processing program](#checking-a-post-processing-program)
- [Exams](#exams)
- [Scenarios](#scenarios)
- [Model errors](#model-errors)
- [Outputs](#outputs)
- [Reproducibility](#reproducibility)
- [Conventions](#conventions)
- [Known limitations](#known-limitations)

## Requirements

| what | version | notes |
|---|---|---|
| Python | **3.10 or newer** | macOS ships 3.9 as `/usr/bin/python3`, which is too old: install a newer one (python.org installer, Homebrew or uv) |
| numpy | 1.25 or newer | installed by pip |
| pyyaml, matplotlib, pillow | any recent | installed by pip |
| pytest | 7 or newer | only for the tests (`.[test]`) |

- Pure Python, CPU only: no GPU, no compiler, no system libraries. Runs on Windows, Linux and macOS (see [Tested on](#tested-on)).
- Disk: the install is ~150 MB (mostly numpy and matplotlib); outputs are small except GIFs and `--image render`.
- Windows: keep output directories short (e.g. `runs/`): catalog row paths are long and the 260-character path
  limit makes deep ones fail.

### Tested on

`pip install -e ".[test]"`, `dms-sim demo` and the full test suite, from a fresh clone except on the Windows 11
development machine (2026-11-09; numpy 2.5.4, pyyaml 6.0.3, matplotlib 3.11.2, pillow 13.0.0 everywhere):

| OS | Python | tests |
|---|---|---|
| Windows 11 (x86_64) | 3.12.10 | 192 passed, 1 skipped |
| Windows 10 (x86_64) | 3.12.10 | 192 passed, 1 skipped |
| Ubuntu 26.04 on WSL2 (x86_64) | 3.14.4 | 192 passed, 1 skipped |
| macOS 15.5 (Apple silicon) | 3.12.16 (via uv) | 192 passed, 1 skipped |

The skipped test is a timing test (skipped when `DMS_SKIP_TIMING=1`).

No Python 3.10+ on the machine? [uv](https://docs.astral.sh/uv/) fetches one without touching the system Python:

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv -e ".[test]"
```

## Quick start

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[test]"     # Linux / macOS: .venv/bin/pip
dms-sim demo --scenario phone_distraction --out out/demo           # every camera: outputs, cards, timelines, GIFs
dms-sim export --scenario phone_distraction --camera dash_right --format frame_yaml --profile bad --out out/x
dms-sim catalog expand POS-PHN --out runs/r1 && dms-sim catalog run runs/r1 --jobs 4   # one card per row
dms-sim catalog run runs/r1 --jobs 4 --timeline --gif              # and a timeline picture + camera-view GIF per row
dms-sim catalog gallery runs/r1                                    # runs/r1/index.html + index.en.html: every entry with its pictures
```

More commands:

```bash
dms-sim run --scenario drowsy --camera steering_column --format all --out out     # JSONL + dms.bin
dms-sim render --scenario mirror_check --camera all --out out                     # GIF, one tile per camera (real time, --view-fps 25 default)
dms-sim render --scenario face_vs_eyes --fps 25 --out out                          # face direction vs eye gaze, every frame at 25 fps
dms-sim render --scenario face_vs_eyes --fps 25 --story --out out                  # the same as a story: top-down view in plain words
dms-sim compare --scenario phone_distraction                                      # model-vs-truth table per camera
dms-sim cabin --out out                                                           # cabin with all cameras
dms-sim inspect-bin out/drowsy__steering_column.dms.bin --head 3                  # dms.bin as text
dms-sim gen-scenario --duration 120 --seed 1                                      # random scenario
dms-sim batch --gen 3 --variants 5 --out out/batch                                # scenarios x drivers x cameras
dms-sim catalog list                                                              # scenario catalog (catalog/README.md)
dms-sim catalog review                                                            # catalog/REVIEW.md, .en.md, .csv, .en.csv: one row per entry
python -m pytest -q tests                                                         # several minutes
```

### Common options

| option | meaning |
|---|---|
| `--seed` | same seed = identical output, and the same behaviour on every camera |
| `--emulator-seed` | same behaviour, different model errors |
| `--profile` | error profile (see [Error profiles](#error-profiles)) |
| `--fps` | default 30; the `export` / `catalog` default is 20 |
| `--variant N` | driver and camera-mount variant from `configs/variation.yaml` |
| `--layout` | binary / class layout, default `configs/binlog.yaml` |
| `--labels sim\|mapped` | pipeline JSONL labels: simulator classes, or the classes of the layout's `class_map` |
| `--include-face` / `--no-face` | include or leave out the face box |

## What it produces

Per frame, for one camera:

- **Model output**
  - OD boxes with scores: `face`, `left_eye`, `right_eye`, `mouth`, `hand`, `phone`, `cigarette`, `seatbelt`
    (optionally mapped to your own classes, e.g. open / closed eye)
  - head pose, gaze (face direction and eye gaze), eye / mouth openness
- **Ground truth**, always from the script, never from the model
  - exact boxes, landmarks, head pose, gaze (face direction and eye gaze), openness
  - hand / phone / cigarette / seatbelt states
  - behaviour labels: `eyes_off_road`, `off_road_s`, `eyes_closed`, `closed_s`, `phone_use`, `phone_use_visible`,
    `yawning`, `smoking`, `seatbelt_off`
  - `observable`: per evidence channel (gaze, head pose, each box), whether any detector could report it this frame
- **Files**: pipeline JSONL (model output), truth JSONL, full JSONL (both side by side, with error flags), a packed
  binary log (`dms.bin`, layout in YAML) or one YAML per frame (`frame_yaml`). Details in [Outputs](#outputs).
- **Behaviour card** (`card.md`, `card.json`): timeline of what the driver does, plus the injected errors.
- **Pictures**: wireframe camera view, cabin plot, timeline plot, GIFs (see [Reading the pictures](#reading-the-pictures)).

### Reading the pictures

Every view uses one code. Colour is the quantity: **orange = eyes** (where the eyes look), **blue = face** (where the
nose points). Line style is the source: **truth solid and filled, model dashed and hollow**. Shape says it again: the
face is a short wedge ("nose"), the eyes a long sight line with an arrowhead. Boxes: truth solid green, model dashed
red. A direction along the line of sight (at the camera, or out of a cabin plot) is drawn as a ring: a dot inside =
towards you, a cross = away.

The camera view is a mirror image (the driver's left is on the right). Its HUD reads in the vehicle frame first, in
words: `truth: face 31° L | eyes 0° | eyes−face 31° R`, then `model: face 27° L (err 4°) | eyes 4° L (err 5°)`; the
camera-frame numbers are on the dim last line.

**Story view** (for people who don't read JSONL): `dms-sim render --scenario face_vs_eyes --story` writes, instead
of the engineering view, `story.gif` (one large top-down picture per frame, real-time playback at `--view-fps`) and
`story_sheet.png` (up to four key moments: the middle of each scenario segment, picked where face and eyes differ
most or the eyes are off the road). Under each picture: the directions in plain words and the model error. Each
moment's title is the segment's optional `caption:` key, else a description such as "face 30° left, eyes ahead".
One camera only; `--scale` sets the size (0.5 = 960x720).

## Checking a post-processing program

1. **Export the runs.** `dms-sim export ...` for one scenario, or `dms-sim catalog expand ... && dms-sim catalog run ...`
   for many (`rows.tsv` lists every row's frames, truth and card).
2. **Run your post-processing** on the exported model output: the `frame_yaml` labels directory, `dms.bin`, or the
   pipeline JSONL through `dms_model_simulator.replay`:

   ```python
   from dms_model_simulator.replay import iter_pipeline
   for rec in iter_pipeline("out/x/phone_distraction__dash_right.pipeline.jsonl"):
       my_logic.step(rec)              # rec: frame_id, timestamp_ms, od [{label, bbox, score}], head_pose, gaze, ...
   ```

3. **Compare its alerts with the behaviour card.** The timeline says what the driver did when; the error list says
   where the model output was degraded on purpose. An alert that drops during a burst of missed face boxes is a
   different finding from one that never fires.

## Exams

An exam is a frozen set of catalog rows plus a grader. You write an adapter that turns your program's output into
an answer sheet (named signals, each tied to one behaviour label, 0 / 1 per frame); the grader reports how each
signal follows the truth. It has no thresholds of its own: delays are measured, and what should raise an alert
comes from the exam's design or from a rule you declare.

```bash
dms-sim exam build A-v1 --out runs/A --jobs 4       # questions/ (model output) + key/ (truth)
python examples/naive_reader.py runs/A answers      # an example adapter: replace it with yours
dms-sim exam grade runs/A answers --out report      # report.md + report.json
```

- The answers are in `runs/A/key/answer_key.md`: per row, when the driver does what. The report ends with every
  miss and false response, each with its row and time, to look up there.
- `dms-sim exam oracle runs/A --out DIR` writes the truth as an answer sheet: grading it must give full marks, which
  checks an installation.
- Truth frames no detector could see (a phone in the lap, a mouth behind a hand) do not count against a signal.
- `--camera`, `--layout`, `--labels mapped` and `--include-face` build the same exam for your camera and classes.
- `--timeline` and `--gif` add a picture of every row to the answer side; `answer_key.md` then shows them.
- `examples/naive_reader.py` reads one frame at a time and raises no alerts: a format example and a baseline, not a
  reference post-processing.

The answer-sheet format and every measure of the report are in [`docs/EXAM.md`](docs/EXAM.md).

## Scenarios

### Scenario files

`scenarios/*.yaml`: `mirror_check`, `phone_distraction`, `drowsy`, `smoking`, `hand_to_face`, `random_120s`.

A scenario is a list of segments. Each segment needs `target` (a gaze target from `configs/cabin.yaml`) and
`duration` (s). Optional fields (defaults in `configs/behavior.yaml`):

| field | values |
|---|---|
| `head_pose`, `eye_openness`, `closure`, `blink`, `mouth_openness` | see `configs/behavior.yaml` |
| `yawn` | rate / shape overrides; `{start: true, duration_s, peak_openness}` starts one scripted yawn at the segment (sets the truth label `yawning`) |
| `hand` | `wheel` \| `lap` \| `phone` \| `call` \| `smoke` \| `at_ear` \| `at_mouth` \| `at_eye` \| `mirror` \| `wheel_cross_l` / `_c` / `_r` \| `chest_left` |
| `phone` | `none` \| `lap` \| `call` |
| `cigarette` | `none` \| `mouth` \| `hand` |
| `seatbelt` | `on` \| `off` |
| `held` | `none` \| `cup` \| `bottle` (with `hand: at_mouth`) \| `lollipop` \| `toothpick` \| `straw` \| `pen` (in the mouth, no hand) |

`gen-scenario` draws a Markov chain over the gaze targets, with phone, drowsiness, smoking and seatbelt-off
episodes (`configs/scenario_gen.yaml`).

### Catalog

`catalog/` holds ~70 parameterised scenarios (see `catalog/README.md` and `catalog/INDEX.md`):

- behaviours that should be noticed, and look-alikes that should not
- duration sweeps, stop-and-repeat
- degraded detection
- long natural drives
- Euro NCAP style cases

### Cameras and variants

Camera presets (`configs/cameras.yaml`): `steering_column`, `cluster`, `rearview_mirror`, `a_pillar_left`,
`center_stack`, `overhead_console`, `dash_right` (Full-HD, wide lens).

A scenario may carry a `cabin:` block (driver position / size) and a `camera:` block (mount error);
`--variant N` draws them from `configs/variation.yaml`.

## Model errors

`configs/emulator.yaml` holds the error model. Always on:

- detection probability vs box size / head yaw / occlusion
- box and score noise
- head pose / gaze (face direction and eye gaze, independent) / openness noise
- dropped frames and latency

Optional modes, off by default. Each is a two-state Markov chain with a long-run share and a mean run length.

Numbers given in frames (every `temporal_rho`, the mean run lengths, `latency_frames`) are meant at
`reference_fps: 20`. A run at another fps converts them so the noise and the error runs last as long in seconds:
rho -> rho ** (20 / fps), runs and latency x fps / 20. Shares and probabilities (`drop_frame_prob`, `p_*`,
`rate_per_frame`) are not converted. Details in [`docs/SPEC.md`](docs/SPEC.md).

| mode | config key |
|---|---|
| burst misses | `detection.<class>.mean_miss_frames` |
| burst drops | `pipeline.drop_mean_frames` |
| eye / mouth open-closed class flips | `eye_state_flip`, `mouth_state_flip` |
| context-dependent false positives: hand at the ear → phone, hand at the mouth → cigarette, bottle at the lips → phone | `contextual_false_positives` |
| box jumps | `box_jump` |
| view-dependent jitter | `box_conditions` |

### Error profiles

`catalog/profiles.yaml` holds named overrides, graded per dimension:

| profile | dimension |
|---|---|
| `jit1..3` | box jitter |
| `jmp1..3` | box jumps |
| `mis1..3` | burst misses |
| `drp1..3` | dropped frames |
| `lat1..2` | latency |
| `cnf1..2` | scores near typical thresholds |
| `flp1..2` | class flips |
| `fp1..2` | false positives |
| `bad`, `worst` | bundles |
| `ideal`, `exact` | logic tests. `ideal`: no dropped frames, false positives or score noise; box corner noise and misses in hard views (strong yaw, occlusion, small boxes) remain. `exact` (= `ideal+pose0+alldet`): also no head pose / gaze (face and eye) noise, and every box of the truth is reported, so the output holds exactly what the truth marks `observable` |

Stack them with `+`, e.g. `--profile jit2+jmp2+mis1`. Use them on the command line, in `catalog expand --profile`,
or in a catalog entry's `variation.profiles`.

The levels are deliberately rough: they show how much model error a post-processing tolerates, not what one
particular model does.

## Outputs

`dms-sim run --format full|pipeline|both|all|dmsbin` writes files named `<scenario>__<camera>` plus:

| suffix | content |
|---|---|
| `.jsonl` | full JSONL |
| `.pipeline.jsonl` + `.truth.jsonl` | model output + ground truth |
| `.dms.bin` | packed binary log |
| `__camera.json`, `__metrics.json` | camera parameters, metrics |

All files share `frame_id`; `timestamp_ms` = frame / fps. The field contract is in [`docs/SPEC.md`](docs/SPEC.md).

- **Pipeline JSONL**: `frame_id`, `timestamp_ms`, `camera_id`, `od` [`{label, bbox [x1,y1,x2,y2], score}`],
  `head_pose`, `gaze` (`{face, eye}`: where the head points and the eye gaze, each with yaw / pitch, vector,
  `valid`, score, vehicle angles and zone; `face` is valid whenever the face is found), `eye_openness`,
  `mouth_openness`. A dropped frame has an empty `od` and null fields. Format change (0.2.0): `gaze` in the
  pipeline and truth JSONL became `{face, eye}`; before it was the flat eye gaze, now `gaze.eye`.
- **Truth JSONL**: exact values, the behaviour labels under `events`, and `observable`.
- **Full JSONL**: `{frame, t, camera, truth, pred}` with diagnostics. `pred` carries the error flags `dropped`,
  `latency_fill`, `source_frame`, per-box `jump` / `false_positive` / `p_detect`, and `class_flip`.
- **dms.bin**: one header + N object records per frame. Every field, class id, class map, per-class score threshold
  and the box format come from a YAML layout (`configs/binlog.yaml` documents the syntax). Header fields other than
  timestamp / frame index / size / object count are written as 0.
- **frame_yaml**: `<stem>.frames/labels/<frame>.yaml` with `detections` (class_id, confidence, left, top, width,
  height, class_name) and `timestamp_msec`. `--image render` adds the camera view per frame.
- **Pictures of a catalog or exam row** (`catalog run` / `exam build` with `--timeline`, `--gif`; off by default):
  `timeline.png` (truth gaze zone, eye openness, truth behaviour labels and injected model errors over time) and
  `preview.gif` (the camera view, truth boxes green, model boxes red). They are drawn from the row's own
  simulation, change no other file and are left out of `catalog verify`. A GIF of a 5-minute row is about 20 MB.
- **Overview page** (`dms-sim catalog gallery RUN`): `RUN/index.html`, to browse a run one thing at a time: a map
  (behaviour family x category, a count and the entry numbers per cell), the list of one family, then one entry with
  one of its rows (script, camera view, timeline) and previous / next. Every entry has a fixed number (`number: 45` in
  its YAML, shown as `#045` on the page, in `catalog/INDEX.md` and in `catalog list`; never changed or reused, a new
  entry takes the largest + 1), and the search box finds it by number. The bar on top stays on screen and marks the
  current family; on a wide screen an entry page also lists its family on the left. The home page says what ★
  means and where the entries come from (their `source`, counted from the catalog). `catalog/README.md` has the
  commands that build it for the whole catalog.
  The command writes two pages by default: `RUN/index.html` (Chinese) and `RUN/index.en.html` (English);
  `--lang zh|en` writes only one. Each links to the other at the same place (`#...`). The English page uses the
  entries' `title_en` / `summary_en` and the behaviours' `name_en` (the Chinese where they are missing) and builds
  the script lines from each row's `card.json` keys; `card.md`, `card.json`, `INDEX.md` and exams stay Chinese.
  Standards (`catalog/standards.yaml`) get a page each, reached from the bar on top: their requirements in three
  groups (scenarios in the catalog relate to it / no scenario yet / out of the simulator's reach). Entries with a
  `standards` field carry the standard's code as a small mark; the mark means related, not compliant. The public
  file holds neutral markers only (an id such as `STD-A`, requirement ids, coverage); names and requirement texts
  can come from an optional, git-ignored `catalog/standards.local.yaml`, which the gallery uses when it exists
  (`--no-local` for the markers only).

## Reproducibility

Four independent random sources, each with its own seed:

| source | decides | set with |
|---|---|---|
| behaviour seed | blink timing, random events, what `gen-scenario` generates | `--seed`, entry `variation.seeds` |
| model-error seed | box jitter, misses, false positives, head pose / gaze noise | `--emulator-seed`, entry `variation.emulator_seeds` |
| variant N | driver position and size, camera mount error | `--variant N` |
| expand seed | parameter draws in `catalog expand --mode random` only | `catalog expand --seed` |

They do not disturb each other: enabling yawns does not shift the model noise, each box class has its own noise
stream, and an entry's random parameters do not depend on which other entries are selected. The default
`boundary` / `grid` expansion draws nothing.

**What makes two runs the same test.** `catalog expand` writes a fingerprint into `manifest.json`: package
version, git commit (and whether code, configs or catalog have uncommitted changes), numpy version, and a hash of
every config file, of `catalog/profiles.yaml` / `behaviors.yaml` and of each selected entry. Every behaviour card
prints its short id. A row id already encodes the entry, level, parameters, behaviour seed, scenario content,
camera, variant, model-error seed and profile; the fingerprint adds code and configs.

**Same machine:** identical inputs give byte-identical files. **Across operating systems:** text files are always
written with LF, but the maths libraries may differ in the last digit of a float (measured Windows vs Linux:
largest difference below 1e-12, no difference in which boxes are detected or in any label). Compare runs with:

```bash
dms-sim catalog verify runs/benchmark runs/rerun      # fingerprint, row ids, every row (numbers within --tol, default 1e-9)
```

It prints what differs and exits non-zero unless both runs are the same test with the same output. For a
standard benchmark the simplest is to generate it once with a tagged version and share the files.

## Conventions

| frame | axes | notes |
|---|---|---|
| vehicle | X forward, Y left, Z up, metres | origin = nominal driver eye point |
| head | X out of the face, Y left, Z up | origin = eye midpoint; rotates about `driver.head_pivot` |
| camera (OpenCV) | x right, y down, z forward | `R_cam_from_vehicle` in `<run>__camera.json` |
| camera "face frame" | forward = reversed optical axis, left = image right, up = image up | head pose / gaze (face and eye) `*_camera` are 0 when parallel to the reversed optical axis; yaw>0 = subject's own left, pitch>0 = up |

- **Angles**: vehicle-frame yaw>0 = left, pitch>0 = up; head pose = `Rz(yaw) Ry(-pitch) Rx(roll)`.
- **Boxes**: `[x1, y1, x2, y2]` pixels in the full image; scores 0..1.
- **Box extents** follow a labelling convention (`box_style: annotation` in `configs/cabin.yaml`):
  - the face box follows the face contour from forehead to chin, without the ears;
  - the eye box covers the whole eye socket (open / closed is a class, not the box height).
- **Rendering** is a schematic wireframe: truth boxes green, model boxes red.

## Known limitations

- Phone, hand and cigarette boxes have no visibility test: they are output even when the head hides them.
- A hand is a sphere (plus a cylinder for the forearm in some poses), not a hand shape. Its size is
  `driver.hand_scale` in `configs/cabin.yaml` (a placeholder until measured on real recordings): it sets how much
  of the face a hand hides, e.g. while rubbing an eye. Entries that use it take `--param hand_scale=…`, and
  `configs/variation.yaml` `hands:` can vary it per `--variant` (own random stream).
- The `dash_right` optical axis does not point at the eye point, so the viewing angle the error model uses is
  smaller than the true one for that camera.
- Landmarks outside the image count as visible; the far eye can reappear after a large head turn.
- At full mouth opening the upper lip leaves the mouth box.
- `head_scale` does not scale the cigarette or held objects.
- Burst misses do not keep their configured long-run share when the detection probability changes with the view.
- The `ideal` profile still misses some boxes.
- A box that jumps out of the image is dropped (counted as a miss).

## License

See [LICENSE](LICENSE).

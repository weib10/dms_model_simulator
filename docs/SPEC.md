# dms_model_simulator — specification

## Purpose
Simulate the frame-by-frame output of a DMS object-detection model (plus head pose, gaze and openness) for a
scripted driver, with realistic and tunable model errors, so that post-processing (alert logic) can be tuned
against it. Alert logic and alert thresholds are out of scope: the simulator writes model outputs, ground truth and
a behaviour card. An exam (`docs/EXAM.md`) grades what a post-processing reports against the truth; what should
raise an alert stays with whoever owns the post-processing. Not a goal: photoreal images, training data, or
an exact copy of one particular model.

## Decisions
| topic | decision |
|---|---|
| OD model output | 8 simulator classes, each `{label, bbox [x1,y1,x2,y2] px (full image, clipped), score}`: **face** (off by default in the pipeline JSONL, on in dms.bin; `--include-face` / `--no-face`), **left_eye**, **right_eye**, **mouth**, **hand**, **phone**, **cigarette** (at the lips or in the hand), **seatbelt** (only while worn, plus optional false positives when off). A layout `class_map` maps them to consumer classes (e.g. open / closed eye from the predicted openness) for dms.bin, frame_yaml and `--labels mapped`. |
| OD box extents | `box_style: annotation` (default): face = forehead to chin along the face contour, ears excluded; eye = whole socket (height does not follow openness); mouth = philtrum to below the lower lip, height grows with the lip gap; a part the head hides is cut off; an eye / mouth with no more than `min_visible_fraction` visible gets no box. `geometric` = tight landmark boxes. |
| other outputs | head pose (face frame) with score; gaze as two directions, `face` (where the head points) and `eye` (the eye gaze), each with yaw / pitch, unit vector, `valid`, score, vehicle-frame angles and zone; eye openness per eye and mouth openness (0..1). |
| ground truth | exact geometry; the emulator never changes it. Truth and model output share `frame_id`. |
| behaviour labels | derived from the script (`events.py`, `configs/behavior.yaml` `events`), never from the model, and without duration thresholds: they say what the driver does, not what an alert should decide. |
| observability | per frame and evidence channel, whether a detector that misses nothing could report it (`observable`); the `exact` profile outputs exactly that. Exams count a truth frame only when its evidence is observable. |
| behaviour card | per run: segment timeline (time and frame ranges, what the driver does) and the injected model errors as frame runs per kind and class (`behavior_card.py`). |
| tunability | every geometric, behavioural, error-model and binary-layout number lives in YAML. |
| determinism | same seed = identical output and the same behaviour on every camera; behaviour and model noise use independent random streams. |

## Geometry and behaviour rules
- Camera presets aim at the nominal eye point (`look_at_offset`), so camera-frame head pose and gaze read 0/0 for
  the nominal pose looking into the lens. Angles are axis based.
- Detection `occlusion` counts external occluders only (wheel, hands, held objects). Turning away sets
  `landmarks_visible` and `self_occluded_fraction`.
- Saccades interpolate direction with a peak speed cap; eye-in-head angles are clamped to `eye_in_head_limit_deg`.
- Blink / closure / yawn `rate_per_min` are true event rates. Frame k is the instant k / fps; a frame belongs to a
  segment when start <= k / fps < end; a run has ceil(duration * fps) frames.

## Output contract (`pipeline` format)
One JSON object per frame, field names in one place (`dms_model_simulator/schema.py`):
```
{ "frame_id": int, "timestamp_ms": int, "camera_id": str,
  "od": [ {"label": str, "bbox": [x1,y1,x2,y2], "score": float} ],
  "head_pose": {"yaw": deg, "pitch": deg, "roll": deg, "score": float} | null,
  "gaze": {"face": DIRECTION, "eye": DIRECTION} | null,
  "eye_openness": {"left": 0..1 | null, "right": 0..1 | null} | null,
  "mouth_openness": 0..1 | null }

DIRECTION = {"yaw": deg, "pitch": deg, "vector": [x,y,z], "valid": bool, "score": float,
             "yaw_vehicle": deg, "pitch_vehicle": deg, "zone": str}
```
Dropped frame: `od: []`, other fields null. No face: head pose / gaze / openness null. With a face, `gaze.face` is
always valid; when the eyes are closed or not visible (or the gaze points too far from the camera) `gaze.eye` has
`valid: false`, null angles / vector / zone and score 0 (a full-JSONL record stored before 0.2.0, without
`face_dir`, converts with `gaze.face: null`). Openness only for what the model detected in that frame.
`gaze.face` is the head's forward axis from the eye midpoint (camera yaw / pitch are direction angles in the camera
face frame, vehicle angles and zone as for the eye gaze). `head_pose` and `gaze.face` are two estimates of the same
head orientation with independent noise (two models): in the truth their camera yaw / pitch are equal, in the
model output they differ by that noise. `head_pose` also has roll; `gaze.face` also has the vector, vehicle angles
and zone. Latency (`pipeline.latency_frames`) delays the whole output by N frames; with latency the full-JSONL `pred` carries
`source_frame`.

Format change (0.2.0): `gaze` in the pipeline and truth JSONL became `{face, eye}`; before it was the flat eye gaze,
now `gaze.eye`.

Truth JSONL (`*.truth.jsonl`), same `frame_id`: exact `od`, head pose, gaze as `{face, eye}` (each: camera and
vehicle angles and vectors as `vector` / `vector_vehicle`, zone; `eye` also has the scripted target; no `valid` /
`score`, both always present), openness, blink / closure, hand / phone / cigarette / seatbelt / held states, `phone_visible`, and `events`:
`eyes_off_road` (scripted target off the road, or the actual gaze outside the yaw / pitch box of the on-road
targets widened by `zone_classifier.tolerance_deg`), `off_road_s` (current continuous off-road time), `eyes_closed` (mean scripted openness below
`closed_openness_threshold`), `closed_s`, `phone_use` (scripted, also when the phone is out of view),
`phone_use_visible` (phone box in the image), `yawning`, `smoking`, `seatbelt_off`; and `observable`
(`emulator.observable`): `face` / `head_pose` (face box in the truth), `head_away` (head pose there and the head's
facing direction outside the road box), `gaze` (the gaze model gives an eye gaze direction: `gaze.eye.valid`;
`gaze.face` follows the `face` / `head_pose` channel), `eyes`, `mouth`, `hand`, `phone`,
`cigarette` (that box in the truth; eyes and mouth need the face), `seatbelt_region` (the strap's place is in the
image, worn or not). With `perfect_detection: true` (profile `exact`) the model output has a quantity exactly
when its channel is observable. The full JSONL `truth` also
has `held_bbox`: the image box of an object at the lips (`held`; no OD class of its own), the anchor of
`anchor: held` context rules.

## Error flags (full JSONL `pred`)
`dropped` (whole frame without output), `latency_fill` (initial frames of a latency queue), `source_frame`,
per detection `p_detect`, `jump` (box moved elsewhere this frame), `false_positive` (+ `context` for a
context rule), and `class_flip` (`["eye"]` / `["mouth"]` on frames whose open / closed class is reported wrong).
A missed box is a class present in the truth detections and absent from a non-dropped `pred`.

## Random streams
- `Simulation`: `SeedSequence(seed)` spawns a behaviour and an emulator stream; `emulator_seed` replaces the
  emulator stream, so the model noise can vary with the behaviour fixed.
- `Emulator` splits its stream per error source (drops, false positives, each class's detection draws, each AR(1)
  noise process, flips, jumps), so changing one error rate does not change the others' draws. AR(1) noise is
  stationary from frame 0 and spans gaps with correlation rho**gap.
- `scenario_gen`: the target chain and each episode kind use their own stream; kinds with `target_bias`
  re-weight target transitions.

## Replay API
`replay.iter_pipeline(path, realtime=False)` yields pipeline records in order (`realtime=True` sleeps to the
timestamps); `replay.iter_dms_bin(path, layout, realtime=False)` does the same for dms.bin;
`replay.iter_pairs(pipeline, truth)` yields aligned (pred, truth). Records with a null `timestamp_ms` are yielded
without pacing.

## dms.bin
Packed binary log: optional file header (magic, version), then per frame a header followed by `num_objects`
object records. Everything comes from the layout YAML (`configs/binlog.yaml`): endianness, field order and types
(`u8..u64`, `i8..i64`, `f32`, `f64`, `count: N` arrays, `pad`, nested `record`s), field roles (header: timestamp,
frame_index, width, height, num_objects; object: class_id, score, box fields), `bbox_format` (xywh / ltwh / xyxy),
`conf_scale`, `timestamp_unit`, `class_ids` / `class_names`, optional `class_map` (with open / closed rules) and
per-class `conf_threshold`. Fields without a role the writer knows are written as their default (0) and read back
raw. Every frame writes a record; a dropped frame is a zero-object record.

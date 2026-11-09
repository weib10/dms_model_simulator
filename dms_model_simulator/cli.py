"""dms-sim command line."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from .camera import Camera
from .config import load_configs, load_scenario
from .metrics import markdown_table
from .simulate import FORMATS, Simulation
from .world import Cabin


def _cameras(cfgs, names):
    cabin = Cabin(cfgs["cabin"])
    allc = list(cfgs["cameras"]["cameras"].keys())
    names = allc if names in (None, ["all"]) else names
    return cabin, [Camera.from_config(n, cfgs["cameras"], cabin.reference_head_center) for n in names]


def cmd_run(a):
    cfgs = load_configs()
    cabin, cams = _cameras(cfgs, a.camera)
    rows = {}
    for cam in cams:
        sim = Simulation(a.scenario, cam.name, seed=a.seed, cfgs=load_configs(), emulator_seed=a.emulator_seed, fps=a.fps,
                         variant=a.variant, emulator_overrides=_profile_overrides(a))
        sim.run()
        paths = sim.write(a.out, fmt=a.format, include_face=a.include_face, dms_layout=a.layout, labels=a.labels)
        m = sim.summarize()
        rows[cam.name] = m
        with open(Path(a.out) / f"{sim.scenario['name']}__{cam.name}__metrics.json", "w") as fh:
            json.dump(m, fh, indent=2)
        for p in paths:
            print(f"wrote {p}  ({len(sim.records)} frames)")
        st = getattr(sim, "dmsbin_stats", None)
        if st:
            sk = ", ".join(f"{k}={v}" for k, v in sorted(st["skipped"].items())) or "none"
            bl = ", ".join(f"{k}={v}" for k, v in sorted(st["below_threshold"].items())) or "none"
            print(f"dms.bin: {st['frames']} frames, {st['objects']} objects, skipped (class not in layout): {sk}; "
                  f"below conf threshold: {bl}")
    print(markdown_table(rows, RUN_COLS))


RUN_COLS = ["face_detection_rate", "mean_face_width_px", "mean_abs_head_yaw_rel_camera_deg", "mean_gaze_to_camera_deg",
            "mean_face_occlusion", "gaze_valid_rate_eyes_open", "gaze_mean_error_deg", "zone_accuracy", "head_yaw_mae_deg"]


def _save_video(viz, frames, path: Path, fps: float, mp4: bool):
    viz.save_gif(frames, path, fps)
    if not mp4:
        return
    mp = path.with_suffix(".mp4")
    if not viz.ffmpeg_available():
        print(f"note: ffmpeg not found on PATH, skipping {mp.name} (GIF written)")
    elif viz.save_mp4(frames, mp, fps):
        print(f"wrote {mp}")
    else:
        print(f"note: ffmpeg failed to encode {mp.name} (GIF written)")


def _render_grid(a, viz, cams):
    """One tile per camera, same scenario and seed, so every tile shows the same driver behaviour."""
    sims = []
    for cam in cams:
        sim = Simulation(a.scenario, cam.name, seed=a.seed, cfgs=load_configs(), fps=a.fps, variant=a.variant,
                         emulator_overrides=_profile_overrides(a)); sim.run()
        sims.append(sim)
    out = Path(a.out) / f"{sims[0].scenario['name']}__grid"; out.mkdir(parents=True, exist_ok=True)
    n = min(len(s.records) for s in sims)
    frames = []
    t0 = time.perf_counter()
    for i in range(0, n, a.every):
        tiles = [viz.draw_camera_view(s.records[i], s.camera, s.cabin, a.scale) for s in sims]
        img = viz.make_grid(tiles, [s.camera.name for s in sims], a.grid_cols)
        if a.png:
            img.save(out / f"frame_{i:05d}.png")
        frames.append(img)
    dt = time.perf_counter() - t0
    fps = sims[0].behavior.fps / a.every
    _save_video(viz, frames, out / "preview.gif", fps, a.mp4)
    frames[len(frames) // 3].save(out / "sample.png")
    print(f"wrote {out}/preview.gif ({len(frames)} frames @ {fps:.1f} fps, {len(cams)} cameras, "
          f"{frames[0].width}x{frames[0].height}) and sample.png  [rendered at {len(frames) / max(dt, 1e-9):.1f} frames/s]")


def cmd_render(a):
    from . import viz
    cfgs = load_configs()
    cabin, cams = _cameras(cfgs, a.camera)
    if len(cams) > 1:
        return _render_grid(a, viz, cams)
    cam = cams[0]
    _, all_cams = _cameras(cfgs, None)
    sim = Simulation(a.scenario, cam.name, seed=a.seed, cfgs=cfgs, fps=a.fps, variant=a.variant,
                     emulator_overrides=_profile_overrides(a))
    sim.run()
    cam, cabin = sim.camera, sim.cabin                  # with the scenario's / variant's driver and mount
    all_cams = [cam if c.name == cam.name else c for c in all_cams]
    out = Path(a.out) / f"{sim.scenario['name']}__{cam.name}"; out.mkdir(parents=True, exist_ok=True)
    frames = []
    t0 = time.perf_counter()
    for r in sim.records[::a.every]:
        img = viz.composite(r, cam, cabin, all_cams, a.scale) if a.cabin else viz.draw_camera_view(r, cam, cabin, a.scale)
        if a.png:
            img.save(out / f"frame_{r['frame']:05d}.png")
        frames.append(img)
    dt = time.perf_counter() - t0
    fps = sim.behavior.fps / a.every
    viz.save_gif(frames, out / "preview.gif", fps)
    frames[len(frames) // 3].save(out / "sample.png")
    print(f"wrote {out}/preview.gif ({len(frames)} frames @ {fps:.1f} fps) and sample.png"
          f"  [rendered at {len(frames) / max(dt, 1e-9):.1f} frames/s]")
    if a.mp4:
        if not viz.ffmpeg_available():
            print("note: ffmpeg not found on PATH, skipping preview.mp4 (GIF written)")
        elif viz.save_mp4(frames, out / "preview.mp4", fps):
            print(f"wrote {out}/preview.mp4")
        else:
            print("note: ffmpeg failed to encode preview.mp4 (GIF written)")


def _write_jsonl(path: Path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


def cmd_demo(a):
    """Scenario on each camera -> all formats, behaviour card, timeline, GIFs, report.md."""
    from . import viz
    from .behavior_card import write_card
    cfgs = load_configs()
    _, cams = _cameras(cfgs, a.cameras)
    out = Path(a.out); data = out / "data"; data.mkdir(parents=True, exist_ok=True)
    scenario = load_scenario(a.scenario)
    name = scenario["name"] if a.variant is None else f"{scenario['name']}_v{a.variant}"   # as Simulation names it
    metrics, stems, gif_frames = {}, {}, {}
    fps = None
    t_start = time.perf_counter()
    for cam in cams:
        sim = Simulation(a.scenario, cam.name, seed=a.seed, cfgs=load_configs(), emulator_seed=a.emulator_seed,
                         fps=a.fps, variant=a.variant, emulator_overrides=_profile_overrides(a)); sim.run()
        sim.write(data, fmt="all", include_face=a.include_face, dms_layout=a.layout, labels=a.labels)
        stem = stems[cam.name] = f"{name}__{cam.name}"
        write_card(sim, data / f"{stem}__card")
        metrics[cam.name] = sim.summarize()
        fig = viz.plot_timeline(sim.records, out / f"{stem}__timeline.png",
                                title=f"{name} / {cam.name}: truth behaviour and injected model errors")
        viz.plt.close(fig)
        frames = [viz.draw_camera_view(r, sim.camera, sim.cabin, a.scale) for r in sim.records[::a.every]]
        fps = sim.behavior.fps / a.every
        viz.save_gif(frames, out / f"{stem}.gif", fps)
        frames[len(frames) // 3].save(out / f"{stem}__sample.png")
        gif_frames[cam.name] = frames
        print(f"[{cam.name}] {len(sim.records)} frames -> {data}/{stem}.*, {out}/{stem}.gif")
    grid = None
    if len(cams) > 1:
        n = min(len(f) for f in gif_frames.values())
        grid_frames = [viz.make_grid([gif_frames[c.name][i] for c in cams], [c.name for c in cams], a.grid_cols)
                       for i in range(n)]
        grid = f"{name}__grid.gif"
        viz.save_gif(grid_frames, out / grid, fps)
        del grid_frames
    gif_frames.clear()

    mt = markdown_table(metrics, RUN_COLS)
    L = [f"# DMS model-output simulator demo: `{name}`", "",
         str(scenario.get("description", "")).strip(), "",
         f"Seed {a.seed}" + (f" (emulator seed {a.emulator_seed})" if a.emulator_seed is not None else "")
         + ", cameras: " + ", ".join(f"`{c.name}`" for c in cams) + ". Per-frame outputs (full / pipeline / truth "
         "JSONL, dms.bin) and one behaviour card per camera (`<stem>__card/card.md`) are in `data/`.", "",
         "## Model output vs ground truth", "", mt, ""]
    if grid:
        L += [f"All cameras on the same frames: [{grid}]({grid})", "", f"![all cameras]({grid})", ""]
    for c in cams:
        stem = stems[c.name]
        L += [f"### {c.name}", "", f"Behaviour card: [data/{stem}__card/card.md](data/{stem}__card/card.md)", "",
              f"![timeline {c.name}]({stem}__timeline.png)", "",
              f"Preview: [{stem}.gif]({stem}.gif) (truth boxes green, model red, truth behaviour tags top right)", ""]
    report = out / "report.md"
    report.write_text("\n".join(L), encoding="utf-8")
    with open(out / "report.json", "w") as fh:
        json.dump({"scenario": name, "seed": a.seed, "emulator_seed": a.emulator_seed, "metrics": metrics}, fh, indent=2)
    print()
    print(mt); print()
    print(f"wrote {report}  ({time.perf_counter() - t_start:.1f} s)")


# what an export step raises for bad input: its own errors, an unknown camera (KeyError), a missing scenario file
# (OSError), an unusable file or config (ValueError)
_EXPORT_INPUT_ERRORS = (KeyError, OSError, ValueError)


def _export_fail(a, e):
    """Export error: a JSON object on stdout with --json (for agents), else a message on stderr; exit code 2."""
    from .export import ExportError
    msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)     # KeyError's str() adds quotes
    if not isinstance(e, ExportError):
        msg = f"{type(e).__name__}: {msg}"
    if getattr(a, "json", False):
        print(json.dumps({"ok": False, "error": msg}))
    else:
        print(f"dms-sim {a.cmd}: {msg}", file=sys.stderr)
    raise SystemExit(2)


def cmd_export(a):
    """Simulate and write the export (dms.bin / frame_yaml directory) plus the pipeline / truth / full JSONL and
    the behaviour card."""
    from . import export as X
    from .behavior_card import write_card
    try:
        sim, st, truth = X.export_scenario(a.scenario, a.camera, a.format, a.out, a.seed, a.fps, a.config,
                                           image=a.image, variant=a.variant,
                                           emulator_overrides=_profile_overrides(a), full=True)
        card = write_card(sim, Path(a.out) / f"{sim.scenario['name']}__{sim.camera.name}__card",
                          row={"profile": a.profile})
    except (X.ExportError, *_EXPORT_INPUT_ERRORS) as e:
        _export_fail(a, e)
    info = {"ok": True, "input": str(st["path"]), "format": st["format"], "truth": str(truth), "card": str(card),
            "frames": st["frames"], "objects": st["objects"], "fps": sim.behavior.fps,
            "skipped": st["skipped"], "below_threshold": st["below_threshold"]}
    if st["format"] == "frame_yaml":
        info.update(labels_dir=str(st["labels_dir"]), image_dir=str(st["image_dir"]), images=st["images"])
    if a.json:
        print(json.dumps(info))
    else:
        print(f"wrote {info['input']} ({info['format']}, {info['frames']} frames @ {info['fps']} fps, "
              f"{info['objects']} objects), {info['truth']} and {info['card']}")


def cmd_cabin(a):
    from . import viz
    cfgs = load_configs()
    cabin, cams = _cameras(cfgs, None)
    fig = viz.plot_cabin(cabin, cams)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(a.out) / "cabin.png", dpi=110)
    print(f"wrote {a.out}/cabin.png")


def cmd_compare(a):
    cfgs = load_configs()
    _, cams = _cameras(cfgs, None)
    rows = {}
    for cam in cams:
        sim = Simulation(a.scenario, cam.name, seed=a.seed, cfgs=load_configs(), variant=a.variant); sim.run()
        rows[cam.name] = sim.summarize()
        name = sim.scenario["name"]                       # with the variant suffix
    cols = ["face_detection_rate", "mean_face_width_px", "mean_eye_width_px", "mean_abs_head_yaw_rel_camera_deg",
            "mean_gaze_to_camera_deg", "mean_face_occlusion", "gaze_valid_rate_eyes_open", "gaze_mean_error_deg",
            "gaze_p95_error_deg", "zone_accuracy", "head_yaw_mae_deg"]
    table = markdown_table(rows, cols)
    print(table)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / f"compare__{name}.md").write_text(table + "\n")
        with open(Path(a.out) / f"compare__{name}.json", "w") as fh:
            json.dump(rows, fh, indent=2)


def cmd_inspect_bin(a):
    from .binlog import DmsBinLayout, format_frame, read_dms_bin
    layout = DmsBinLayout.from_yaml(a.layout)
    n = n_obj = 0
    warn = None
    try:
        for fr in read_dms_bin(a.path, layout):
            if a.head is None or n < a.head:
                print(format_frame(fr, layout))
            n += 1; n_obj += len(fr["objects"])
    except BrokenPipeError:          # e.g. piped into head / grep -m
        sys.stdout = None
        return
    except ValueError as e:          # truncated / malformed file: keep what was read
        warn = str(e)
    try:
        print(f"-- {n} frames, {n_obj} objects (layout {layout.source}; header {layout.header.size} B, "
              f"object {layout.obj.size} B)")
        if warn:
            print(f"warning: {warn}")
    except BrokenPipeError:
        sys.stdout = None


def cmd_gen_scenario(a):
    from . import scenario_gen
    cfg = scenario_gen.load_gen_config(a.config)
    out = Path(a.out) if a.out else Path("scenarios") / f"random_{int(round(a.duration))}s_seed{a.seed}.yaml"
    sc = scenario_gen.generate(cfg, a.duration, a.seed, name=a.name or out.stem)
    p = scenario_gen.save_yaml(sc, out)
    print(f"wrote {p}  ({len(sc['segments'])} segments, {a.duration} s)")


BATCH_COLS = ["face_in_view_rate", "face_detection_rate", "mean_face_width_px", "mean_eye_width_px",
              "mean_face_occlusion", "mean_abs_head_yaw_rel_camera_deg", "gaze_valid_rate_eyes_open", "zone_accuracy"]


def cmd_batch(a):
    """Scenarios (given and / or generated) x cameras x driver / camera-mount variants -> outputs + summary table."""
    from . import scenario_gen
    from .variation import sample_variant, load_variation_config
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    scenarios = list(a.scenario or [])
    if a.gen:
        gcfg = scenario_gen.load_gen_config(a.gen_config)
        for k in range(a.gen):
            sd = a.seed + k
            p = out / "scenarios" / f"random_{int(round(a.duration))}s_seed{sd}.yaml"
            scenario_gen.save_yaml(scenario_gen.generate(gcfg, a.duration, sd, name=p.stem), p)
            scenarios.append(str(p))
    if not scenarios:
        raise SystemExit("batch: give --scenario and / or --gen N")
    vcfg = load_variation_config(a.variation)
    variants = ([None] if a.nominal else []) + [sample_variant(v, vcfg) for v in range(a.variant_start,
                                                                                          a.variant_start + a.variants)]
    if not variants:
        raise SystemExit("batch: nothing to run (--variants 0 with --no-nominal)")
    cams = list(load_configs()["cameras"]["cameras"]) if a.camera == ["all"] else a.camera
    rows, recs = {}, []
    for sc in scenarios:
        for var in variants:
            for cam in cams:
                sim = Simulation(sc, cam, seed=a.seed, cfgs=load_configs(), fps=a.fps, variant=var); sim.run()
                sim.write(data := out / "data", fmt=a.format, labels=a.labels, dms_layout=a.layout)
                m = sim.summarize()
                drv = sim.cfgs["cabin"]["driver"]
                info = {"scenario": sim.scenario["name"], "camera": cam,
                        "variant": var["variant"] if var else None,
                        "eye_point_offset": [round(float(x), 4) for x in drv["eye_point_offset"]],
                        "head_scale": sim.cabin.head_scale,
                        "mount_offset": sim.camera.mount_offset.round(4).tolist(),
                        "mount_rotation_deg": sim.camera.mount_rotation.round(3).tolist()}
                recs.append({**info, "metrics": m})
                rows[f"{sim.scenario['name']} / {cam}"] = m
                print(f"{sim.scenario['name']} / {cam}: {len(sim.records)} frames, face in view "
                      f"{m['face_in_view_rate']:.3f}, face detected {m['face_detection_rate']:.3f}")
    table = markdown_table(rows, BATCH_COLS)
    vt = ["| run | eye_point_offset (m) | head_scale | mount_offset (m) | mount_rotation_deg |", "|---|---|---|---|---|"]
    vt += [f"| {r['scenario']} / {r['camera']} | {r['eye_point_offset']} | {r['head_scale']:.3f} | {r['mount_offset']} "
           f"| {r['mount_rotation_deg']} |" for r in recs]
    md = (f"# Batch: {len(scenarios)} scenario(s) x {len(variants)} variant(s) x {len(cams)} camera(s)\n\n"
          f"seed {a.seed}; outputs in data/\n\n## Model vs truth\n\n{table}\n\n## Driver and mount per run\n\n"
          + "\n".join(vt) + "\n")
    (out / "batch_summary.md").write_text(md)
    with open(out / "batch_summary.json", "w") as fh:
        json.dump(recs, fh, indent=2, default=float)
    print(table)
    print(f"wrote {out}/batch_summary.md, batch_summary.json and {len(recs)} runs in {out}/data")


def parse_param_overrides(items: list[str]) -> dict:
    """["name=value", ...] -> {name: value}; values are parsed as YAML (numbers, lists, dicts, strings)."""
    import yaml
    out = {}
    for it in items or []:
        name, sep, val = it.partition("=")
        if not sep or not name.strip():
            raise SystemExit(f"--param {it!r}: expected NAME=VALUE")
        out[name.strip()] = yaml.safe_load(val)
    return out


def cmd_catalog(a):
    """Scenario catalog (catalog/, see catalog/README.md): list / show / validate / index / expand / run."""
    from . import catalog as C
    if a.action in ("list", "show", "validate", "index", "expand"):
        entries = C.load_entries()
    if a.action == "list":
        sel = C.select(entries, a.select, a.status, a.tag)
        for e in sel:
            print(f"{e['id']:<34} {e.get('status', 'ready'):<8} {', '.join(e.get('tags') or []):<36} {e.get('title', '')}")
        print(f"{len(sel)} entries")
    elif a.action == "show":
        sel = C.select(entries, a.select)
        if not sel:
            raise SystemExit(f"no entry matches {a.select}")
        for e in sel:
            print(Path(e["_path"]).read_text(encoding="utf-8"))
    elif a.action == "validate":
        pr = C.load_profiles()
        bad = 0
        for e in C.select(entries, a.select):
            errs = C.validate(e, pr)
            bad += bool(errs)
            print(f"{e.get('id')}: " + ("ok" if not errs else "\n  " + "\n  ".join(errs)))
        if bad:
            raise SystemExit(f"{bad} invalid entr{'y' if bad == 1 else 'ies'}")
    elif a.action == "index":
        out = Path(a.out or C.CATALOG_DIR / "INDEX.md")
        out.write_text(C.index_markdown(entries), encoding="utf-8")
        print(f"wrote {out}")
    elif a.action == "expand":
        sel = C.select(entries, a.select, a.status, a.tag)
        overrides = parse_param_overrides(a.param)
        unused = sorted(k for k in overrides if not any(k in (e.get("params") or {}) for e in sel))
        if unused:
            raise SystemExit(f"--param {', '.join(unused)}: no selected entry has this parameter")
        man = C.expand(sel, a.out, a.mode, a.n, a.seed, cameras=a.camera, variants=a.variants,
                       emulator_seeds=a.emulator_seeds, profile_names=a.profile, max_rows=a.max_rows,
                       levels=a.level, overrides=overrides)
        print(f"wrote {a.out}/manifest.json: {len(man['rows'])} rows from {len(man['entries'])} entries"
              + (f" (planned, skipped: {', '.join(man['skipped_planned'])})" if man["skipped_planned"] else ""))
    elif a.action == "run":
        done = C.run(a.run_dir, a.select, a.jobs)
        print(f"exported {len(done)} rows; paths in {a.run_dir}/rows.tsv, one card.md per row")


def _face_args(parser):
    """--include-face / --no-face: the optional face OD box, for the pipeline JSONL and dms.bin alike."""
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--include-face", dest="include_face", action="store_const", const=True, default=None,
                   help="emit the face box in pipeline JSONL and dms.bin")
    g.add_argument("--no-face", dest="include_face", action="store_const", const=False,
                   help="omit the face box from pipeline JSONL and dms.bin "
                        "(default: pipeline without face, dms.bin with face if the layout maps it)")


def _profile_arg(parser):
    parser.add_argument("--profile", default=None,
                        help="model-error profile of catalog/profiles.yaml, or a stack like jit2+jmp2+mis1 (later parts "
                             "win); default: configs/emulator.yaml as is")


def _profile_overrides(a) -> dict | None:
    """Emulator overrides of --profile (None without it)."""
    if not getattr(a, "profile", None):
        return None
    from . import catalog as C
    try:
        return C.resolve_profile(a.profile, C.load_profiles())
    except C.CatalogError as e:
        raise SystemExit(f"dms-sim: --profile: {e}")


def _emulator_seed_arg(parser):
    parser.add_argument("--emulator-seed", type=int, default=None,
                        help="seed of the model-output noise only (default: derived from --seed); same --seed + "
                             "different --emulator-seed = same driver behaviour, different model errors")


def _non_negative_int(s: str) -> int:
    v = int(s)
    if v < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {v}")
    return v


def _variant_arg(parser):
    parser.add_argument("--variant", type=_non_negative_int, default=None, metavar="N",
                        help="driver / camera-mount variant N drawn from configs/variation.yaml (seat position, sitting "
                             "height, head size, mount position and aim error); default: nominal. Output names get _vN")


def _fps_arg(parser):
    parser.add_argument("--fps", type=float, default=None,
                        help="frame rate, overrides configs/behavior.yaml fps (default 30) and the scenario")


def _label_args(parser):
    from .schema import LABEL_MODES
    parser.add_argument("--labels", choices=LABEL_MODES, default="sim",
                        help="pipeline JSONL OD labels: sim = simulator classes; mapped = the consumer classes of the "
                             "--layout class_map (per-class conf thresholds; face included unless --no-face). dms.bin "
                             "always follows its layout")


def main(argv=None):
    p = argparse.ArgumentParser(prog="dms-sim")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="simulate a scenario and write per-frame JSONL (truth + model output)")
    r.add_argument("--scenario", required=True, help="scenario name or path to a YAML file"); r.add_argument("--camera", nargs="+", default=["steering_column"])
    r.add_argument("--out", default="out"); r.add_argument("--seed", type=int, default=0)
    _emulator_seed_arg(r)
    _profile_arg(r)
    r.add_argument("--format", choices=FORMATS, default="both",
                   help="full = nested truth+pred JSONL; pipeline = flat model-output JSONL + truth JSONL; both; "
                        "dmsbin = binary log; all = full + pipeline + dmsbin")
    r.add_argument("--layout", default=None, help="dms.bin layout YAML (default configs/binlog.yaml)")
    _face_args(r)
    _label_args(r)
    _fps_arg(r)
    _variant_arg(r)
    r.set_defaults(fn=cmd_run)
    v = sub.add_parser("render", help="render the camera view (and cabin layout) as PNG frames + GIF")
    v.add_argument("--scenario", required=True)
    v.add_argument("--camera", nargs="+", default=["steering_column"],
                   help="one camera (composite view), or several names / 'all' for a grid with one tile per camera")
    v.add_argument("--grid-cols", type=int, default=None, help="grid columns (default ceil(sqrt(n)))")
    v.add_argument("--out", default="out"); v.add_argument("--seed", type=int, default=0)
    v.add_argument("--every", type=int, default=3, help="render every Nth frame"); v.add_argument("--scale", type=float, default=0.5)
    v.add_argument("--png", action="store_true", help="also save individual PNG frames")
    v.add_argument("--mp4", action="store_true", help="also write preview.mp4 (needs ffmpeg on PATH)")
    _fps_arg(v)
    _variant_arg(v)
    _profile_arg(v)
    v.add_argument("--no-cabin", dest="cabin", action="store_false", help="camera view only"); v.set_defaults(fn=cmd_render)
    d = sub.add_parser("demo", help="scenario on each camera -> all formats, behaviour cards, timelines, GIFs, report.md")
    d.add_argument("--scenario", required=True, help="scenario name or path to a YAML file")
    d.add_argument("--cameras", nargs="+", default=["all"], help="camera names or 'all'")
    d.add_argument("--out", default="out/demo"); d.add_argument("--seed", type=int, default=0)
    _emulator_seed_arg(d)
    _profile_arg(d)
    d.add_argument("--layout", default=None, help="dms.bin layout YAML (default configs/binlog.yaml)")
    _face_args(d)
    _label_args(d)
    _fps_arg(d)
    _variant_arg(d)
    d.add_argument("--every", type=int, default=3, help="GIF: render every Nth frame")
    d.add_argument("--scale", type=float, default=0.4, help="GIF: camera view scale")
    d.add_argument("--grid-cols", type=int, default=None, help="columns of the all-camera grid GIF"); d.set_defaults(fn=cmd_demo)
    c = sub.add_parser("cabin", help="plot the cabin with all camera presets"); c.add_argument("--out", default="out"); c.set_defaults(fn=cmd_cabin)
    m = sub.add_parser("compare", help="run one scenario over all cameras and print a metrics table")
    m.add_argument("--scenario", required=True)
    m.add_argument("--out", default=None, help="also write compare__<scenario>.md / .json here (default: print only)"); m.add_argument("--seed", type=int, default=0)
    _variant_arg(m); m.set_defaults(fn=cmd_compare)
    ib = sub.add_parser("inspect-bin", help="print a dms.bin file as text")
    ib.add_argument("path"); ib.add_argument("--layout", default=None, help="layout YAML (default configs/binlog.yaml)")
    ib.add_argument("--head", type=int, default=None, help="print only the first N frames"); ib.set_defaults(fn=cmd_inspect_bin)
    g = sub.add_parser("gen-scenario", help="generate a random scenario (Markov gaze targets + phone / drowsy / smoking / seatbelt-off episodes)")
    g.add_argument("--duration", type=float, default=120.0); g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default=None, help="output YAML (default scenarios/random_<dur>s_seed<seed>.yaml)")
    g.add_argument("--name", default=None, help="scenario name (default: output file stem)")
    g.add_argument("--config", default=None, help="generator config (default configs/scenario_gen.yaml)")
    g.set_defaults(fn=cmd_gen_scenario)
    bt = sub.add_parser("batch", help="scenarios (given and / or generated) x cameras x driver / camera-mount variants "
                                      "-> outputs + batch_summary.md")
    bt.add_argument("--scenario", nargs="*", default=[], help="scenario names or YAML paths")
    bt.add_argument("--gen", type=int, default=0, metavar="N",
                    help="also generate N random scenarios (gen-scenario seeds --seed .. --seed+N-1) into OUT/scenarios")
    bt.add_argument("--duration", type=float, default=120.0, help="duration of generated scenarios (s)")
    bt.add_argument("--gen-config", default=None, help="generator config (default configs/scenario_gen.yaml)")
    bt.add_argument("--camera", nargs="+", default=["all"], help="camera names or 'all'")
    bt.add_argument("--variants", type=int, default=5, metavar="N", help="driver / camera-mount variants per run (default 5)")
    bt.add_argument("--variant-start", type=int, default=0, metavar="K", help="first variant number (variants K .. K+N-1)")
    bt.add_argument("--no-nominal", dest="nominal", action="store_false", help="skip the nominal driver and mount")
    bt.add_argument("--variation", default=None, help="variation config (default configs/variation.yaml)")
    bt.add_argument("--out", default="out/batch"); bt.add_argument("--seed", type=int, default=0)
    bt.add_argument("--format", choices=FORMATS, default="both", help="as for run")
    bt.add_argument("--layout", default=None, help="dms.bin layout YAML (default configs/binlog.yaml)")
    _label_args(bt)
    _fps_arg(bt)
    bt.set_defaults(fn=cmd_batch)
    ex = sub.add_parser("export", help="simulate and write the model output (dms.bin / per-frame YAML directory) + "
                                       "pipeline / truth / full JSONL + behaviour card")
    ex.add_argument("--scenario", required=True); ex.add_argument("--camera", required=True)
    ex.add_argument("--format", required=True, choices=("dms_bin", "frame_yaml"))
    ex.add_argument("--image", choices=("none", "render"), default="none",
                    help="frame_yaml: none (default) or render images/<stem>.png (clean camera view, full size)")
    ex.add_argument("--out", default="out/export"); ex.add_argument("--seed", type=int, default=0)
    ex.add_argument("--fps", type=float, default=None, help="frame rate (default: simulation.fps of the export config)")
    ex.add_argument("--config", default=None, help="export config (default configs/export.yaml)")
    ex.add_argument("--json", action="store_true", help="print the written paths as one JSON object")
    _variant_arg(ex)
    _profile_arg(ex)
    ex.set_defaults(fn=cmd_export)
    cg = sub.add_parser("catalog", help="scenario catalog: list / show / validate / index / expand / run "
                                        "(see catalog/README.md)")
    cs = cg.add_subparsers(dest="action", required=True)

    def _sel(q, nargs="*"):
        q.add_argument("select", nargs=nargs, default=None,
                       help="entry ids, categories (POS NEG BND CLR ROB NAT REG), families (DIS DRW YWN PHN SMK SBT ALL), "
                            "prefixes like BND-PHN, or all (default)")

    def _filters(q):
        q.add_argument("--status", choices=["ready", "planned"], default=None)
        q.add_argument("--tag", default=None)
    q = cs.add_parser("list", help="one line per entry"); _sel(q); _filters(q)
    q = cs.add_parser("show", help="print entries' YAML"); _sel(q, "+")
    q = cs.add_parser("validate", help="check entries (schema, templates, targets, frame-aligned durations)"); _sel(q)
    q = cs.add_parser("index", help="write catalog/INDEX.md"); q.add_argument("--out", default=None)
    q = cs.add_parser("expand", help="entries -> concrete scenarios + manifest.json"); _sel(q); _filters(q)
    q.add_argument("--out", required=True, help="run directory, e.g. runs/r1")
    q.add_argument("--mode", choices=["boundary", "grid", "random"], default="boundary",
                   help="boundary: the grid values (duration sweeps), one value of the others; grid: every "
                        "combination; random: --n draws")
    q.add_argument("--n", type=int, default=5, help="draws per entry for --mode random")
    q.add_argument("--seed", type=int, default=0, help="seed of the random parameter draws")
    q.add_argument("--camera", nargs="+", default=None, help="override the entries' cameras")
    q.add_argument("--variants", type=int, default=None, help="override: driver / mount variants 0..N-1 (+ nominal)")
    q.add_argument("--emulator-seeds", type=int, default=None, help="override: model-noise seeds 0..N-1")
    q.add_argument("--profile", nargs="+", default=None, help="override: error profiles (catalog/profiles.yaml)")
    q.add_argument("--max-rows", type=int, default=2000)
    q.add_argument("--level", nargs="+", default=None, choices=["small", "medium", "large"],
                   help="difficulty levels of entries with `levels` parameters (default: all three, one row each)")
    q.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                   help="fix a parameter in every selected entry that has it (value parsed as YAML); repeatable")
    q = cs.add_parser("run", help="export every manifest row (frame_yaml + JSONL + behaviour card)")
    q.add_argument("run_dir"); q.add_argument("select", nargs="*", default=None,
                                         help="only these rows: entry ids, categories, families or prefixes (as for list)")
    q.add_argument("--jobs", type=int, default=1, help="parallel processes")
    cg.set_defaults(fn=cmd_catalog)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

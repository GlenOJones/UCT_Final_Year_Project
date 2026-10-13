"""Compare camera calibration methods on the underwater AprilGrid data, on held-out recordings only.

Reads one Calibration.py run (its YAML and corners JSON) and, for every split that holds out
`held_out_recordings` recordings, fits each technique in config.json on the rest and scores it on the
held-out ones: reprojection residuals per corner, and lengths between stereo-triangulated corners.
Nothing is detected here and the input files are only read.

Normally run through reproduce.py, which first makes the Calibration.py runs the config lists under
"pipeline_runs":
    src/venv/bin/python src/experiments/reproduce.py compensation
On its own, once those runs exist (from the project root):
    src/venv/bin/python src/experiments/compensation/run_comparison.py
    src/venv/bin/python src/experiments/compensation/run_comparison.py --config my_config.json --workers 8

Writes to the config's output folder (default results/experiments/compensation/water_final/):
    summary.md, summary.csv, comparisons.csv    every metric per technique and camera, and paired differences
    params/<technique>.json                     fitted parameters per split, plus a fit on all recordings
    points.npz, bars.npz                        the held-out residual and scale-bar rows everything is computed from
    fig_heatmap, fig_radial, fig_distance, fig_scale, fig_summary  (.png and .pdf)
    run_info.json                               inputs, settings, versions, splits
"""
import os

# One thread per worker: the splits run in parallel, and OpenCV/BLAS threads inside each worker only
# fight over the same cores (one split took 60 s with them, 3.5 s without).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import sys  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)
from calibmethods import (covariance, data, evaluate, models, plots, splits, tables, target,  # noqa: E402
                          techniques as techniques_mod)
from calibmethods.evaluate import BAR_COLUMNS, POINT_COLUMNS  # noqa: E402

_RUN = None     # the loaded PipelineRun, shared with worker processes


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Compare calibration methods on held-out recordings.")
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"), help="settings (default config.json here)")
    ap.add_argument("--out", help="output folder (default: the config's 'output', relative to the project root)")
    ap.add_argument("--workers", type=int, default=min(28, os.cpu_count() or 1), help="parallel splits")
    ap.add_argument("--max-splits", type=int, help="only the first N splits: a quick trial run, not a result")
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw every figure from a finished run's saved results; fits nothing")
    return ap.parse_args(argv)


def project_path(path):
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def _init_worker(run):
    global _RUN
    _RUN = run
    cv2.setNumThreads(1)


def _work(job):
    split, techs, cfg = job
    return evaluate.run_split(_RUN, split, techs, cfg)


def run_all(run, split_list, techs, cfg, workers):
    jobs = [(s, techs, cfg) for s in split_list]
    if workers <= 1:
        _init_worker(run)
        return [_work(j) for j in jobs]
    # Workers start from a fresh server process ("forkserver"), not a fork of this one: reproduce.py calls this
    # after Calibration.py runs in the same process, and a fork taken while OpenCV's thread pool is alive
    # inherits its locks with no thread to release them (all 28 workers hung on a futex, 12 Oct). The run is
    # sent to each worker once, by the initializer.
    with multiprocessing.get_context("forkserver").Pool(workers, initializer=_init_worker, initargs=(run,)) as pool:
        return pool.map(_work, jobs)


def stack(results, key):
    """Rows of every split with the split index as column 0."""
    return np.concatenate([np.column_stack([np.full(len(r[key]), r["split"]), r[key]]) for r in results])


def write_params(out_dir, results, all_data, techs, run):
    os.makedirs(os.path.join(out_dir, "params"), exist_ok=True)
    for tech in techs:
        doc = {"technique": tech.key, "label": tech.label, "model": tech.model,
               "train_distance_mm": tech.train_distance_mm,
               "image_size": list(run.image_size), "opencv_version": cv2.__version__,
               "all_recordings": all_data["params"][tech.key],
               "splits": [{"split": r["split"], "train": r["train"], "test": r["test"], **r["params"][tech.key]}
                          for r in results]}
        with open(os.path.join(out_dir, "params", f"{tech.key}.json"), "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)


IN_AIR_MODELS = ("pinax", "air_direct")    # intrinsics from the in-air calibration, none fitted in water


def measure_target(cfg, run, techs, fitted_models, out_dir):
    """The WaterComparison tags triangulated with every technique fitted on all recordings (target.py).
    Writes target.json and target_tags.csv (make_figures draws fig_target from them)."""
    tcfg = cfg["target"]
    size = tcfg["tag_size_mm"]
    detections = target.detect(tcfg, tcfg["detection"], project_path)
    rows = {t.key: target.measure(fitted_models[t.key], detections, size) for t in techs}
    distances = [f.distance_mm for f in run.frames["left"] if np.isfinite(f.distance_mm)]
    calib_range = (float(min(distances)), float(max(distances)))
    doc = {"target": {k: v for k, v in tcfg.items() if not k.startswith("_")}, "calibration_range_mm": calib_range,
           "detected": {f"{lab:g}": {"left": sorted(L), "right": sorted(R)} for lab, L, R in detections},
           "summary": {t.key: target.summary(rows[t.key], size) for t in techs}, "tags": rows}
    with open(os.path.join(out_dir, "target.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    with open(os.path.join(out_dir, "target_tags.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["technique", "label_cm", "id", "distance_mm", "edge1_mm", "edge2_mm", "edge3_mm", "edge4_mm", "error_mm"])
        for t in techs:
            for r in rows[t.key]:
                w.writerow([t.key, f"{r['label_cm']:g}", r["id"], f"{r['distance_mm']:.1f}",
                            *(f"{e:.2f}" for e in r["edges_mm"]), f"{r['error_mm']:+.2f}"])


def parameter_covariance(cfg, run, techs, fitted_models, out_dir):
    """Covariance and correlation of each fitted technique's intrinsics (covariance.py), on the frames it was
    fitted on (all recordings). The in-air techniques fit nothing in water; their intrinsics come from the
    in-air calibration, so its covariance is computed once, on the in-air run's own frames."""
    doc = {}
    for t in techs:
        if t.model in IN_AIR_MODELS:
            continue
        for camera in data.CAMERAS:
            frames = t.training_frames(run.frames[camera], set(run.groups()))
            res = covariance.intrinsic_covariance(fitted_models[t.key][camera], frames)
            doc.setdefault(t.key, {})[camera] = res
    air_path = cfg["covariance"].get("air_calibration")
    if air_path:
        air_run = data.load(project_path(air_path))
        for camera in data.CAMERAS:
            K, D = air_run.pipeline_calibration[camera]
            res = covariance.intrinsic_covariance(models.BrownConrady(K, D), air_run.frames[camera])
            doc.setdefault("in_air_calibration", {})[camera] = res
    with open(os.path.join(out_dir, "covariance.json"), "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)


def make_figures(cfg, run, techs, points, bars, table, params_all, out_dir):
    """Every figure, from the held-out rows, the per-split table, the all-recordings parameters and (when the
    config asks for them) target.json and covariance.json. Fits nothing except the Pinax ray-mapping models,
    which are closed-form reads of the calibrations."""
    fitted = [i for i, t in enumerate(techs) if t.model not in IN_AIR_MODELS]
    cameras = cfg.get("report_cameras", list(data.CAMERAS))   # which cameras the figures show
    plots.show_cameras(cameras)
    figures = {"heatmap": plots.heatmaps(points, techs, params_all, cfg, run.image_size,
                                         os.path.join(out_dir, "fig_heatmap"), log=len(fitted) < len(techs))}
    if len(fitted) < len(techs):
        # Pinax's (and the unchanged in-air calibration's) errors are 10-40x the fitted models', so on a shared
        # linear scale the fitted ones are all one colour: they also get a figure of their own.
        figures["heatmap_fitted"] = plots.heatmaps(points, techs, params_all, cfg, run.image_size,
                                                   os.path.join(out_dir, "fig_heatmap_fitted"), subset=fitted)
    plots.radial(points, techs, cfg, os.path.join(out_dir, "fig_radial"))
    plots.distance(points, techs, cfg, os.path.join(out_dir, "fig_distance"))
    left_out = set(cfg.get("scale_plot_exclude", []))           # e.g. T1, whose bars are ~30 mm short
    plots.scale(bars, techs, cfg, os.path.join(out_dir, "fig_scale"),
                subset=[i for i, t in enumerate(techs) if t.key not in left_out])
    plots.summary(table, techs, [(cameras, "rms", "held-out RMS (px)"),
                                 (cameras, "outer_to_centre", "outer / centre"),
                                 (cameras, "far_to_near_centre", "far / near, centre zone"),
                                 (("stereo",), "length_rms_mm", "length RMS (mm)")],
                  os.path.join(out_dir, "fig_summary"))

    pinax = [t for t in techs if t.model == "pinax"]
    if pinax:
        # Where Pinax and the in-water Brown-Conrady (all recordings) send the same ray: why Pinax fails or not.
        reference, other, observed = {}, {}, {}
        ref_tech = next(t for t in techs if t.model == "brown_conrady" and t.train_distance_mm is None)
        for camera in cameras:
            reference[camera] = ref_tech.new_model().fit(run.frames[camera], run.image_size, camera)
            other[camera] = pinax[0].new_model().fit(run.frames[camera], run.image_size, camera)
            observed[camera] = np.concatenate([f.img for f in run.frames[camera]])
        plots.ray_mapping(reference, other, run.image_size, os.path.join(out_dir, "fig_pinax_mapping"),
                          ref_tech.label, "Pinax", observed)

    if cfg.get("quiver_cells"):
        # One figure for every technique: the in-air ones (errors of pixels) and those fitted in water (tenths of
        # a pixel) each get the arrow gain the config sets for them, so neither saturates nor vanishes.
        g = cfg.get("quiver_gains", {})
        gains = ({i: g["in_air"] if t.model in IN_AIR_MODELS else g["fitted"] for i, t in enumerate(techs)}
                 if g else None)
        figures["quiver"] = plots.quiver(points, techs, cfg, run.image_size, params_all,
                                         os.path.join(out_dir, "fig_quiver"), gains=gains)
    if cfg.get("target"):
        doc = load_json(os.path.join(out_dir, "target.json"))
        size, calib_range = doc["target"]["tag_size_mm"], doc["calibration_range_mm"]
        left_out = set(cfg.get("target_plot_exclude", []))      # e.g. T1, whose tags are 6-14 mm short
        plots.target_errors(doc["tags"], techs, size, calib_range, os.path.join(out_dir, "fig_target"),
                            subset=[i for i, t in enumerate(techs) if t.key not in left_out])
        if len(fitted) < len(techs):
            plots.target_errors(doc["tags"], techs, size, calib_range, os.path.join(out_dir, "fig_target_fitted"),
                                subset=fitted)
        figures["target"] = {"calibration_range_mm": calib_range}
    if cfg.get("covariance"):
        # The figure shows the five tests' fits; the close-range checks are in covariance.json and the report.
        doc = load_json(os.path.join(out_dir, "covariance.json"))
        label = {t.key: t.label for t in techs}
        panels = [(label[t.key], camera, doc[t.key][camera]) for t in techs
                  if t.key in doc and t.train_distance_mm is None for camera in cameras]
        panels += [("In-air calibration (T1, Pinax)", camera, doc["in_air_calibration"][camera])
                   for camera in cameras if "in_air_calibration" in doc]
        plots.correlation(panels, os.path.join(out_dir, "fig_correlation"))
        figures["covariance"] = {"panels": [(a, b) for a, b, _ in panels]}
    return figures


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def saved_results(out_dir, techs):
    """The per-split results (points, bars, stereo baseline) and the all-recordings parameters as a finished run
    saved them, for --figures-only."""
    P, B = np.load(os.path.join(out_dir, "points.npz")), np.load(os.path.join(out_dir, "bars.npz"))
    if list(P["techniques"]) != [t.key for t in techs]:
        raise SystemExit("the saved run has different techniques from the config: rerun it")
    points, bars = P["rows"].astype(np.float64), B["rows"].astype(np.float64)
    params = {t.key: load_json(os.path.join(out_dir, "params", f"{t.key}.json")) for t in techs}
    results = []
    for k, split in enumerate(params[techs[0].key]["splits"]):
        s = split["split"]
        results.append({"split": s, "points": points[points[:, 0] == s][:, 1:], "bars": bars[bars[:, 0] == s][:, 1:],
                        "params": {t.key: params[t.key]["splits"][k] for t in techs}})
    return points, bars, results, {t.key: params[t.key]["all_recordings"] for t in techs}


def main(argv=None):
    args = parse_args(argv)
    with open(args.config, encoding="utf-8") as fh:
        cfg = json.load(fh)
    out_dir = project_path(args.out or cfg["output"])
    if os.path.commonpath([os.path.abspath(out_dir), os.path.join(PROJECT_ROOT, "data")]) == \
            os.path.join(PROJECT_ROOT, "data"):
        raise SystemExit("refusing to write under data/: raw data is read-only")
    os.makedirs(out_dir, exist_ok=True)

    run = data.load(project_path(cfg["calibration"]))
    techs = techniques_mod.from_config(cfg["techniques"], project_path)
    if args.figures_only:
        points, bars, results, params_all = saved_results(out_dir, techs)
        make_figures(cfg, run, techs, points, bars, tables.per_split(results, techs, cfg), params_all, out_dir)
        print(f"redrew the figures in {os.path.relpath(out_dir, PROJECT_ROOT)}/ from its saved results")
        return
    groups = run.groups()
    split_list = splits.leave_groups_out(groups, cfg["held_out_recordings"])
    if args.max_splits:
        split_list = split_list[:args.max_splits]
    print(f"{run.preset} ({run.channel}): {len(run.frames['left'])} left / {len(run.frames['right'])} right frames, "
          f"{len(run.stereo_pairs())} stereo pairs, {len(groups)} recordings, {len(split_list)} splits")

    results = run_all(run, split_list, techs, cfg, args.workers)
    all_data = evaluate.run_split(run, splits.Split(-1, tuple(groups), ()), techs, cfg, keep_models=True)
    notes = [n for r in results for n in r["notes"]]

    points, bars = stack(results, "points"), stack(results, "bars")
    np.savez_compressed(os.path.join(out_dir, "points.npz"), rows=points.astype(np.float32),
                        columns=np.array(("split",) + POINT_COLUMNS),
                        techniques=np.array([t.key for t in techs]), cameras=np.array(data.CAMERAS),
                        **{f"frames_{c}": np.array([f.name for f in run.frames[c]]) for c in data.CAMERAS})
    np.savez_compressed(os.path.join(out_dir, "bars.npz"), rows=bars, columns=np.array(("split",) + BAR_COLUMNS),
                        techniques=np.array([t.key for t in techs]),
                        pairs=np.array([lf.name for lf, _ in run.stereo_pairs()]))
    write_params(out_dir, results, all_data, techs, run)

    table = tables.per_split(results, techs, cfg)
    comparison_rows = tables.comparisons(table, techs)
    tables.write_csv(os.path.join(out_dir, "summary.csv"), tables.summary_rows(table, techs))
    tables.write_csv(os.path.join(out_dir, "comparisons.csv"), comparison_rows)

    fitted = [i for i, t in enumerate(techs) if t.model not in IN_AIR_MODELS]
    if cfg.get("target"):
        measure_target(cfg, run, techs, all_data["models"], out_dir)
    if cfg.get("covariance"):
        parameter_covariance(cfg, run, techs, all_data["models"], out_dir)
    figures = make_figures(cfg, run, techs, points, bars, table, all_data["params"], out_dir)

    info = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
            "script": os.path.relpath(os.path.abspath(__file__), PROJECT_ROOT),
            "opencv_version": cv2.__version__, "numpy_version": np.__version__,
            "calibration": os.path.relpath(run.yaml_path, PROJECT_ROOT),
            "corners": os.path.relpath(run.corners_path, PROJECT_ROOT),
            "preset": run.preset, "channel": run.channel, "groups": groups,
            "held_out": cfg["held_out_recordings"], "n_splits": len(split_list),
            "splits": [{"split": s.index, "test": s.test} for s in split_list],
            "frames": {c: len(run.frames[c]) for c in data.CAMERAS}, "stereo_pairs": len(run.stereo_pairs()),
            "zone_radii": cfg["zone_radii"], "bands": cfg["distance_bands_mm"], "figures": figures,
            "config": cfg, "notes": notes}
    with open(os.path.join(out_dir, "run_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=1)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(tables.markdown(table, comparison_rows, techs, all_data["params"], info))
    print(f"wrote {os.path.relpath(out_dir, PROJECT_ROOT)}/ (summary.md, summary.csv, comparisons.csv, figures)")


if __name__ == "__main__":
    main()

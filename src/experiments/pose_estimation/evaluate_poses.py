"""Score every markerless trajectory of the pose experiment against the tag poses of the same frames.

    src/venv/bin/python src/experiments/pose_estimation/evaluate_poses.py --config src/experiments/pose_estimation/config.json
"""
import argparse
import csv
import datetime
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "04_pose"))
import common  # noqa: E402
import pose_files  # noqa: E402
from run_poses import recording_paths  # noqa: E402
from stereo_rig import invert  # noqa: E402


def umeyama(source, target, with_scale):
    """(scale, R, t) with target ~= scale * R @ source + t, least squares."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    cs, ct = source - mu_s, target - mu_t
    U, S, Vt = np.linalg.svd(ct.T @ cs / len(source))
    flip = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ flip @ Vt
    scale = np.trace(np.diag(S) @ flip) / np.mean(np.sum(cs ** 2, 1)) if with_scale else 1.0
    return scale, R, mu_t - scale * R @ mu_s


def pose_points(T_world_cam, lever):
    """Camera centre and the points `lever` mm along its x, y, z axes, in the world frame: (4, 3)."""
    local = np.vstack([np.zeros(3), np.eye(3) * lever])
    return local @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]


def rotation_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def stats(values):
    v = np.asarray(values, float)
    if not len(v):
        return {"rmse": None, "median": None, "p95": None, "max": None}
    return {"rmse": float(np.sqrt(np.mean(v ** 2))), "median": float(np.median(v)),
            "p95": float(np.percentile(v, 95)), "max": float(v.max())}


def evaluate(reference, estimate, diagnostics, names, cfg):
    """Metrics of one trajectory."""
    lever, distance = cfg["evaluation"]["align_lever_mm"], cfg["evaluation"]["rpe_distance_mm"]
    posed = [n for n in names if n in estimate and not diagnostics.get(n, {}).get("lost", 0)]
    common_names = [n for n in names if n in posed and n in reference]
    result = {"frames": len(names), "posed": len(posed), "registration": len(posed) / len(names),
              "with_reference": len(common_names)}
    if len(common_names) < 3:
        return result, None
    ref = {n: invert(reference[n]) for n in common_names}   # T_board_cam
    est = {n: invert(estimate[n]) for n in common_names}    # T_world_cam
    source = np.vstack([pose_points(est[n], lever) for n in common_names])
    target = np.vstack([pose_points(ref[n], lever) for n in common_names])
    _, R, t = umeyama(source, target, with_scale=False)
    # Scale from the camera centres alone: the lever points are fixed lengths and would pull it towards 1.
    best_scale = umeyama(source[::4], target[::4], with_scale=True)[0]
    align = np.eye(4)
    align[:3, :3], align[:3, 3] = R, t
    aligned = {n: align @ est[n] for n in common_names}

    centres_ref = np.array([ref[n][:3, 3] for n in common_names])
    centres_est = np.array([aligned[n][:3, 3] for n in common_names])
    ate_t = np.linalg.norm(centres_est - centres_ref, axis=1)
    ate_r = np.array([rotation_deg(ref[n][:3, :3].T @ aligned[n][:3, :3]) for n in common_names])
    # Position along the track: reference centres projected on their main direction of travel.
    centred = centres_ref - centres_ref.mean(0)
    direction = np.linalg.svd(centred)[2][0]
    track = centred @ direction
    track -= track.min()

    rpe_t, rpe_r = [], []
    for i in range(len(common_names)):
        later = np.flatnonzero(np.abs(track - track[i]) >= distance)
        later = later[later > i]
        if not len(later):
            continue
        j = later[0]
        a, b = common_names[i], common_names[j]
        rel_ref = invert(ref[a]) @ ref[b]
        rel_est = invert(est[a]) @ est[b]
        error = invert(rel_ref) @ rel_est
        rpe_t.append(np.linalg.norm(error[:3, 3]) / np.linalg.norm(rel_ref[:3, 3]) * 100)
        rpe_r.append(rotation_deg(error[:3, :3]))
    span_ref = np.linalg.norm(centres_ref[-1] - centres_ref[0])
    span_est = np.linalg.norm(centres_est[-1] - centres_est[0])
    constraint = [d["translation_constraint"] for n, d in diagnostics.items()
                  if n in posed and "translation_constraint" in d]
    result.update({"ate_mm": stats(ate_t), "ate_deg": stats(ate_r), "rpe_percent": stats(rpe_t),
                   "rpe_deg": stats(rpe_r), "rpe_distance_mm": distance, "best_fit_scale": float(best_scale),
                   "travel_ratio": float(span_est / span_ref) if span_ref > 0 else None,
                   "travel_mm": float(span_ref),
                   "translation_constraint_median": float(np.median(constraint)) if constraint else None})
    trace = {"names": common_names, "track": track, "ate_t": ate_t, "direction": direction,
             "xy_ref": centres_ref[:, :2], "xy_est": centres_est[:, :2],
             "xyz_ref": centres_ref, "xyz_est": centres_est}
    return result, trace


def plot(path, traces, recording, reference_label="tags"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    first = next(iter(traces.values()))
    ax[0].plot(first["xy_ref"][:, 0], first["xy_ref"][:, 1], "k-", lw=3, alpha=0.35, label=f"{reference_label} (reference)")
    for name, tr in traces.items():
        if name == "tags":
            continue
        ax[0].plot(tr["xy_est"][:, 0], tr["xy_est"][:, 1], lw=1, label=name)
        ax[1].plot(tr["track"], tr["ate_t"], lw=1, label=name)
    ax[0].set(title="Camera path, board frame (after one rigid fit)", xlabel="x (mm)", ylabel="y (mm)")
    ax[0].set_aspect("equal", adjustable="datalim")
    ax[1].set(title="Position error along the track", xlabel="reference travel (mm)", ylabel="ATE (mm)", yscale="log")
    ax[0].legend(fontsize=8)
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")
    fig.suptitle(f"{recording}: markerless poses vs tag poses")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fmt(value, spec):
    return "-" if value is None else format(value, spec)


def evaluate_recording(cfg, recording):
    p = recording_paths(cfg, recording)
    _, reference, _ = pose_files.read_trajectory(p["reference"])
    out_dir = os.path.join(p["out"], "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    results, traces, rows = {}, {}, []
    candidates = [("tags", p["reference"])] + [(e["name"], p["poses"](e["name"])) for e in cfg["estimators"]]
    for name, path in candidates:
        if not os.path.exists(path):
            print(f"  {name}: no poses at {common.rel(path)}, skipped")
            continue
        metadata, estimate, diagnostics = pose_files.read_trajectory(path)
        if name == "tags":   # the reference scored on itself, over the same frames the estimators get
            names = [n for n in diagnostics][::cfg["frame_step"]]
            seconds = None
        else:
            names = list(diagnostics)
            seconds = metadata.get("seconds")
        result, trace = evaluate(reference, estimate, diagnostics, names, cfg)
        result["seconds"] = seconds
        result["depth_seconds"] = metadata.get("depth_seconds") or None
        results[name] = result
        if trace is not None:
            traces[name] = trace
        a, r = result.get("ate_mm", {}), result.get("rpe_percent", {})
        rows.append({"estimator": name, "frames": result["frames"], "posed": result["posed"],
                     "registration_%": round(100 * result["registration"], 1),
                     "ate_rmse_mm": a.get("rmse"), "ate_median_mm": a.get("median"), "ate_max_mm": a.get("max"),
                     "ate_rmse_deg": result.get("ate_deg", {}).get("rmse"),
                     "rpe_median_%": r.get("median"), "rpe_p95_%": r.get("p95"),
                     "rpe_median_deg": result.get("rpe_deg", {}).get("median"),
                     "best_fit_scale": result.get("best_fit_scale"), "travel_ratio": result.get("travel_ratio"),
                     "translation_constraint": result.get("translation_constraint_median"),
                     "seconds": seconds, "depth_seconds": result["depth_seconds"]})

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    with open(os.path.join(out_dir, "run_info.json"), "w", encoding="utf-8") as fh:
        json.dump({"created": datetime.datetime.now().isoformat(timespec="seconds"),
                   "git_commit": common.git("rev-parse", "HEAD"),
                   "git_uncommitted_src_changes": common.uncommitted_src()}, fh, indent=1)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    distance = cfg["evaluation"]["rpe_distance_mm"]
    lines = [f"# {recording['name']}: markerless poses vs tag poses", "",
             f"Every {cfg['frame_step']}th frame pair of `{recording['frames']}`; tags masked: "
             f"{cfg['tag_masks']['enabled']}. ATE after one rigid fit; RPE over {distance} mm of travel.", "",
             "| estimator | posed | ATE RMSE mm | ATE max mm | ATE RMSE deg | RPE median % | RPE p95 % | RPE median deg "
             "| scale | travel ratio | transl. constraint | s (+depth s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['estimator']} | {row['posed']}/{row['frames']} | {fmt(row['ate_rmse_mm'], '.2f')} | "
                     f"{fmt(row['ate_max_mm'], '.2f')} | {fmt(row['ate_rmse_deg'], '.3f')} | {fmt(row['rpe_median_%'], '.2f')} | "
                     f"{fmt(row['rpe_p95_%'], '.2f')} | {fmt(row['rpe_median_deg'], '.3f')} | {fmt(row['best_fit_scale'], '.4f')} | "
                     f"{fmt(row['travel_ratio'], '.4f')} | {fmt(row['translation_constraint'], '.3f')} | "
                     f"{fmt(row['seconds'], '.0f')}{'' if not row['depth_seconds'] else ' (+' + format(row['depth_seconds'], '.0f') + ')'} |")
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    if traces:
        plot(os.path.join(out_dir, "trajectories.png"), traces, recording["name"])
    print("\n".join(lines))
    print(f"-> {common.rel(out_dir)}/")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    args = ap.parse_args(argv)
    cfg = common.load_config(args.config)
    return {r["name"]: evaluate_recording(cfg, r) for r in cfg["recordings"]}


if __name__ == "__main__":
    main()

"""Per-split metrics, their intervals, the paired model comparisons, and the summary tables."""
import csv
import itertools

import numpy as np

from . import data, metrics, stats

# (key, label, unit). Lower is better for all of them except the ratios, where 1 means "uniform".
CAMERA_METRICS = [
    ("rms", "held-out RMS", "px"),
    ("rms_centre", "centre RMS", "px"),
    ("rms_mid", "mid RMS", "px"),
    ("rms_outer", "outer RMS", "px"),
    ("mid_to_centre", "mid / centre", ""),
    ("outer_to_centre", "outer / centre", ""),
    ("rms_near", "near RMS", "px"),
    ("rms_far", "far RMS", "px"),
    ("far_to_near", "far / near", ""),
    ("rms_near_centre", "near RMS, centre zone", "px"),
    ("rms_far_centre", "far RMS, centre zone", "px"),
    ("far_to_near_centre", "far / near, centre zone", ""),
]
STEREO_METRICS = [
    ("length_bias_mm", "length bias", "mm"),
    ("length_rms_mm", "length RMS", "mm"),
    ("length_bias_far_mm", "far length bias", "mm"),
    ("length_rms_far_mm", "far length RMS", "mm"),
    ("baseline_mm", "baseline", "mm"),
]


def camera_metrics(p, cfg):
    """Metrics of one technique and camera on one split, from its point rows (evaluate.POINT_COLUMNS)."""
    du, dv, r, dist = p[:, 5], p[:, 6], p[:, 7], p[:, 9]
    z = metrics.zone_rms(du, dv, r, cfg["zone_radii"])
    near = (dist >= cfg["distance_bands_mm"]["near"][0]) & (dist < cfg["distance_bands_mm"]["near"][1])
    far = (dist >= cfg["distance_bands_mm"]["far"][0]) & (dist < cfg["distance_bands_mm"]["far"][1])
    out = {"rms": metrics.rms(du, dv),
           "rms_centre": z["centre"]["rms"], "rms_mid": z["mid"]["rms"], "rms_outer": z["outer"]["rms"],
           "mid_to_centre": z["mid_to_centre"], "outer_to_centre": z["outer_to_centre"],
           "rms_near": metrics.rms(du[near], dv[near]), "rms_far": metrics.rms(du[far], dv[far]),
           "n_points": len(p), "n_centre": z["centre"]["n"], "n_mid": z["mid"]["n"], "n_outer": z["outer"]["n"]}
    # Centre zone only, so near and far are compared at the same radii.
    centre = metrics.zone_of(r, cfg["zone_radii"]) == 0
    out["rms_near_centre"] = metrics.rms(du[near & centre], dv[near & centre])
    out["rms_far_centre"] = metrics.rms(du[far & centre], dv[far & centre])
    for suffix in ("", "_centre"):
        n = out[f"rms_near{suffix}"]
        out[f"far_to_near{suffix}"] = out[f"rms_far{suffix}"] / n if n > 0 else float("nan")
    return out


def stereo_metrics(b, baseline_mm, cfg):
    """Metrics of one technique on one split, from its bar rows (evaluate.BAR_COLUMNS)."""
    lo, hi = cfg["distance_bands_mm"]["far"]
    dist, bias, rms = b[:, 3], b[:, 4], b[:, 5]
    far = (dist >= lo) & (dist < hi)

    def rms_of(sel):
        return float(np.sqrt(np.mean(rms[sel] ** 2))) if sel.any() else float("nan")

    return {"length_bias_mm": float(bias.mean()) if len(b) else float("nan"),
            "length_rms_mm": rms_of(np.ones(len(b), bool)),
            "length_bias_far_mm": float(bias[far].mean()) if far.any() else float("nan"),
            "length_rms_far_mm": rms_of(far), "baseline_mm": baseline_mm, "n_pairs": len(b)}


def per_split(results, techniques, cfg):
    """{(technique key, camera or "stereo"): {metric: [value per split]}}."""
    table = {}
    for res in results:
        P, B = res["points"], res["bars"]
        for t, tech in enumerate(techniques):
            for c, camera in enumerate(data.CAMERAS):
                m = camera_metrics(P[(P[:, 0] == t) & (P[:, 1] == c)], cfg)
                for k, v in m.items():
                    table.setdefault((tech.key, camera), {}).setdefault(k, []).append(v)
            m = stereo_metrics(B[B[:, 0] == t], res["params"][tech.key]["stereo"]["baseline_mm"], cfg)
            for k, v in m.items():
                table.setdefault((tech.key, "stereo"), {}).setdefault(k, []).append(v)
    return table


def comparisons(table, techniques):
    """Paired differences between every two techniques, per camera and metric."""
    rows = []
    for a, b in itertools.combinations(techniques, 2):
        for camera, metric_list in [(c, CAMERA_METRICS) for c in data.CAMERAS] + [("stereo", STEREO_METRICS)]:
            for key, label, unit in metric_list:
                d = stats.paired_difference(table[(a.key, camera)][key], table[(b.key, camera)][key])
                rows.append({"a": a.key, "b": b.key, "camera": camera, "metric": key, "label": label, "unit": unit,
                             **{f"diff_{k}": d[k] for k in ("mean", "lo", "hi")}, "n_splits": d["n"],
                             "within_noise": d["within_noise"]})
    return rows


def fmt(s, digits=3):
    if not np.isfinite(s["mean"]):
        return "-"
    return f"{s['mean']:.{digits}f} [{s['lo']:.{digits}f}, {s['hi']:.{digits}f}]"


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summary_rows(table, techniques):
    rows = []
    for tech in techniques:
        for camera, metric_list in [(c, CAMERA_METRICS) for c in data.CAMERAS] + [("stereo", STEREO_METRICS)]:
            for key, label, unit in metric_list:
                s = stats.summarise(table[(tech.key, camera)][key])
                rows.append({"technique": tech.key, "camera": camera, "metric": key, "label": label, "unit": unit,
                             "mean": s["mean"], "lo": s["lo"], "hi": s["hi"], "n_splits": s["n"]})
    return rows


def markdown(table, comparison_rows, techniques, params_all, info):
    """The summary as Markdown: one table per camera, the stereo table, and the paired differences."""
    L = [f"# Calibration method comparison: {info['preset']} ({info['channel']})", "",
         f"Source: `{info['calibration']}`  ",
         f"{info['n_splits']} splits, each holding out {info['held_out']} of {len(info['groups'])} recordings "
         f"({', '.join(info['groups'])}).  ",
         "Values: mean [2.5, 97.5 percentile] over splits, held-out frames only. "
         f"Zones: centre r < {info['zone_radii'][0]}, mid < {info['zone_radii'][1]}, outer beyond "
         "(r normalised to the farthest image corner). "
         f"Near {info['bands']['near'][0]}-{info['bands']['near'][1]} mm, "
         f"far {info['bands']['far'][0]}-{info['bands']['far'][1]} mm (board distance).", ""]
    header = "| metric | " + " | ".join(t.label for t in techniques) + " |"
    rule = "|---|" + "---|" * len(techniques)
    L += ["## Parameters (fitted on all recordings)", "",
          "| | " + " | ".join(t.label for t in techniques) + " |", rule]
    for camera in data.CAMERAS:
        L.append(f"| {camera}: intrinsic parameters | "
                 + " | ".join(str(params_all[t.key][camera]["n_intrinsic_parameters"]) for t in techniques) + " |")
        L.append(f"| {camera}: training frames | "
                 + " | ".join(str(len(params_all[t.key][camera]["training_frames"])) for t in techniques) + " |")
        L.append(f"| {camera}: fit RMS px | "
                 + " | ".join(f"{params_all[t.key][camera]['fit_rms_px']:.3f}" for t in techniques) + " |")
    L.append("")
    for camera, metric_list in [(c, CAMERA_METRICS) for c in data.CAMERAS] + [("stereo", STEREO_METRICS)]:
        L += [f"## {camera.capitalize()}" + (" camera" if camera != "stereo" else " (triangulated scale bars)"),
              "", header, rule]
        for key, label, unit in metric_list:
            L.append(f"| {label}{f' ({unit})' if unit else ''} | "
                     + " | ".join(fmt(stats.summarise(table[(t.key, camera)][key])) for t in techniques) + " |")
        L.append("")
    L += ["## Paired differences (A - B)", "",
          "Within noise: the interval of the per-split difference contains 0.", "",
          "| A - B | camera | metric | difference | within noise |", "|---|---|---|---|---|"]
    for r in comparison_rows:
        s = {"mean": r["diff_mean"], "lo": r["diff_lo"], "hi": r["diff_hi"]}
        L.append(f"| {r['a']} - {r['b']} | {r['camera']} | {r['label']}{f' ({r['unit']})' if r['unit'] else ''} "
                 f"| {fmt(s)} | {'yes' if r['within_noise'] else '**no**'} |")
    if info["notes"]:
        L += ["", "## Notes", ""] + [f"- {n}" for n in info["notes"]]
    return "\n".join(L) + "\n"

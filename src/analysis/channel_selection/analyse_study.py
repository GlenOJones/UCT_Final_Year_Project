"""Compare the image channels of the channel selection study, from the runs run_study.py saved.

Three metrics, each answering one question:

  1. Detection: recall against distance, with the false-id rate (calibration frames)
     Does the detector find the board's tags in this channel, and out to what range? Recall is the
     number of DISTINCT board ids found in an image divided by the 70 on the board. It is plotted per
     board angle (scan) of recall_medium, against the frame's distance label (pairs.csv): pooling the
     angles would mix in how much of the board each angle shows, which is larger than any channel effect. The false-id and duplicate rates are the share of all
     detections that are ids not on the board, or a board id reported twice in one image: a channel
     that only gains tags by misreading them shows up here. Read from each run's YAML.

  2. Accuracy: triangulated tag size (held-out test target, Water only)
     Is the calibration made from this channel any better? Each channel's Water calibration
     triangulates the test target's tags, detected in that same channel, in stereo pairs never used
     for calibration. A tag's four edges are compared with the printed size (tag_size_mm in
     study.json), against the measured depth. Only tags every channel found in both cameras of a pair
     are compared, so all channels are judged on the same tags. Without a printed size the edge
     lengths are still reported: how much they drift with depth needs no ground truth.

  3. Corner precision: RMS reprojection error on the common set (calibration frames)
     Per camera, over only the images every channel's calibration used. A channel that rescues hard
     frames is otherwise penalised for their larger errors (see compare_calibrations.py).

Everything comes from run_study.py's outputs except the test target's detections, which use
Calibration.py's own load_image(), detector preset and corner refinement, so they match the pipeline.

Use (from the project root, after run_study.py):
    src/venv/bin/python src/analysis/channel_selection/analyse_study.py
    src/venv/bin/python src/analysis/channel_selection/analyse_study.py --study src/analysis/channel_selection/study_opencv_default.json
Writes results.json, fig_recall_by_angle.{pdf,png}, fig_recall_mean.{pdf,png}, fig_tag_size_vs_depth.{pdf,png} and
table_*.tex (booktabs, for \\input in the report) to the study's out_dir.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_study  # noqa: E402
from run_study import Calibration, PROJECT_ROOT  # noqa: E402

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "03_Reconstruction"))
from stereo_rig import load_rig  # noqa: E402

CAMERAS = Calibration.CAMERAS
N_TAGS = Calibration.N_TAGS

# Each channel keeps one colour, marker and line style in every figure. Colour is a natural label
# (r red, ...), and the marker and line style keep the channels apart in greyscale print and for
# colour-blind readers.
STYLE = {
    "gray": dict(color="#7a7a7a", marker="o", linestyle="-"),
    "y":    dict(color="#1a1a1a", marker="s", linestyle="--"),
    "r":    dict(color="#c0392b", marker="^", linestyle="-"),
    "g":    dict(color="#2e8b57", marker="D", linestyle="-."),
    "b":    dict(color="#2c6fbb", marker="v", linestyle=":"),
}
NAMES = {"gray": "Gray (computed)", "y": "Y (JPEG luma)", "r": "Red", "g": "Green", "b": "Blue"}


# ====== READING THE RUNS ======
def seq(node):
    """A YAML sequence node as a Python list (strings or numbers)."""
    return [node.at(i).string() if node.at(i).isString() else node.at(i).real() for i in range(node.size())]


def read_run(path):
    """What the metrics need from one calibration YAML."""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise SystemExit(f"missing {os.path.relpath(path, PROJECT_ROOT)} - run run_study.py first")
    run = {"cameras": {}, "detection": {},
           "stereo_rms": fs.getNode("stereo").getNode("rms_reprojection_error_px").real()}
    for camera in CAMERAS:
        node = fs.getNode(f"{camera}_camera")
        run["cameras"][camera] = {
            "rms": node.getNode("rms_reprojection_error_px").real(),
            "per_image": dict(zip(seq(node.getNode("images")), seq(node.getNode("per_image_error_px")))),
        }
        det = fs.getNode("detection").getNode(camera)
        run["detection"][camera] = {key: seq(det.getNode(key)) for key in
                                    ("images", "tags_found", "false_found", "duplicate_found")}
    fs.release()
    return run


def distance_labels(frames_dir):
    """{"subfolder/file.png": distance label} for every image, from each subfolder's pairs.csv."""
    labels = {}
    for sub in sorted(os.listdir(frames_dir)):
        path = os.path.join(frames_dir, sub, "pairs.csv")
        if not os.path.exists(path):
            continue
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                for camera in CAMERAS:
                    labels[f"{sub}/{row[f'{camera}_file']}"] = float(row["label"])
    return labels


# ====== METRIC 1: DETECTION ======
def angle_of(name):
    """Board angle code of an image, from its scan folder: "Water_C3_frames/c3_Left_20.png" -> "C3"."""
    match = re.fullmatch(r"[^_]+_([A-Za-z]+\d+)_frames", name.split("/")[0], re.IGNORECASE)
    if not match:
        raise SystemExit(f"{name}: scan folder is not named <medium>_<angle>_frames")
    return match.group(1).upper()


def angle_key(angle):
    """Sort C1, C3, ..., C8, L1, ...: by letter, then number."""
    letters = angle.rstrip("0123456789")
    return letters, int(angle[len(letters):])


def detection(study, runs):
    out = {}
    for medium, frames_dir in study["media"].items():
        labels = distance_labels(frames_dir)
        out[medium] = {}
        for channel in study["channels"]:
            by_angle = defaultdict(lambda: defaultdict(list))   # angle -> distance label -> [recall per camera]
            per_image = {}                                      # "camera/subfolder/file" -> recall
            by_label = defaultdict(list)                        # distance label -> [recall], all angles
            found = false = dup = usable = 0
            for camera in CAMERAS:
                d = runs[medium][channel]["detection"][camera]
                for name, n, f, u in zip(d["images"], d["tags_found"], d["false_found"], d["duplicate_found"]):
                    if name not in labels:
                        raise SystemExit(f"{name}: no distance label in its pairs.csv")
                    by_angle[angle_of(name)][labels[name]].append(n / N_TAGS)
                    per_image[f"{camera}/{name}"] = n / N_TAGS
                    by_label[labels[name]].append(n / N_TAGS)
                    found, false, dup = found + n, false + f, dup + u
                    usable += n >= study["calibration"]["min_tags"]
            total = found + false + dup
            out[medium][channel] = {
                "recall_mean": float(np.mean([r for a in by_angle.values() for v in a.values() for r in v])),
                "false_id_rate": false / total if total else float("nan"),
                "duplicate_rate": dup / total if total else float("nan"),
                "detections": int(total),
                "usable_images": int(usable),
                "per_image": per_image,
                "by_label": {f"{lab:g}": {"recall_mean": float(np.mean(v)), "n_images": len(v)}
                             for lab, v in sorted(by_label.items())},
                # Mean of the left and right image of each pair: one value per angle and distance.
                "by_angle": {angle: {f"{lab:g}": float(np.mean(v)) for lab, v in sorted(a.items())}
                             for angle, a in sorted(by_angle.items(), key=lambda t: angle_key(t[0]))},
            }
        # Paired against the first channel (the reference): the same image in both, so how much of the
        # board each angle and distance shows cancels, and only the channel's own effect is left.
        ref = out[medium][study["channels"][0]]["per_image"]
        for channel in study["channels"]:
            diff = [out[medium][channel]["per_image"][k] - ref[k] for k in ref]
            out[medium][channel].update(recall_delta_vs_ref=float(np.mean(diff)),
                                        recall_delta_se=float(np.std(diff, ddof=1) / np.sqrt(len(diff))))
    return out


# ====== METRIC 3: REPROJECTION ON THE COMMON SET ======
def rms(values):
    return float(np.sqrt(np.mean(np.square(values)))) if len(values) else float("nan")


def reprojection(study, runs):
    out = {}
    for medium in study["media"]:
        out[medium] = {}
        for camera in CAMERAS:
            per = {c: runs[medium][c]["cameras"][camera]["per_image"] for c in study["channels"]}
            common = sorted(set.intersection(*(set(p) for p in per.values())))
            out[medium][camera] = {"n_common": len(common), "channels": {
                c: {"rms_common": rms([per[c][n] for n in common]), "rms_all": runs[medium][c]["cameras"][camera]["rms"],
                    "n_images": len(per[c])} for c in study["channels"]}}
        out[medium]["stereo_rms_all"] = {c: runs[medium][c]["stereo_rms"] for c in study["channels"]}
    return out


# ====== METRIC 2: TRIANGULATED TAG SIZE ======
def target_detector(study):
    """The calibration preset's detector, with only the border width changed for the test target."""
    cal, target = study["calibration"], study["test_target"]
    if cal["detector"] != "aruco":
        raise SystemExit("the test-target detection is only implemented for the aruco detector")
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE   # refined by Calibration.refine_corners
    for key, value in {**Calibration.load_presets().settings(cal["preset"]),
                       "markerBorderBits": target["marker_border_bits"]}.items():
        setattr(params, key, value)
    return cv2.aruco.ArucoDetector(Calibration.dictionary, params)


def detect_target(detector, path, channel, ids):
    """{tag id: 4x2 refined corners} for the test target's tags in one image, seen through one channel."""
    _, image = Calibration.load_image(path, channel)
    corners, found, _ = detector.detectMarkers(image)
    if found is None:
        return {}
    corners = Calibration.refine_corners(image, corners)
    hits = defaultdict(list)
    for i, c in zip(found.ravel(), corners):
        hits[int(i)].append(c.reshape(4, 2))
    # A target id seen twice is ambiguous, so it is left out rather than guessed.
    return {i: cs[0] for i, cs in hits.items() if i in ids and len(cs) == 1}


def tag_size(study):
    target = study["test_target"]
    medium, frames, ids, size = target["medium"], target["frames"], set(target["ids"]), target["tag_size_mm"]
    with open(os.path.join(frames, "pairs.csv"), newline="") as fh:
        pairs = sorted(csv.DictReader(fh), key=lambda r: float(r["label"]))
    detector = target_detector(study)
    rigs = {c: load_rig(run_study.yaml_path(study, medium, c)) for c in study["channels"]}

    out = {"medium": medium, "tag_size_mm": size, "pairs": {}}
    edges_by_channel = defaultdict(list)          # channel -> [(depth, edge lengths)] over all pairs
    for row in pairs:
        seen = {c: {cam: detect_target(detector, os.path.join(frames, cam, row[f"{cam}_file"]), c, ids)
                    for cam in CAMERAS} for c in study["channels"]}
        common = sorted(set.intersection(*(set(s["left"]) & set(s["right"]) for s in seen.values())))
        entry = {"common_ids": common, "channels": {}}
        for c in study["channels"]:
            lengths, depths, reproj = [], [], []
            for t in common:
                X = rigs[c].triangulate(seen[c]["left"][t], seen[c]["right"][t])
                lengths += list(np.linalg.norm(X - np.roll(X, -1, axis=0), axis=1))
                depths.append(float(X[:, 2].mean()))
                reproj.append(rigs[c].reprojection_error(X, seen[c]["left"][t], seen[c]["right"][t]))
            e = {"found_left": len(seen[c]["left"]), "found_right": len(seen[c]["right"]),
                 "depth_mm": float(np.mean(depths)) if depths else float("nan"),
                 "edge_mean_mm": float(np.mean(lengths)) if lengths else float("nan"),
                 "edge_sd_mm": float(np.std(lengths)) if lengths else float("nan"),
                 "reprojection_px": rms(reproj)}
            if size and lengths:
                err = np.array(lengths) - size
                e.update(error_mean_mm=float(err.mean()), error_rms_mm=rms(err),
                         error_mean_pct=float(100 * err.mean() / size))
            entry["channels"][c] = e
            if lengths:
                edges_by_channel[c].append((e["depth_mm"], lengths))
        out["pairs"][row["label"]] = entry

    out["summary"] = {}
    for c in study["channels"]:
        means = [np.mean(l) for _, l in edges_by_channel[c]]
        allv = np.concatenate([l for _, l in edges_by_channel[c]]) if means else np.array([])
        s = {"edge_mean_mm": float(allv.mean()) if allv.size else float("nan"),
             # Spread of the per-pair mean edge across depths: scale drift with range, no ground truth needed.
             "edge_drift_mm": float(np.ptp(means)) if means else float("nan")}
        if size and allv.size:
            s.update(error_mean_mm=float(allv.mean() - size), error_rms_mm=rms(allv - size))
        out["summary"][c] = s
    return out


# ====== FIGURES ======
def fig_recall(study, det, rep, path):
    """One panel per board angle of recall_medium: recall against distance, one line per channel, and a
    table of the summary numbers per channel in the spare cells."""
    medium, channels = study["recall_medium"], study["channels"]
    names = study.get("angle_names", {})
    angles = list(det[medium][channels[0]]["by_angle"])
    ncols = 4
    nrows = -(-(len(angles) + 2) // ncols)        # at least two cells spare for the table
    fig = plt.figure(figsize=(3.0 * ncols, 2.4 * nrows + 0.4), constrained_layout=True)
    grid = fig.add_gridspec(nrows, ncols)
    first = None
    for k, angle in enumerate(angles):
        ax = fig.add_subplot(grid[k // ncols, k % ncols], sharex=first, sharey=first)
        first = first or ax
        for c in channels:
            by = det[medium][c]["by_angle"][angle]
            ax.plot([float(lab) for lab in by], list(by.values()), label=NAMES[c], markersize=4,
                    linewidth=1.3, **STYLE[c])
        ax.set_title(angle + (f" ({names[angle]})" if angle in names else ""), loc="left", fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3, linewidth=0.5)
    handles, labels = first.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=len(channels), fontsize=9, frameon=False)

    k = len(angles)
    table_ax = fig.add_subplot(grid[k // ncols, k % ncols:])
    table_ax.axis("off")
    ref = NAMES[channels[0]].split(" ")[0]
    header = ["", "mean\nrecall", f"Δ vs {ref}\n(pp, paired)", "usable\nimages", "false\nids (%)",
              "RMS px\nL / R"]
    rows = []
    for c in channels:
        d, r = det[medium][c], rep[medium]
        delta = "ref" if c == channels[0] else f"{100 * d['recall_delta_vs_ref'] + 0:+.1f} ± {100 * d['recall_delta_se']:.1f}".replace("-0.0 ", "0.0 ")
        rows.append([NAMES[c], f"{d['recall_mean']:.3f}", delta, f"{d['usable_images']}",
                     f"{100 * d['false_id_rate']:.2f}",
                     " / ".join(f"{r[cam]['channels'][c]['rms_common']:.2f}" for cam in CAMERAS)])
    table = table_ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center", edges="horizontal")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_height(cell.get_height() * 1.6)
        if col == 0:
            cell.set_text_props(ha="left")
    n_common = " / ".join(str(rep[medium][cam]["n_common"]) for cam in CAMERAS)
    table_ax.set_title(f"All angles. Δ: mean ± s.e. over the same images. Usable: ≥{study['calibration']['min_tags']} "
                       f"tags (L+R). RMS over the {n_common} images every channel used.",
                       fontsize=7, loc="left", wrap=True)

    fig.supxlabel(f"board distance ({study['distance_unit']})", fontsize=10)
    fig.supylabel(f"recall (distinct tags / {N_TAGS})", fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


def fig_recall_mean(study, det, path):
    """Mean recall over every image of recall_medium (all angles, both cameras) against distance, one
    line per channel; each channel's mean over all images is in its legend entry."""
    medium = study["recall_medium"]
    fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
    for c in study["channels"]:
        d = det[medium][c]
        by = d["by_label"]
        ax.plot([float(lab) for lab in by], [v["recall_mean"] for v in by.values()], markersize=5,
                linewidth=1.5, label=f"{NAMES[c]}: {d['recall_mean']:.3f}", **STYLE[c])
    n = sum(v["n_images"] for v in det[medium][study["channels"][0]]["by_label"].values())
    ax.set_xlabel(f"board distance ({study['distance_unit']})")
    ax.set_ylabel(f"mean recall (distinct tags / {N_TAGS})")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{medium}, preset {study['calibration']['preset']}: mean recall over all {n} images",
                 loc="left", fontsize=10)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(title="channel: mean over all images", title_fontsize=8, fontsize=8, frameon=False,
              loc="upper right")
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


def fig_tag_size(study, ts, path):
    size = ts["tag_size_mm"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    for c in study["channels"]:
        pts = sorted((e["depth_mm"], e["edge_mean_mm"], e["edge_sd_mm"])
                     for e in (p["channels"][c] for p in ts["pairs"].values()) if np.isfinite(e["edge_mean_mm"]))
        if pts:
            d, m, s = map(np.array, zip(*pts))
            y = m - size if size else m
            ax.errorbar(d / 1000, y, yerr=s, label=NAMES[c], markersize=5, linewidth=1.5, capsize=2,
                        elinewidth=0.8, **STYLE[c])
    if size:
        ax.axhline(0, color="#1a1a1a", linewidth=0.8)
        ax.set_ylabel(f"edge length error (mm), printed {size:g} mm")
    else:
        ax.set_ylabel("triangulated edge length (mm)")
    ax.set_xlabel("measured depth (m)")
    ax.set_title(f"Test target, {ts['medium']} calibrations", loc="left", fontsize=10)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=8, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", dpi=200)
    plt.close(fig)


# ====== TABLES ======
def write_table(path, header, rows, caption, label):
    """A booktabs table, ready for \\input in the report."""
    lines = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{l" + "r" * (len(header) - 1) + "}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule",
             *(" & ".join(r) + r" \\" for r in rows), r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("% Generated by src/analysis/channel_selection/analyse_study.py - do not edit by hand.\n")
        fh.write("\n".join(lines) + "\n")


def tables(study, det, rep, ts, out_dir):
    media = list(study["media"])
    write_table(os.path.join(out_dir, "table_detection.tex"),
                ["Channel"] + [f"{m} {h}" for m in media for h in ("recall", r"false id (\%)", r"dup.\ (\%)")],
                [[NAMES[c]] + [x for m in media for x in (f"{det[m][c]['recall_mean']:.3f}",
                                                         f"{100 * det[m][c]['false_id_rate']:.2f}",
                                                         f"{100 * det[m][c]['duplicate_rate']:.2f}")]
                 for c in study["channels"]],
                "Mean tag recall over all calibration images, and the share of detections that were false ids "
                "or duplicates, per image channel.", "tab:channel_detection")
    write_table(os.path.join(out_dir, "table_reprojection.tex"),
                ["Channel"] + [f"{m} {cam} ({rep[m][cam]['n_common']})" for m in media for cam in CAMERAS],
                [[NAMES[c]] + [f"{rep[m][cam]['channels'][c]['rms_common']:.3f}" for m in media for cam in CAMERAS]
                 for c in study["channels"]],
                "RMS reprojection error (px) per camera over only the images every channel's calibration used "
                "(number of images in brackets).", "tab:channel_reprojection")
    size = ts["tag_size_mm"]
    cols = ["Channel", "mean edge (mm)", "drift (mm)"] + (["error (mm)", "RMS error (mm)"] if size else [])
    write_table(os.path.join(out_dir, "table_tag_size.tex"), cols,
                [[NAMES[c], f"{s['edge_mean_mm']:.2f}", f"{s['edge_drift_mm']:.2f}"]
                 + ([f"{s['error_mean_mm']:+.2f}", f"{s['error_rms_mm']:.2f}"] if size else [])
                 for c, s in ts["summary"].items()],
                f"Test-target tag edges triangulated with each channel's {ts['medium']} calibration, over the tags "
                "every channel found. Drift is the range of the per-pair mean edge across depths.",
                "tab:channel_tag_size")


# ====== MAIN ======
def print_summary(study, det, rep, ts):
    for medium in study["media"]:
        print(f"\n{medium}: recall / false id % / duplicate %   |   common-set RMS px "
              + " ".join(f"{cam} ({rep[medium][cam]['n_common']})" for cam in CAMERAS))
        for c in study["channels"]:
            d = det[medium][c]
            print(f"  {c:5s} {d['recall_mean']:.3f} / {100 * d['false_id_rate']:5.2f} / {100 * d['duplicate_rate']:5.2f}"
                  "   |   " + "  ".join(f"{rep[medium][cam]['channels'][c]['rms_common']:.3f}" for cam in CAMERAS))
    print(f"\nTest target ({ts['medium']}), tags compared per pair: "
          + ", ".join(f"{lab}: {len(p['common_ids'])}" for lab, p in ts["pairs"].items()))
    for c, s in ts["summary"].items():
        extra = f"  error {s['error_mean_mm']:+.2f} mm (RMS {s['error_rms_mm']:.2f})" if "error_mean_mm" in s else ""
        print(f"  {c:5s} mean edge {s['edge_mean_mm']:.2f} mm, drift over depth {s['edge_drift_mm']:.2f} mm{extra}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    run_study.add_study_argument(ap)
    study = run_study.load_study(ap.parse_args().study)
    out_dir = study["out_dir"]
    runs = {m: {c: read_run(run_study.yaml_path(study, m, c)) for c in study["channels"]} for m in study["media"]}

    det, rep, ts = detection(study, runs), reprojection(study, runs), tag_size(study)
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump({"detection": det, "reprojection": rep, "tag_size": ts}, fh, indent=1)
    fig_recall(study, det, rep, os.path.join(out_dir, "fig_recall_by_angle"))
    fig_recall_mean(study, det, os.path.join(out_dir, "fig_recall_mean"))
    fig_tag_size(study, ts, os.path.join(out_dir, "fig_tag_size_vs_depth"))
    tables(study, det, rep, ts, out_dir)
    print_summary(study, det, rep, ts)
    print(f"\nwrote results.json, figures and tables to {os.path.relpath(out_dir, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

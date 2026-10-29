"""Accuracy check of detection setups on the underwater calibration and the WaterComparison target."""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402
from common import PROJECT_ROOT, Calibration as C  # noqa: E402
sys.path.insert(0, os.path.join(common.SRC_DIR, "03_Reconstruction"))
from stereo_rig import load_rig  # noqa: E402

CONFIG_FILE = os.path.join(HERE, "config.json")
INTRINSICS = ("fx_px", "fy_px", "cx_px", "cy_px", "k1", "k2", "p1", "p2", "k3")
# One colour per preset, one marker per channel, so a setup reads as its two halves.
PRESET_COLOURS = ("#9a9a9a", "#1F5673", "#c0392b", "#2e8b57")
CHANNEL_MARKERS = {"gray": "o", "merge": "P", "y": "s", "r": "^", "g": "D", "b": "v"}

# The experiment's settings, filled from config.json "accuracy" by configure() before anything runs.
FRAMES = TARGET = CAL_DIR = OUT = None   # calibration frames, target pairs, calibrations, output folder
SETUPS = []                              # (preset, channel) per setup
TAG_SIZE_MM = None                       # printed edge of the target tags' outer black square
TARGET_IDS = MIDDLE_ROW = set()
STYLE = {}


def configure(path):
    """Set the module settings above from the experiment config."""
    global FRAMES, TARGET, CAL_DIR, OUT, SETUPS, TAG_SIZE_MM, TARGET_IDS, MIDDLE_ROW, STYLE
    cfg = common.load_config(path)
    acc, out_dir = cfg["accuracy"], common.project_path(cfg["out_dir"])
    FRAMES = common.project_path(cfg["media"][acc["medium"]])
    TARGET = common.project_path(acc["target"])
    CAL_DIR = os.path.join(out_dir, "calibration", acc["medium"])
    OUT = os.path.join(out_dir, "accuracy")
    SETUPS = [tuple(s) for s in acc["setups"]]
    TAG_SIZE_MM = acc["tag_size_mm"]
    TARGET_IDS, MIDDLE_ROW = set(acc["target_ids"]), set(acc["middle_row"])
    presets = list(dict.fromkeys(p for p, _ in SETUPS))
    STYLE = {s: dict(color=PRESET_COLOURS[presets.index(s[0]) % len(PRESET_COLOURS)], marker=CHANNEL_MARKERS[s[1]])
             for s in SETUPS}


def label(setup):
    return f"{setup[0]} + {setup[1]}"


def yaml_path(setup):
    return os.path.join(CAL_DIR, C.run_name(FRAMES, "aruco", setup[0], setup[1]) + "_stereo.yaml")


# ====== 1-2. CALIBRATION METRICS ======
def read_calibration(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise SystemExit(f"missing {os.path.relpath(path, PROJECT_ROOT)}: run reproduce.py detector_sweep first")
    seq = lambda n: [n.at(i).string() if n.at(i).isString() else n.at(i).real() for i in range(n.size())]  # noqa: E731
    out = {"cameras": {}}
    for cam in C.CAMERAS:
        node = fs.getNode(f"{cam}_camera")
        K, D = node.getNode("camera_matrix").mat(), node.getNode("distortion_coefficients").mat().ravel()
        sig = node.getNode("uncertainty_1sigma")
        out["cameras"][cam] = {
            "per_image": dict(zip(seq(node.getNode("images")), seq(node.getNode("per_image_error_px")))),
            "rms": node.getNode("rms_reprojection_error_px").real(),
            "values": dict(zip(INTRINSICS, [K[0, 0], K[1, 1], K[0, 2], K[1, 2], *D[:5]])),
            "K": K.tolist(), "D": D.tolist(),
            "sigma": {k: sig.getNode(k).real() for k in INTRINSICS},
            "pose_rot_deg": float(np.median(seq(node.getNode("pose_uncertainty_1sigma").getNode("rotation_deg")))),
            "pose_trans_mm": float(np.median(seq(node.getNode("pose_uncertainty_1sigma").getNode("translation_mm")))),
        }
    st = fs.getNode("stereo")
    out["stereo"] = {"rms": st.getNode("rms_reprojection_error_px").real(), "pairs": int(st.getNode("pairs_used").real()),
                     "baseline_mm": st.getNode("baseline_mm").real()}
    return out


def error_stats(values):
    v = np.asarray(values)
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)), "max": float(v.max()),
            "rms": float(np.sqrt(np.mean(v ** 2)))}


def shared_tag_errors(cal):
    """Corner errors (px) of the tags every setup detected in the same image, each setup judged with its own intrinsics
    and a robust per-image pose, so every setup is scored on identical tags."""
    corners = {s: json.load(open(yaml_path(s).replace("_stereo.yaml", "_corners.json")))["cameras"] for s in SETUPS}
    holder = SimpleNamespace(board=C.build_board())
    sets = {}
    for s in SETUPS:
        for cam in C.CAMERAS:
            for n, d in corners[s][cam].items():
                for i in d["ids"]:
                    if 0 <= i < C.N_TAGS and d["ids"].count(i) == 1:
                        sets.setdefault(s, set()).add((cam, n, i))
    shared = set.intersection(*sets.values())
    images = sorted({(cam, n) for cam, n, _ in shared})
    scores = {}
    for s in SETUPS:
        out = []
        for cam, n in images:
            d = corners[s][cam][n]
            K = np.array(cal[s]["cameras"][cam]["K"]); D = np.array(cal[s]["cameras"][cam]["D"])
            ids = np.int32(d["ids"]).reshape(-1, 1)
            corner_list = tuple(np.float32(c).reshape(1, 4, 2) for c in d["corners"])
            errs = C.tag_errors(holder, corner_list, ids, K, D)
            kept = [i for i in errs if (cam, n, i) in shared]
            # per-corner distances need the pose again; tag_errors gives worst and mean-square per tag
            out += [np.sqrt(errs[i][1]) for i in kept]
        scores[s] = np.array(out)
    return shared, scores


# ====== 3. MEASUREMENT ON THE REFERENCE TARGET ======
def target_detector(preset):
    """The preset's detector with the printed target's 1-bit border instead of the AprilGrid's 2."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    for key, value in {**C.load_presets().settings(preset), "markerBorderBits": 1}.items():
        setattr(params, key, value)
    return cv2.aruco.ArucoDetector(C.dictionary, params)


def detect_target(detector, path, channel):
    """{id: 4x2 corners} of the target tags, found the way Calibration.py finds board tags for this channel."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if channel == "merge":
        gray = C.channel_image(image, "gray")
        results = []
        for c in C.MERGE_ORDER:                       # channels find tags; corners refined on gray
            corners, ids, rejected = detector.detectMarkers(C.channel_image(image, c))
            results.append((C.refine_corners(gray, corners), ids, rejected))
        corners, ids, _ = C.merge_detections(results)
    else:
        img = C.channel_image(image, channel)
        corners, ids, _ = detector.detectMarkers(img)
        corners = C.refine_corners(img, corners)
    if ids is None:
        return {}
    flat = [int(i) for i in ids.ravel()]
    return {i: c.reshape(4, 2) for i, c in zip(flat, corners) if i in TARGET_IDS and flat.count(i) == 1}


def measure(setup, pairs):
    """Per tag seen in both cameras: label, id, region, depth, the four edge lengths and reprojection error."""
    rig = load_rig(yaml_path(setup))
    det = target_detector(setup[0])
    rows = []
    for r in pairs:
        L = detect_target(det, os.path.join(TARGET, "left", r["left_file"]), setup[1])
        R = detect_target(det, os.path.join(TARGET, "right", r["right_file"]), setup[1])
        common = sorted(set(L) & set(R))
        if not common:
            continue
        u = {t: L[t][:, 0].mean() for t in common}
        split = np.median(list(u.values()))
        for t in common:
            X = rig.triangulate(L[t], R[t])
            edges = np.linalg.norm(X - np.roll(X, -1, axis=0), axis=1)
            region = "middle row" if t in MIDDLE_ROW else ("left column" if u[t] < split else "right column")
            rows.append({"label": r["label"], "id": t, "region": region, "depth_mm": float(X[:, 2].mean()),
                         "edges_mm": edges.tolist(), "reprojection_px": rig.reprojection_error(X, L[t], R[t])})
    return rows


def edge_stats(rows):
    e = np.concatenate([np.array(r["edges_mm"]) for r in rows]) - TAG_SIZE_MM if rows else np.array([])
    if not e.size:
        return {"n_tags": 0}
    return {"n_tags": len(rows), "n_edges": int(e.size), "bias_mm": float(e.mean()), "mae_mm": float(np.abs(e).mean()),
            "rms_mm": float(np.sqrt(np.mean(e ** 2))), "max_abs_mm": float(np.abs(e).max()),
            "bias_pct": float(100 * e.mean() / TAG_SIZE_MM)}


# ====== REPORT ======
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=CONFIG_FILE, help="experiment config (default config.json here)")
    configure(ap.parse_args(argv).config)
    os.makedirs(OUT, exist_ok=True)
    cal = {s: read_calibration(yaml_path(s)) for s in SETUPS}
    with open(os.path.join(TARGET, "pairs.csv"), newline="") as fh:
        pairs = sorted(csv.DictReader(fh), key=lambda r: float(r["label"]))
    meas = {s: measure(s, pairs) for s in SETUPS}

    L = [f"Accuracy check, Water: {', '.join(label(s) for s in SETUPS)}",
         "Generated by src/experiments/detector_sweep/accuracy_check.py", ""]
    res = {"setups": [label(s) for s in SETUPS], "calibration": {}, "measurement": {}}

    L += ["1. REPROJECTION ERROR (per-image RMS, px): mean / median / max over the images each run used,",
          "   and over the images all four runs used (common set)."]
    common = {cam: set.intersection(*(set(cal[s]["cameras"][cam]["per_image"]) for s in SETUPS)) for cam in C.CAMERAS}
    L.append(f"   common set: {len(common['left'])} left, {len(common['right'])} right images")
    L.append(f"   {'setup':30s}{'cam':>6s}{'images':>7s}{'mean':>7s}{'median':>8s}{'max':>8s}{'RMS':>7s}"
             f"{'common: mean':>14s}{'median':>8s}{'max':>8s}")
    for s in SETUPS:
        res["calibration"][label(s)] = {"stereo": cal[s]["stereo"], "cameras": {}}
        for cam in C.CAMERAS:
            c = cal[s]["cameras"][cam]
            a = error_stats(list(c["per_image"].values()))
            b = error_stats([c["per_image"][n] for n in common[cam]])
            res["calibration"][label(s)]["cameras"][cam] = {"all_images": a, "common_images": b, "rms": c["rms"],
                                                            "intrinsics": c["values"], "sigma_1": c["sigma"],
                                                            "pose_sigma_median": {"rot_deg": c["pose_rot_deg"], "trans_mm": c["pose_trans_mm"]}}
            L.append(f"   {label(s):30s}{cam:>6s}{a['n']:7d}{a['mean']:7.3f}{a['median']:8.3f}{a['max']:8.3f}{c['rms']:7.3f}"
                     f"{b['mean']:14.3f}{b['median']:8.3f}{b['max']:8.3f}")
    L += ["", f"   {'setup':30s}{'stereo RMS px':>14s}{'pairs':>7s}{'baseline mm':>13s}"]
    for s in SETUPS:
        st = cal[s]["stereo"]
        L.append(f"   {label(s):30s}{st['rms']:14.3f}{st['pairs']:7d}{st['baseline_mm']:13.2f}")

    L += ["", "1b. SHARED TAGS: every (image, tag) all four setups detected, posed robustly per image with each",
          "    setup's own calibration; distance of each corner from where the board model puts it (px)."]
    shared, scores = shared_tag_errors(cal)
    L.append(f"    {len(shared)} shared tags")
    L.append(f"   {'setup':30s}{'mean':>8s}{'median':>8s}{'p95':>8s}{'RMS':>8s}{'> 2 px':>9s}{'unposed':>9s}")
    res["shared_tags"] = {"n": len(shared)}
    for s in SETUPS:
        failed = int(np.sum(~np.isfinite(scores[s])))   # tags in images whose robust pose failed
        e = scores[s][np.isfinite(scores[s])]
        st = {"unposed_tags": failed, "mean": float(e.mean()), "median": float(np.median(e)), "p95": float(np.percentile(e, 95)),
              "rms": float(np.sqrt(np.mean(e ** 2))), "over_2px_pct": float(100 * np.mean(e > 2))}
        res["shared_tags"][label(s)] = st
        L.append(f"   {label(s):30s}{st['mean']:8.3f}{st['median']:8.3f}{st['p95']:8.3f}{st['rms']:8.3f}{st['over_2px_pct']:8.2f}%{failed:9d}")

    L += ["", "2. CAMERA MATRIX: value +- OpenCV 1-sigma; median board-pose 1-sigma"]
    for cam in C.CAMERAS:
        L.append(f"   {cam} camera" + "".join(f"{n:>20s}" for n in ("fx_px", "fy_px", "cx_px", "cy_px", "k1")) + f"{'pose rot/trans':>20s}")
        for s in SETUPS:
            c = cal[s]["cameras"][cam]
            vals = "".join((f"{c['values'][n]:9.2f} +- {c['sigma'][n]:5.2f}" if n.endswith("_px")
                            else f"{c['values'][n]:8.4f}+-{c['sigma'][n]:.4f}").rjust(20) for n in ("fx_px", "fy_px", "cx_px", "cy_px", "k1"))
            L.append(f"     {label(s):28s}{vals}{c['pose_rot_deg']:9.3f} deg {c['pose_trans_mm']:5.2f} mm")

    L += ["", f"3. MEASUREMENT ACCURACY on the WaterComparison target (tag edge {TAG_SIZE_MM:g} mm): edge length - "
          f"{TAG_SIZE_MM:g} mm", "   bias = mean error, MAE = mean |error|, RMS, max |error|; mm"]
    tags_per_pair = {s: {(r["label"], r["id"]) for r in meas[s]} for s in SETUPS}
    shared = set.intersection(*tags_per_pair.values())
    for scope, pick in (("all tags each setup found", lambda s: meas[s]),
                        (f"the {len(shared)} tags all four setups found", lambda s: [r for r in meas[s] if (r["label"], r["id"]) in shared])):
        L.append(f"   -- {scope}")
        L.append(f"   {'setup':30s}{'tags':>5s}{'bias':>8s}{'bias %':>8s}{'MAE':>7s}{'RMS':>7s}{'max':>7s}   by label (tags: bias)")
        for s in SETUPS:
            rows = pick(s)
            st = edge_stats(rows)
            by_lab = defaultdict(list)
            for r in rows:
                by_lab[r["label"]].append(r)
            per = "  ".join(f"{lab}({len(v)}): {edge_stats(v)['bias_mm']:+.1f}" for lab, v in sorted(by_lab.items(), key=lambda t: float(t[0])))
            res["measurement"].setdefault(label(s), {})[scope] = st
            if st["n_tags"]:
                L.append(f"   {label(s):30s}{st['n_tags']:5d}{st['bias_mm']:+8.2f}{st['bias_pct']:+8.2f}{st['mae_mm']:7.2f}"
                         f"{st['rms_mm']:7.2f}{st['max_abs_mm']:7.2f}   {per}")
            else:
                L.append(f"   {label(s):30s}    0")
    L.append("   -- by position on the target (all tags each setup found)")
    L.append(f"   {'setup':30s}" + "".join(f"{reg:>24s}" for reg in ("left column", "middle row", "right column")))
    for s in SETUPS:
        cells = []
        for reg in ("left column", "middle row", "right column"):
            st = edge_stats([r for r in meas[s] if r["region"] == reg])
            res["measurement"][label(s)][reg] = st
            cells.append(f"{st['n_tags']} tags, {st['bias_mm']:+.2f} / {st['mae_mm']:.2f}" if st["n_tags"] else "none")
        L.append(f"   {label(s):30s}" + "".join(f"{c:>24s}" for c in cells))
    L.append("   (cells: tags, bias / MAE in mm)")
    L.append("")
    L.append("   Tag reprojection (px) shows how well the two views agree for each measured tag:")
    for s in SETUPS:
        rp = [r["reprojection_px"] for r in meas[s]]
        L.append(f"   {label(s):30s}median {np.median(rp):.2f}, max {np.max(rp):.2f}" if rp else f"   {label(s):30s}none")
    res["measurement_rows"] = {label(s): meas[s] for s in SETUPS}

    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "accuracy.txt"), "w").write(text)
    json.dump(res, open(os.path.join(OUT, "accuracy.json"), "w"), indent=1)

    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    for k, s in enumerate(SETUPS):
        rows = meas[s]
        if rows:
            d = np.array([r["depth_mm"] for r in rows]) / 1000 + (k - (len(SETUPS) - 1) / 2) * 0.012
            ax.scatter(d, [np.mean(r["edges_mm"]) - TAG_SIZE_MM for r in rows], s=22, label=label(s),
                       edgecolor="white", linewidth=0.5, **STYLE[s])
    ax.axhline(0, color="#111", linewidth=0.8)
    ax.set_xlabel("measured depth of the tag (m)")
    ax.set_ylabel(f"mean edge error (mm), printed {TAG_SIZE_MM:g} mm")
    ax.set_title("WaterComparison target: one point per tag", loc="left", fontsize=10)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=8, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_accuracy.{ext}"), dpi=200)
    print(text)


if __name__ == "__main__":
    main()

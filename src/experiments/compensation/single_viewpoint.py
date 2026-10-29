"""Single-viewpoint test: does the standard OpenCV calibration keep its scale beyond its calibration range?"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common  # noqa: E402
from common import Calibration as C  # noqa: E402
sys.path.insert(0, os.path.join(common.SRC_DIR, "03_Reconstruction"))
from stereo_rig import load_rig  # noqa: E402

CONFIG_FILE = os.path.join(HERE, "config_single_viewpoint.json")
REFINEMENTS = {"contour": cv2.aruco.CORNER_REFINE_CONTOUR, "none": cv2.aruco.CORNER_REFINE_NONE,
               "subpix": cv2.aruco.CORNER_REFINE_NONE}   # subpix: the pipeline's cornerSubPix, applied after
MARK = "#1F5673"
RANGE_SHADE = "#d9e3ea"


# ====== CALIBRATION RANGE ======
def kept_images(path, camera):
    """Names of the images the calibration run kept for one camera."""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    node = fs.getNode(f"{camera}_camera").getNode("images")
    names = [node.at(i).string() for i in range(node.size())]
    fs.release()
    return names


def board_distances(rig, calibration_path):
    """Camera-to-board-centre distance (mm) of every left-camera image the calibration was fitted on."""
    corners = common.load_config(calibration_path.replace("_stereo.yaml", "_corners.json"))["cameras"]["left"]
    board = C.build_board()
    centre = np.vstack(board.getObjPoints()).reshape(-1, 3).mean(axis=0)
    distances = []
    for name in kept_images(calibration_path, "left"):
        d = corners[name]
        obj, img = board.matchImagePoints(tuple(np.float32(c).reshape(1, 4, 2) for c in d["corners"]),
                                          np.int32(d["ids"]).reshape(-1, 1))
        ok, rvec, tvec = cv2.solvePnP(obj, img, rig.K["left"], rig.D["left"])
        if ok:
            distances.append(float(np.linalg.norm(cv2.Rodrigues(rvec)[0] @ centre + tvec.ravel())))
    return np.array(distances)


# ====== MEASUREMENT ======
def make_detector(det_cfg, border_bits):
    """The preset's detector, with the printed target's border width and the configured refinement."""
    params = cv2.aruco.DetectorParameters()
    for key, value in {**C.load_presets().settings(det_cfg["preset"]), "markerBorderBits": border_bits}.items():
        setattr(params, key, value)
    params.cornerRefinementMethod = REFINEMENTS[det_cfg["corner_refinement"]]
    return cv2.aruco.ArucoDetector(C.dictionary, params)


def detect_target(detector, det_cfg, path, ids):
    """{id: 4x2 corners (px)} of the target tags found once each in one image."""
    gray = C.channel_image(cv2.imread(path, cv2.IMREAD_COLOR), "gray")
    clahe = cv2.createCLAHE(clipLimit=det_cfg["clahe_clip_limit"], tileGridSize=tuple(det_cfg["clahe_tile_grid"]))
    corners, found, _ = detector.detectMarkers(clahe.apply(gray))
    if found is None:
        return {}
    if det_cfg["corner_refinement"] == "subpix":
        corners = C.refine_corners(gray, corners)
    flat = [int(i) for i in found.ravel()]
    return {i: c.reshape(4, 2) for i, c in zip(flat, corners) if i in ids and flat.count(i) == 1}


def measure(rig, cfg):
    """One row per tag seen in both cameras: label, id, depth, the four edge lengths, stereo reprojection."""
    tgt, det_cfg = cfg["target"], cfg["detection"]
    frames, ids = common.project_path(tgt["frames"]), set(tgt["ids"])
    detector = make_detector(det_cfg, tgt["border_bits"])
    with open(os.path.join(frames, "pairs.csv"), newline="") as fh:
        pairs = sorted(csv.DictReader(fh), key=lambda r: float(r["label"]))
    rows = []
    for pair in pairs:
        L = detect_target(detector, det_cfg, os.path.join(frames, "left", pair["left_file"]), ids)
        R = detect_target(detector, det_cfg, os.path.join(frames, "right", pair["right_file"]), ids)
        for t in sorted(set(L) & set(R)):
            X = rig.triangulate(L[t], R[t])
            edges = np.linalg.norm(X - np.roll(X, -1, axis=0), axis=1)
            rows.append({"label_cm": float(pair["label"]), "id": t, "depth_mm": float(X[:, 2].mean()),
                         "distance_mm": float(np.linalg.norm(X.mean(axis=0))), "edges_mm": edges.tolist(),
                         "reprojection_px": rig.reprojection_error(X, L[t], R[t])})
    return [float(p["label"]) for p in pairs], rows


def tag_error(row, size):
    """A tag's scale error (mm): its median edge minus the printed edge."""
    return float(np.median(row["edges_mm"]) - size)


def edge_stats(rows, size):
    """Edge-length error (measured - printed) over all edges of the given tags, in mm and % of the size."""
    if not rows:
        return {"n_tags": 0}
    e = np.concatenate([r["edges_mm"] for r in rows]) - size
    return {"n_tags": len(rows), "n_edges": int(e.size), "depth_mm": float(np.mean([r["depth_mm"] for r in rows])),
            "bias_mm": float(e.mean()), "bias_pct": float(100 * e.mean() / size), "sd_mm": float(e.std(ddof=1)),
            "mae_mm": float(np.abs(e).mean()), "rms_mm": float(np.sqrt(np.mean(e ** 2))),
            "max_abs_mm": float(np.abs(e).max()),
            "tag_median_error_mm": float(np.median([tag_error(r, size) for r in rows]))}


# ====== REPORT ======
def report(cfg, labels, rows, distances):
    size, det = cfg["target"]["tag_size_mm"], cfg["detection"]
    lo, hi = float(distances.min()), float(distances.max())
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label_cm"]].append(r)
    res = {"calibration": cfg["calibration"], "detection": {k: v for k, v in det.items() if not k.startswith("_")},
           "tag_size_mm": size,
           "calibration_range_mm": {"images": int(distances.size), "min": lo, "max": hi,
                                    "p5": float(np.percentile(distances, 5)), "p95": float(np.percentile(distances, 95))},
           "per_label": {}, "inside_range": edge_stats([r for r in rows if r["distance_mm"] <= hi], size),
           "outside_range": edge_stats([r for r in rows if r["distance_mm"] > hi], size),
           "all": edge_stats(rows, size), "tags": rows}

    L = ["Single-viewpoint test: scale of the standard OpenCV calibration against distance",
         "Generated by src/experiments/compensation/single_viewpoint.py", "",
         f"Calibration: {cfg['calibration']}",
         f"Target: {cfg['target']['frames']}, tags {cfg['target']['ids']}, printed edge {size:g} mm",
         f"Detection: preset {det['preset']}, CLAHE {det['clahe_clip_limit']:g} / {det['clahe_tile_grid']}, "
         f"corner refinement {det['corner_refinement']}", "",
         f"Calibration range: {distances.size} left images kept, camera to board centre {lo:.0f}-{hi:.0f} mm "
         f"(5-95%: {res['calibration_range_mm']['p5']:.0f}-{res['calibration_range_mm']['p95']:.0f} mm)", "",
         "Edge error = triangulated edge - printed edge, over the four edges of every tag found in both cameras.",
         f"{'label cm':>9s}{'tags':>6s}{'depth mm':>10s}{'bias mm':>9s}{'bias %':>8s}{'SD':>7s}{'MAE':>7s}"
         f"{'RMS':>7s}{'max':>7s}{'tag med':>9s}  range",
         "(bias .. max: over all edges; tag med: median over tags of each tag's median edge error)"]
    for lab in labels:
        st = edge_stats(by_label[lab], size)
        res["per_label"][f"{lab:g}"] = st
        if not st["n_tags"]:
            L.append(f"{lab:9g}{0:6d}   (no tag found in both cameras)")
            continue
        where = "inside" if all(r["distance_mm"] <= hi for r in by_label[lab]) else "OUTSIDE"
        L.append(f"{lab:9g}{st['n_tags']:6d}{st['depth_mm']:10.0f}{st['bias_mm']:+9.2f}{st['bias_pct']:+8.2f}"
                 f"{st['sd_mm']:7.2f}{st['mae_mm']:7.2f}{st['rms_mm']:7.2f}{st['max_abs_mm']:7.2f}"
                 f"{st['tag_median_error_mm']:+9.2f}  {where}")
    L.append("")
    for key, name in (("inside_range", f"within calibration range (<= {hi:.0f} mm)"),
                      ("outside_range", f"beyond calibration range (> {hi:.0f} mm)"), ("all", "all tags")):
        st = res[key]
        L.append(f"{name:45s}" + (f"{st['n_tags']:3d} tags, bias {st['bias_mm']:+.2f} mm ({st['bias_pct']:+.2f}%), "
                                  f"RMS {st['rms_mm']:.2f} mm, tag median {st['tag_median_error_mm']:+.2f} mm"
                                  if st["n_tags"] else "  0 tags"))
    if len(rows) > 2:
        d = np.array([r["distance_mm"] for r in rows]) / 1000
        e = np.array([tag_error(r, size) for r in rows])
        slope, intercept = np.polyfit(d, e, 1)
        res["trend"] = {"slope_mm_per_m": float(slope), "intercept_mm": float(intercept)}
        L.append(f"\nLinear trend of tag (median-edge) error with distance: {slope:+.2f} mm per m "
                 f"({100 * slope / size:+.2f}% of the edge per m)")
    rp = [r["reprojection_px"] for r in rows]
    if rp:
        L.append(f"Stereo reprojection of the measured tags: median {np.median(rp):.2f} px, max {np.max(rp):.2f} px")
    return "\n".join(L) + "\n", res


def plot(rows, distances, size, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    ax.axvspan(distances.min() / 1000, distances.max() / 1000, color=RANGE_SHADE, linewidth=0)
    ax.text(distances.max() / 1000, 0.02, " calibration range ", transform=ax.get_xaxis_transform(),
            ha="right", va="bottom", fontsize=8, color="#444")
    ax.axhline(0, color="#111", linewidth=0.8)
    if rows:
        ax.scatter([r["distance_mm"] / 1000 for r in rows], [tag_error(r, size) for r in rows],
                   s=36, color=MARK, edgecolor="white", linewidth=1, zorder=3)
    ax.set_xlabel("distance from left camera to tag (m)")
    ax.set_ylabel(f"tag median edge error (mm), printed {size:g} mm")
    ax.set_title("Water calibration (water_final), WaterComparison tags: one point per tag", loc="left", fontsize=10)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_single_viewpoint.{ext}"), dpi=200)
    plt.close(fig)


def write_tags_csv(rows, size, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label_cm", "id", "depth_mm", "distance_mm", "edge1_mm", "edge2_mm", "edge3_mm", "edge4_mm",
                    "median_edge_error_mm", "reprojection_px"])
        for r in rows:
            w.writerow([f"{r['label_cm']:g}", r["id"], f"{r['depth_mm']:.1f}", f"{r['distance_mm']:.1f}",
                        *(f"{e:.2f}" for e in r["edges_mm"]), f"{tag_error(r, size):+.2f}",
                        f"{r['reprojection_px']:.3f}"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=CONFIG_FILE, help="settings (default config_single_viewpoint.json here)")
    cfg = common.load_config(ap.parse_args(argv).config)
    out = common.project_path(cfg["output"])
    os.makedirs(out, exist_ok=True)

    calibration_path = common.project_path(cfg["calibration"])
    rig = load_rig(calibration_path)
    distances = board_distances(rig, calibration_path)
    labels, rows = measure(rig, cfg)
    text, res = report(cfg, labels, rows, distances)

    with open(os.path.join(out, "single_viewpoint.txt"), "w") as fh:
        fh.write(text)
    with open(os.path.join(out, "single_viewpoint.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    write_tags_csv(rows, cfg["target"]["tag_size_mm"], os.path.join(out, "tags.csv"))
    plot(rows, distances, cfg["target"]["tag_size_mm"], out)
    print(text)
    print(f"wrote {common.rel(out)}/")


if __name__ == "__main__":
    main()

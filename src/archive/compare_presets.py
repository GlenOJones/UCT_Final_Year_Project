"""Compare two detector presets on the channel-selection runs, with merged detection and plain gray.

Reads the saved outputs of the two studies (study.json = wide, study_opencv_default.json = stock OpenCV
thresholding); nothing is detected or calibrated here. Writes to results/analysis/preset_merge_comparison/:
    fig_recall_presets.{pdf,png}   mean recall against distance, Water and Air, per preset and channel
    comparison.txt                 recall, usable images, reprojection, stereo, intrinsics with OpenCV's
                                   1-sigma, board-pose (extrinsic) 1-sigma, and the full detector settings

Use (from the project root, after run_study.py and analyse_study.py for both studies):
    src/venv/bin/python src/analysis/channel_selection/compare_presets.py
"""
import json
import os
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_study  # noqa: E402
from run_study import Calibration, PROJECT_ROOT  # noqa: E402

OUT = os.path.join(PROJECT_ROOT, "results", "analysis", "preset_merge_comparison")
STUDIES = {"opencv_default": "study_opencv_default.json", "wide": "study.json"}
CHANNELS = ("gray", "merge")
# One colour per preset, line style per channel, so preset and channel read independently.
COLOUR = {"opencv_default": "#9a9a9a", "wide": "#1F5673"}
LINE = {"gray": dict(linestyle="--", marker="o"), "merge": dict(linestyle="-", marker="P")}
INTRINSICS = ("fx_px", "fy_px", "cx_px", "cy_px", "k1", "k2", "p1", "p2", "k3")


def read_yaml(path):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    seq = lambda n: [n.at(i).string() if n.at(i).isString() else n.at(i).real() for i in range(n.size())]  # noqa: E731
    run = {"cameras": {}}
    for cam in Calibration.CAMERAS:
        node = fs.getNode(f"{cam}_camera")
        K = node.getNode("camera_matrix").mat()
        D = node.getNode("distortion_coefficients").mat().ravel()
        sig = node.getNode("uncertainty_1sigma")
        pose = node.getNode("pose_uncertainty_1sigma")
        run["cameras"][cam] = {
            "values": dict(zip(INTRINSICS, [K[0, 0], K[1, 1], K[0, 2], K[1, 2], *D[:5]])),
            "sigma": {k: sig.getNode(k).real() for k in INTRINSICS},
            "rms": node.getNode("rms_reprojection_error_px").real(),
            "n_images": int(node.getNode("images_used").real()),
            "per_image": dict(zip(seq(node.getNode("images")), seq(node.getNode("per_image_error_px")))),
            "pose_rot_deg": np.array(seq(pose.getNode("rotation_deg"))),
            "pose_trans_mm": np.array(seq(pose.getNode("translation_mm"))),
        }
    st = fs.getNode("stereo")
    run["stereo"] = {"rms": st.getNode("rms_reprojection_error_px").real(), "baseline": st.getNode("baseline_mm").real(),
                     "pairs": int(st.getNode("pairs_used").real()),
                     "angles": [st.getNode("rotation_angles_deg").at(i).real() for i in range(3)],
                     "T": st.getNode("translation_mm").mat().ravel()}
    fs.release()
    return run


def rms(v):
    return float(np.sqrt(np.mean(np.square(v)))) if len(v) else float("nan")


def detector_settings(preset):
    """Every DetectorParameters value the pipeline uses for this preset, and the OpenCV default beside it."""
    defaults = cv2.aruco.DetectorParameters()
    used = {**Calibration.load_presets().settings(preset), "cornerRefinementMethod": cv2.aruco.CORNER_REFINE_NONE}
    return {k: (v, getattr(defaults, k)) for k, v in sorted(used.items())}


def main():
    os.makedirs(OUT, exist_ok=True)
    studies = {p: run_study.load_study(os.path.join(HERE, f)) for p, f in STUDIES.items()}
    results = {p: json.load(open(os.path.join(s["out_dir"], "results.json"))) for p, s in studies.items()}
    runs = {(p, m, c): read_yaml(run_study.yaml_path(studies[p], m, c))
            for p in STUDIES for m in ("Water", "Air") for c in CHANNELS}

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True, constrained_layout=True)
    for ax, medium in zip(axes, ("Water", "Air")):
        for p in STUDIES:
            for c in CHANNELS:
                d = results[p]["detection"][medium][c]
                by = d["by_label"]
                ax.plot([float(k) for k in by], [v["recall_mean"] for v in by.values()], color=COLOUR[p],
                        linewidth=1.6, markersize=4.5, label=f"{p}, {c}: {d['recall_mean']:.3f}", **LINE[c])
        ax.set_title(medium, loc="left", fontsize=10)
        ax.set_xlabel("board distance (cm)")
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.legend(title="preset, channel: mean recall", title_fontsize=7, fontsize=7, frameon=False, loc="upper right")
    axes[0].set_ylabel("mean recall (distinct tags / 70)")
    axes[0].set_ylim(0, 1.0)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_recall_presets.{ext}"), dpi=200)

    # ---- text report ----
    L = ["Preset comparison: opencv_default vs wide, gray and merged (gray+G+R+B) detection",
         "Generated by src/analysis/channel_selection/compare_presets.py from the two channel-selection studies.", ""]
    for medium in ("Water", "Air"):
        L += [f"===== {medium} =====", "",
              f"{'run':24s}{'recall':>8s}{'usable L/R':>12s}{'RMS all L/R':>14s}{'RMS common L/R':>16s}"
              f"{'stereo RMS':>11s}{'pairs':>6s}{'baseline mm':>12s}"]
        keys = [(p, medium, c) for p in STUDIES for c in CHANNELS]
        common = {cam: set.intersection(*(set(runs[k]["cameras"][cam]["per_image"]) for k in keys)) for cam in Calibration.CAMERAS}
        for k in keys:
            r, cams = runs[k], runs[k]["cameras"]
            L.append(f"{k[0] + ', ' + k[2]:24s}{results[k[0]]['detection'][medium][k[2]]['recall_mean']:8.3f}"
                     f"{cams['left']['n_images']:>6d}/{cams['right']['n_images']:<5d}"
                     f"{cams['left']['rms']:>8.3f}/{cams['right']['rms']:<5.3f}"
                     f"{rms([cams['left']['per_image'][n] for n in common['left']]):>10.3f}/"
                     f"{rms([cams['right']['per_image'][n] for n in common['right']]):<5.3f}"
                     f"{r['stereo']['rms']:>11.3f}{r['stereo']['pairs']:>6d}{r['stereo']['baseline']:>12.2f}")
        L.append(f"(common set: the {len(common['left'])} left / {len(common['right'])} right images all four runs used)")
        L.append("")
        for cam in Calibration.CAMERAS:
            L.append(f"{cam} camera intrinsics, value +- OpenCV 1-sigma (calibrateCameraExtended stdDeviationsIntrinsics)")
            L.append(f"  {'':22s}" + "".join(f"{n:>21s}" for n in INTRINSICS))
            for k in keys:
                c = runs[k]["cameras"][cam]
                fmt = lambda n: (f"{c['values'][n]:.2f} +- {c['sigma'][n]:.2f}" if n.endswith("_px")  # noqa: E731
                                 else f"{c['values'][n]:.4f} +- {c['sigma'][n]:.4f}")
                L.append(f"  {k[0] + ', ' + k[2]:22s}" + "".join(f"{fmt(n):>21s}" for n in INTRINSICS))
            L.append(f"{cam} camera board-pose (extrinsic) 1-sigma per image (stdDeviationsExtrinsics): median / max")
            for k in keys:
                c = runs[k]["cameras"][cam]
                L.append(f"  {k[0] + ', ' + k[2]:22s}rotation {np.median(c['pose_rot_deg']):.3f} / {c['pose_rot_deg'].max():.3f} deg"
                         f"   translation {np.median(c['pose_trans_mm']):.2f} / {c['pose_trans_mm'].max():.2f} mm")
            L.append("")
        L.append("Stereo extrinsics (left -> right). OpenCV's stereoCalibrate gives no uncertainty for these.")
        for k in keys:
            s = runs[k]["stereo"]
            L.append(f"  {k[0] + ', ' + k[2]:22s}T = ({s['T'][0]:.2f}, {s['T'][1]:.2f}, {s['T'][2]:.2f}) mm, "
                     f"|T| {s['baseline']:.2f} mm, rotation about x,y,z ({s['angles'][0]:.3f}, {s['angles'][1]:.3f}, {s['angles'][2]:.3f}) deg")
        L.append("")

    L += ["===== Detector settings (cv2.aruco.DetectorParameters) =====",
          "Only parameters the pipeline sets are listed; every other parameter is at its OpenCV default.", "",
          f"{'parameter':32s}" + "".join(f"{p:>16s}" for p in STUDIES) + f"{'OpenCV default':>16s}"]
    sets = {p: detector_settings(p) for p in STUDIES}
    for name in sorted(set().union(*sets.values())):
        default = getattr(cv2.aruco.DetectorParameters(), name)
        L.append(f"{name:32s}" + "".join(f"{str(sets[p].get(name, (default,))[0]):>16s}" for p in STUDIES) + f"{str(default):>16s}")
    L += ["", "Dictionary: DICT_APRILTAG_36h11. cornerRefinementMethod 0 = CORNER_REFINE_NONE: the detector's own",
          f"refinement is off; corners are refined afterwards with cv2.cornerSubPix, half-window "
          f"{Calibration.SUBPIX_WIN_FRACTION:g} x tag side (min {Calibration.SUBPIX_WIN_MIN} px), criteria "
          f"{Calibration.SUBPIX_CRITERIA[1]} iterations / eps {Calibration.SUBPIX_CRITERIA[2]}.",
          f"merge: detected separately in {', '.join(Calibration.MERGE_ORDER)} (each refined on its own channel); a tag found in",
          f"several takes the first one's corners, and is dropped if their corners differ by more than {Calibration.MERGE_TOLERANCE_PX:g} px.",
          "Calibration: cv2.calibrateCameraExtended per camera (default flags), then cv2.stereoCalibrate with",
          "CALIB_FIX_INTRINSIC on pairs where both cameras see >= 10 common tags; images need >= 10 tags."]
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "comparison.txt"), "w").write(text)
    print(text)
    print(f"wrote {os.path.relpath(OUT, PROJECT_ROOT)}/fig_recall_presets.{{pdf,png}} and comparison.txt")


if __name__ == "__main__":
    main()

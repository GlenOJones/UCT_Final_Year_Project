"""Build the pose-estimation report (PDF) from a finished run.

    src/venv/bin/python src/experiments/pose_estimation/make_report.py [--config PATH] [--recording NAME]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "04_pose"))
import common  # noqa: E402
import evaluate_poses  # noqa: E402
import pose_files  # noqa: E402
from run_poses import recording_paths  # noqa: E402

DISCUSSION = os.path.join(HERE, "report_discussion.tex")
LABELS = {"tags": "tags (reference)", "colmap": "COLMAP SfM",
          "icp_sgbm_f2f_plane": "ICP SGBM f2f", "icp_sgbm_map_plane": "ICP SGBM map",
          "icp_sgbm_map_colored": "ICP SGBM map col.", "icp_fs_f2f_plane": "ICP FS f2f",
          "icp_fs_map_plane": "ICP FS map", "icp_fs_map_colored": "ICP FS map col."}
COLOURS = {"colmap": "#1f5fa8", "icp_sgbm_f2f_plane": "#e0a33a", "icp_sgbm_map_plane": "#c4561c",
           "icp_sgbm_map_colored": "#8c3a12", "icp_fs_f2f_plane": "#7fbf7b", "icp_fs_map_plane": "#2b7a3d",
           "icp_fs_map_colored": "#14481f"}


def tex(text):
    return (str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#"))


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fmt(value, spec, missing="--"):
    return missing if value is None else format(value, spec)


def table(caption, label, header, rows, align):
    lines = [r"\begin{table}[H]\centering\small", r"\caption{" + caption + "}", r"\label{tab:" + label + "}",
             r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(str(c) for c in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def figure(name, caption, width=r"\textwidth"):
    return (r"\begin{figure}[H]\centering" + f"\n\\includegraphics[width={width}]{{{name}}}\n"
            + r"\caption{" + caption + r"}\label{fig:" + name.split(".")[0] + "}\n\\end{figure}")


# ====== READING THE RUN ======
def load_run(cfg, recording):
    p = recording_paths(cfg, recording)
    evaluation = os.path.join(p["out"], "evaluation")
    run = {"p": p, "cfg": cfg, "recording": recording, "metrics": load_json(os.path.join(evaluation, "metrics.json")),
           "info": load_json(os.path.join(evaluation, "run_info.json")),
           "masks": load_json(os.path.join(p["masks"], "masks.json")),
           "depth": {s: load_json(os.path.join(p["depth"](s), "depth_frames.json")) for s in cfg["depth"]
                     if os.path.exists(os.path.join(p["depth"](s), "depth_frames.json"))},
           "detections": None, "estimators": {}}
    det = os.path.join(evaluation, "detections", "detections.json")
    if os.path.exists(det):
        run["detections"] = load_json(det)
    meta, reference, diag = pose_files.read_trajectory(p["reference"])
    run["reference"] = reference
    fits = [d["fit_rms_mm"] for d in diag.values() if "fit_rms_mm" in d]
    fs = cv2.FileStorage(p["reference"], cv2.FILE_STORAGE_READ)
    run["tags"] = {"frames": len(diag), "posed": len(reference), "fit_median": float(np.median(fits)),
                   "measured_size": fs.getNode("tags").getNode("measured_size_mm").real(),
                   "printed_size": fs.getNode("tags").getNode("printed_size_mm").real()}
    fs.release()
    for e in cfg["estimators"]:
        path = p["poses"](e["name"])
        if os.path.exists(path):
            m, est, d = pose_files.read_trajectory(path)
            result, trace = evaluate_poses.evaluate(reference, est, d, list(d), cfg)
            run["estimators"][e["name"]] = {"spec": e, "meta": m, "diag": d, "trace": trace, "result": result}
    return run


# ====== FIGURES ======
def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def fig_accuracy(run, out):
    plt = plt_setup()
    names = list(run["estimators"])
    ate = [run["estimators"][n]["result"]["ate_mm"]["rmse"] for n in names]
    worst = [run["estimators"][n]["result"]["ate_mm"]["max"] for n in names]
    rpe = [run["estimators"][n]["result"]["rpe_percent"]["median"] for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    y = np.arange(len(names))[::-1]
    ax[0].barh(y, ate, color=[COLOURS[n] for n in names])
    ax[0].scatter(worst, y, marker="|", s=120, color="black", label="worst frame")
    ax[0].set(xlabel="position error after one rigid fit (mm): bar RMSE", title="Absolute trajectory error")
    ax[0].legend(frameon=False, loc="lower right")
    ax[1].barh(y, rpe, color=[COLOURS[n] for n in names])
    ax[1].set(xlabel=f"error of the motion over {run['cfg']['evaluation']['rpe_distance_mm']} mm of travel (%), median",
              title="Relative pose error")
    ax[0].set_yticks(y, [LABELS[n] for n in names])
    for a, values in zip(ax, (ate, rpe)):
        for yi, v in zip(y, values):
            a.text(v, yi, f" {v:.1f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_accuracy.pdf"))
    plt.close(fig)


def fig_track(run, out, chosen):
    plt = plt_setup()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    for n in chosen:
        t = run["estimators"][n]["trace"]
        ax.plot(t["track"], t["ate_t"], lw=1.1, color=COLOURS[n], label=LABELS[n])
    ax.set(xlabel="reference travel along the pass (mm)", ylabel="position error (mm)",
           title="Position error along the pass, after one rigid fit of the whole trajectory")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_track.pdf"))
    plt.close(fig)


def fig_components(run, out, chosen):
    plt = plt_setup()
    fig, ax = plt.subplots(3, 1, figsize=(10, 6.6), sharex=True)
    titles = ("along the track", "across the track, in the board plane", "height above the board")
    for n in chosen:
        t = run["estimators"][n]["trace"]
        d = t["direction"].copy()
        d[2] = 0
        d /= np.linalg.norm(d)
        across = np.cross([0, 0, 1.0], d)
        error = t["xyz_est"] - t["xyz_ref"]
        for a, axis in zip(ax, (d, across, np.array([0, 0, 1.0]))):
            a.plot(t["track"], error @ axis, lw=1, color=COLOURS[n], label=LABELS[n])
    for a, title in zip(ax, titles):
        a.axhline(0, color="black", lw=0.6)
        a.set_ylabel("error (mm)")
        a.set_title(f"Camera position error {title}", fontsize=9, loc="left")
        a.grid(alpha=0.3)
    ax[0].legend(frameon=False, ncol=3)
    ax[-1].set_xlabel("reference travel along the pass (mm)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_components.pdf"))
    plt.close(fig)


def fig_constraint(run, out, chosen):
    plt = plt_setup()
    fig, ax = plt.subplots(figsize=(10, 3.0))
    for n in chosen:
        e = run["estimators"][n]
        t = e["trace"]
        values = [e["diag"][name].get("translation_constraint", np.nan) for name in t["names"]]
        ax.plot(t["track"], values, lw=1, color=COLOURS[n], label=LABELS[n])
    ax.set(xlabel="reference travel along the pass (mm)", ylabel=r"$\lambda_{min}/\lambda_{max}$",
           title="How well the surface pins ICP's translation (0 = free to slide in some direction)", ylim=(0, None))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_constraint.pdf"))
    plt.close(fig)


def fig_runtime(run, out):
    plt = plt_setup()
    names = list(run["estimators"])
    depth_min = [(run["estimators"][n]["meta"].get("depth_seconds") or 0) / 60 for n in names]
    pose_min = [(run["estimators"][n]["meta"].get("seconds") or 0) / 60 for n in names]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    y = np.arange(len(names))[::-1]
    ax.barh(y, depth_min, color="#b9c3cc", label="per-frame depth (depth_frames.py)")
    ax.barh(y, pose_min, left=depth_min, color=[COLOURS[n] for n in names], label="pose estimation")
    ax.set_yticks(y, [LABELS[n] for n in names])
    ax.set(xlabel=f"minutes for {run['estimators'][names[0]]['result']['frames']} frame pairs (RTX 5060 laptop GPU)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_runtime.pdf"))
    plt.close(fig)


def fig_masks(run, out, name):
    """The masked tags on one frame, both cameras."""
    plt = plt_setup()
    frames = common.project_path(run["recording"]["frames"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.9))
    for a, camera in zip(ax, ("left", "right")):
        image = cv2.imread(os.path.join(frames, camera, name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(run["p"]["masks"], camera, name), cv2.IMREAD_GRAYSCALE)
        a.imshow(np.where(mask > 0, image, image // 4), cmap="gray", vmin=0, vmax=255)
        a.contour(mask == 0, levels=[0.5], colors="tab:red", linewidths=0.8)
        a.set_title(f"{camera} camera, frame {name}")
        a.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_masks.pdf"))
    plt.close(fig)


# ====== SECTIONS ======
PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx,hyperref,caption,float,xcolor}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\captionsetup{font=small,labelfont=bf}
\setlength{\parskip}{0.5em}\setlength{\parindent}{0pt}
\hypersetup{colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black}
\title{Markerless pose estimation for the underwater stereo rig:\\ COLMAP and ICP against the tag poses}
\author{Glen Jones -- UCT final-year project\\\small generated by \texttt{src/experiments/pose\_estimation/make\_report.py}}
"""


def section_question(run):
    best = min(run["estimators"], key=lambda n: run["estimators"][n]["result"]["ate_mm"]["rmse"])
    r = run["estimators"][best]["result"]
    return (r"""\section{The question}
Every reconstruction so far has placed each frame using the AprilTags on the board: \texttt{tag\_poses.py} gives
the rig's pose in every frame, and the per-frame depth is moved into one frame with it. The deployment scans ice,
which carries no tags. This experiment asks how well the pose of every frame can be recovered from the images
alone, and with which method. Two families were tested, with the tags hidden from both:
\begin{itemize}
\item \textbf{COLMAP structure-from-motion}, which matches image features between frames and solves all poses
together, with the stereo rig and the intrinsics held at their calibrated values;
\item \textbf{ICP} (iterative closest point), which registers each frame's depth onto the frames before it, with
the depth from SGBM or from FoundationStereo, frame-to-frame or onto a local map of recent frames, and with
point-to-plane or coloured ICP.
\end{itemize}
Each is scored against the tag poses of the same frames. Only the poses are scored here, not the reconstructions
made with them. The most accurate trajectory overall came from """ + tex(LABELS[best])
            + f" ({r['ate_mm']['rmse']:.2f}~mm RMS position error over {r['travel_mm']:.0f}~mm of travel); the full "
              r"results are in Section~\ref{sec:results}.")


def section_data(run):
    rec, t, m = run["recording"], run["tags"], run["masks"]
    first = next(iter(run["estimators"].values()))["result"]
    return (r"""\section{Data and reference}
\textbf{Recording.} """ + f"\\nolinkurl{{{rec['frames']}}}: {t['frames']} timestamp-matched frame pairs "
            r"(1600$\times$1200, grey) of one pass of the rig on its trolley over the 3DPrintFinal board, about "
            r"300--480~mm above it. The board carries five 50~mm AprilTags and the 3D-printed test objects; "
            r"around it the images also show the tray, bricks and sand. Every "
            + f"{run['cfg']['frame_step']}th pair is used: {first['frames']} frame pairs, the same for every method."
            + "\n\n" + r"\textbf{Calibration.} " + f"\\nolinkurl{{{rec['calibration']}}} (underwater, Brown-Conrady)."
            + "\n\n" + r"\textbf{Reference.} " + f"\\texttt{{tag\\_poses.py}} posed {t['posed']} of the {t['frames']} "
              f"frames ({first['with_reference']} of the {first['frames']} used ones). The tag layout fits each frame "
              f"with a median residual of {t['fit_median']:.2f}~mm, and the tags triangulate to "
              f"{t['measured_size']:.2f}~mm against {t['printed_size']:.0f}~mm printed. The tag poses are the "
              r"yardstick, not the truth: differences of a fraction of a millimetre are below their own noise."
            + "\n\n" + r"\textbf{Tags hidden.} \texttt{make\_tag\_masks.py} masks each tag, and the white margin "
              f"around it (mask side {run['cfg']['tag_masks']['scale']}$\\times$ the tag), wherever it is detected in "
              r"an image and wherever the tag layout projects it in a tag-posed frame, which also covers tags cut off "
              f"at the image edge. Of the {m['frames']} frames, {m['projected']} were masked from the layout, "
              f"{m['detected_only']} from detections only, and {m['unmasked']} had no tag found and were left "
              f"unmasked. A median {100 * m['masked_fraction_median']:.1f}\\% of each image is masked "
              r"(Figure~\ref{fig:fig_masks}). COLMAP extracts no features under the mask, and the depth used by ICP "
              r"is made with the masked pixels removed."
            + "\n" + figure("fig_masks.pdf", "The tag masks on one frame (masked pixels darkened, outline in red). "
                            "The scene outside the board (tray, bricks, sand) is left visible, as it is part of the "
                            "real view.", width=r"0.95\textwidth"))


def section_methods(run):
    d = run["depth"]
    cfg = run["cfg"]
    rows = []
    for source, rec in d.items():
        s = rec["settings"]
        what = ("SGBM, block " + str(s["block_size"]) + ", uniqueness " + str(s["uniqueness"])) if s["matcher"] == "sgbm" \
            else f"FoundationStereo {s['fs_model']}, scale {s['net_scale']}, {s['net_iters']} iterations, L-R check {s['lr_tolerance']}~px"
        rows.append([tex(source), tex(what), f"{rec['points_per_frame_median'] / 1000:.0f}k",
                     f"{rec['seconds_per_frame']:.1f}", f"{rec['seconds'] / 60:.0f}"])
    icp = next(e["spec"]["argv"] for e in run["estimators"].values() if e["spec"]["kind"] == "icp")
    opt = {icp[i]: icp[i + 1] for i in range(0, len(icp) - 1) if icp[i].startswith("--") and not icp[i + 1].startswith("--")}
    ev = cfg["evaluation"]
    return (r"""\section{Methods}
\subsection{Per-frame depth without poses}
\texttt{dense\_stereo.py} needs a pose before it matches: the tag layout crops each frame to a box around the
board and sets SGBM's search range. A tag-free scan has no pose yet, so \texttt{depth\_frames.py} was written:
it rectifies each pair and runs the same matchers, keeps the points between """
            + f"{run['recording']['depth_min_mm']} and {run['recording']['depth_max_mm']}" + r"""~mm from the camera
(which also sets SGBM's disparity range), drops masked pixels, and stores each frame's points in its own left
camera frame, one per 1~mm voxel. ICP then works on these points only.
""" + table("The two depth sources: points kept per frame (median) and time on the RTX 5060 laptop GPU.", "depth",
            ["source", "matcher", "points", "s/frame", "min total"], rows, "lp{8.6cm}rrr")
            + r"""
\subsection{COLMAP structure-from-motion}
\texttt{colmap\_poses.py} runs the structure-from-motion half of \texttt{colmap\_mvs.py}, with the same functions:
SIFT per camera with the calibrated intrinsics fixed (FULL\_OPENCV), the two cameras declared a rig with the
calibrated extrinsics, matching of every camera combination within """
            + f"{next((e['spec']['argv'][1] for e in run['estimators'].values() if e['spec']['kind'] == 'colmap'), '12')}"
            + r""" frames either side, and incremental mapping with intrinsics and rig held fixed. The fixed 150~mm
baseline gives the model metric scale. The tag masks are passed to SIFT, so no feature is taken on a tag.

\subsection{ICP on the depth}
\texttt{icp\_poses.py} chains the frames: each is registered onto what came before, and its pose is the previous
pose times the registration. Each frame is thinned to """ + f"{opt.get('--scales', '4')} and 2" + r"""~mm voxels,
with normals facing the camera, and registered coarse to fine with a search distance of """
            + f"{opt['--max-distance']}" + r"""~voxels. The variants:
\begin{itemize}
\item \textbf{f2f}: onto the previous frame only; \textbf{map}: onto the last """ + f"{opt['--map-frames']}" + r"""
registered frames merged (a local map);
\item \textbf{plane}: point-to-plane ICP on geometry; \textbf{colored}: Park et al.'s coloured ICP, which adds the
grey level.
\end{itemize}
The starting guess for each frame is the previous pose moved by the \emph{translation} of the last registered
step. A frame is marked \emph{lost} if fewer than """ + f"{float(opt['--min-fitness']) * 100:.0f}" + r"""\% of its
points find a match, or if the registration moves the rig more than """ + f"{opt['--max-step']}~mm or {opt['--max-rotation']}"
            + r"""$^\circ$ from the previous frame (the trolley moves at most 15~mm between used frames). A lost frame
takes the guess and is kept out of the map. Section~\ref{sec:dev} explains why both rules are needed.

For every frame ICP also records how well the surface pinned its translation. Point-to-plane ICP minimises
$\sum_i \big(\mathbf{n}_i\cdot(R\mathbf{p}_i+\mathbf{t}-\mathbf{q}_i)\big)^2$, so each match constrains the
translation only along its normal $\mathbf{n}_i$. The translational block of the normal equations is
$H_t=\sum_i \mathbf{n}_i\mathbf{n}_i^\top$, and the ratio $\lambda_{\min}(H_t)/\lambda_{\max}(H_t)$ is near 0 when
the matched surface is close to one plane (or a set of parallel planes): ICP is then free to slide along it.
(Open3D's \texttt{get\_information\_matrix\_from\_point\_clouds} uses point-to-point terms, whose translational
block is always a multiple of the identity, so it cannot show this.)

\subsection{Scoring}
Each estimator poses the frames in a world frame of its own (the left camera of its first frame), so its
trajectory is brought into the board frame by one rigid fit over all frames that both it and the tags posed. The
fit uses each camera's centre and the points """ + f"{ev['align_lever_mm']}" + r"""~mm along its three axes, so
orientation counts as well as position; the centres of a near-straight pass alone would leave the roll about the
track almost free. Then, per frame $k$ with estimated camera-to-board pose $\hat T_k$ and reference $T_k$:
\begin{itemize}
\item \textbf{ATE} (absolute trajectory error): $\|\hat{\mathbf{c}}_k-\mathbf{c}_k\|$ for the camera centres, and the
angle of $R_k^\top\hat R_k$. Drift appears as error growing along the pass.
\item \textbf{RPE} (relative pose error): for each frame $i$ and the first frame $j$ at least """
            + f"{ev['rpe_distance_mm']}" + r"""~mm of reference travel further on, the error of the estimated motion,
$E=(T_i^{-1}T_j)^{-1}(\hat T_i^{-1}\hat T_j)$, as $\|\mathbf{t}_E\|$ in \% of the distance and as an angle. It needs
no alignment and measures local accuracy.
\item \textbf{Scale}: the similarity scale that best maps the estimated camera centres onto the reference ones
(1 = metric agreement).
\item \textbf{Registration}: frames posed (not lost) out of those given.
\end{itemize}
The scoring was checked three ways: the tag poses scored against themselves give exactly zero; the tag poses with
1~mm of Gaussian noise added per axis give an ATE of 1.73~mm ($\sqrt{3}$~mm, as expected); and the tag poses
scaled by 1.01 give an RPE of 1.00\% and a best-fit scale of 0.990.""")


def section_results(run):
    rows = []
    for n, e in run["estimators"].items():
        r = e["result"]
        total = ((e["meta"].get("depth_seconds") or 0) + (e["meta"].get("seconds") or 0)) / 60
        rows.append([tex(LABELS[n]), f"{r['posed']}/{r['frames']}", f"{r['ate_mm']['rmse']:.2f}", f"{r['ate_mm']['max']:.1f}",
                     f"{r['ate_deg']['rmse']:.2f}", f"{r['rpe_percent']['median']:.2f}", f"{r['rpe_percent']['p95']:.1f}",
                     f"{r['best_fit_scale']:.3f}", fmt(r.get("translation_constraint_median"), ".3f"), f"{total:.0f}"])
    icp_fs = [n for n in ("colmap", "icp_fs_map_plane", "icp_sgbm_map_plane", "icp_fs_f2f_plane", "icp_sgbm_f2f_plane")
              if n in run["estimators"]]
    s = [r"\section{Results}\label{sec:results}",
         table("Every estimator against the tag poses. ATE: camera-position error after one rigid fit (RMSE and worst "
               "frame, mm) and orientation error (RMSE, deg). RPE: error of the motion over "
               f"{run['cfg']['evaluation']['rpe_distance_mm']}~mm of travel (median and 95th percentile, \\%). Scale: "
               "best-fit similarity scale. Constraint: median translation constraint of ICP's matches. Time: depth "
               "plus pose estimation, minutes.", "results",
               ["estimator", "posed", "ATE mm", "max", "deg", "RPE \\%", "p95", "scale", "constr.", "min"], rows,
               "lrrrrrrrrr"),
         figure("fig_accuracy.pdf", "Accuracy of every estimator: position error over the whole pass (left; bar: "
                "RMSE, tick: worst frame) and local motion error (right)."),
         figure("fig_track.pdf", "Position error along the pass. The coloured-ICP variants are left out: they lie on "
                "their point-to-plane counterparts."),
         figure("fig_components.pdf", "The position error split into its direction along the track, across it in the "
                "board plane, and height. Errors that grow steadily along the track are drift; a slope is a scale error."),
         figure("fig_constraint.pdf", "Translation constraint of ICP's matches along the pass, for the two "
                "local-map variants."),
         figure("fig_runtime.pdf", "Time per method. ICP's cost is almost all in the depth it needs; COLMAP needs no "
                "depth.")]
    _ = icp_fs
    det = run["detections"]
    if det:
        names = list(det)
        rows = [[tex(n), d["keypoints"], d["triangulated"], fmt(d["triangulated_on_board_%"], ".0f"),
                 f"{d['icp_on_board_%']:.0f}", f"{d['icp_resisting_slide_%']:.0f}",
                 fmt(d["icp_resisting_slide_on_board_%"], ".0f")] for n, d in det.items()]
        s.append(r"\subsection{Where each method gets its grip}"
                 "\nFor four frames along the pass, Figures~\\ref{fig:det_a} and~\\ref{fig:det_b} show where COLMAP's "
                 "features and ICP's constraints come from. Left: every SIFT keypoint, green if the final model "
                 "triangulated it. Right: the frame's FoundationStereo depth at 2~mm, coloured by "
                 "$|\\mathbf{n}\\cdot\\mathbf{d}|$, how much each point resists a slide along the direction of travel "
                 "$\\mathbf{d}$. The flat board scores near 0; surfaces facing along the track score near 1.\n"
                 + table("Per frame: COLMAP keypoints and how many were triangulated, the share of those on the "
                         "board; ICP's points on the board, the share resisting the slide ($|\\mathbf{n}\\cdot\\mathbf{d}|>0.5$) "
                         "and the share of those on the board.", "detections",
                         ["frame", "keypoints", "used", "used on board \\%", "ICP on board \\%", "resisting \\%",
                          "resisting on board \\%"], rows, "lrrrrrr"))
        out = os.path.join(run["p"]["out"], "report")
        for label, name in zip(("det_a", "det_b"), (names[1], names[2])):
            shutil.copy(os.path.join(run["p"]["out"], "evaluation", "detections", name), os.path.join(out, label + ".png"))
            s.append(figure(label + ".png", f"Frame {tex(name)}: COLMAP keypoints (left) and ICP's grip (right). "
                            "Red hatch: masked tags; gold dashes: the board."))
    return "\n\n".join(s)


DEVELOPMENT = r"""\section{Problems found while building the ICP estimator}\label{sec:dev}
Two failures in the first version of \texttt{icp\_poses.py} are recorded here, because they shaped its final
design and because both were found on the data this report scores.

\textbf{A wild step was repeated.} The first version marked a frame lost only on low fitness, and then gave it
the constant-velocity guess (the last motion repeated). On SGBM depth, frame-to-frame, one registration jumped
44~mm and the next 1.5~m; the guess repeated the jump and the trajectory left the scene for good. Steps larger than
the trolley can make (20~mm, $5^\circ$; the largest reference step between used frames is 15~mm) are now rejected
too, and a lost frame is kept out of the local map.

\textbf{The guess fed a rotation.} With that fixed, ICP on FoundationStereo depth still lost almost every frame
after the first 35, and the rejected steps all had high fitness. The rotation of each registered step grew by
about 1.5$\times$ per frame, while the true step stayed at 5--11~mm with no rotation:
\begin{center}\small
\begin{tabular}{lrrrrrrrrrrr}\toprule
frame & 0150 & 0155 & 0160 & 0165 & 0170 & 0175 & 0180 & 0185 & 0190 & 0195 & 0200\\\midrule
step rotation (deg) & 0.6 & 1.0 & 1.5 & 2.4 & 3.7 & 5.7 & 8.9 & 13.7 & 21.3 & 32.7 & 49.7\\
fitness & 0.99 & 0.99 & 0.98 & 0.98 & 0.99 & 0.99 & 1.00 & 0.99 & 0.97 & 0.96 & 0.87\\\bottomrule
\end{tabular}\end{center}
The constant-velocity guess repeated each step's rotation, and on this mostly flat scene ICP has almost no
restoring force against a small rotation combined with a slide, so it kept what the guess offered. The guess now
carries the translation of the last step only. With that change the same run posed 340 of 344 frames instead of 95.

Because both fixes were made after looking at this recording, the ICP rows are tuned on the data they are scored on.
"""


REPRODUCE = r"""\section{Reproducing this report}
Every step, from the raw frames; then the grip figures; then this PDF:
\begin{verbatim}
src/venv/bin/python src/experiments/reproduce.py pose_estimation
src/venv/bin/python src/experiments/pose_estimation/plot_detections.py
src/venv/bin/python src/experiments/pose_estimation/make_report.py
\end{verbatim}
Settings: \texttt{src/experiments/pose\_estimation/config.json}. Another recording is tested by adding it to
\texttt{recordings} there (frames, calibration, tag size, camera depth range).
"""


def section_discussion(run):
    if not os.path.exists(DISCUSSION):
        return r"\section{Discussion}" + "\nNo discussion written yet.\n"
    with open(DISCUSSION, encoding="utf-8") as fh:
        text = fh.read()
    written_for = next((ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.startswith("% written for run:")), None)
    if written_for and written_for != run["info"]["created"]:
        text = (r"\textbf{\color{red}Note: this discussion was written for the evaluation run of " + tex(written_for)
                + r"; the numbers above are from the run of " + tex(run["info"]["created"]) + r". Check it still holds.}"
                + "\n\n" + text)
    return text


def provenance(run):
    info = run["info"]
    dirty = info["git_uncommitted_src_changes"]
    return (r"\par\small Evaluation run " + tex(info["created"]) + ", code at commit \\texttt{" + info["git_commit"][:10] + "}"
            + (r" \textbf{with uncommitted changes in src/} (" + str(len(dirty)) + " files), so these results are a trial run."
               if dirty else ".") + r"\normalsize")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--recording", default=None, help="recording name (default: the config's first)")
    args = ap.parse_args(argv)
    cfg = common.load_config(args.config)
    recording = next(r for r in cfg["recordings"] if args.recording in (None, r["name"]))
    run = load_run(cfg, recording)
    out = os.path.join(run["p"]["out"], "report")
    os.makedirs(out, exist_ok=True)

    chosen = [n for n in ("colmap", "icp_fs_map_plane", "icp_sgbm_map_plane", "icp_fs_f2f_plane", "icp_sgbm_f2f_plane")
              if n in run["estimators"]]
    fig_accuracy(run, out)
    fig_track(run, out, chosen)
    fig_components(run, out, [n for n in ("colmap", "icp_fs_map_plane", "icp_sgbm_map_plane") if n in run["estimators"]])
    fig_constraint(run, out, [n for n in ("icp_fs_map_plane", "icp_sgbm_map_plane") if n in run["estimators"]])
    fig_runtime(run, out)
    fig_masks(run, out, "0600.png" if os.path.exists(os.path.join(run["p"]["masks"], "left", "0600.png"))
              else sorted(os.listdir(os.path.join(run["p"]["masks"], "left")))[0])

    body = [PREAMBLE, r"\begin{document}", r"\maketitle", provenance(run), section_question(run), section_data(run),
            section_methods(run), section_results(run), section_discussion(run), DEVELOPMENT, REPRODUCE, r"\end{document}"]
    tex_path = os.path.join(out, "pose_estimation_report.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(body))
    if not shutil.which("pdflatex"):
        print(f"pdflatex not found: wrote {common.rel(tex_path)} only")
        return
    for _ in range(2):
        proc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", os.path.basename(tex_path)],
                              cwd=out, capture_output=True, text=True)
        if proc.returncode:
            sys.stdout.write(proc.stdout[-3000:])
            raise SystemExit(f"pdflatex failed on {tex_path}")
    print(f"-> {common.rel(tex_path[:-4] + '.pdf')}")


if __name__ == "__main__":
    main()

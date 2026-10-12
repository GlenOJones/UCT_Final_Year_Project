"""Show, on real frames, which tags each channel found and where the channels overlap.

Reads the corners files run_study.py saved (no detection is run here). Every board tag any of the
chosen channels found is outlined, with one dot per channel inside it: filled if that channel found the
tag, hollow if it did not. Solid outline: every channel found it. Purple outline: only some did, which is
where the channels differ. Board tags no channel found are dashed, placed by the board pose fitted to the
tags that were found (solvePnP with the run's own calibration).

By default the frames shown are the ones where the channels disagree most.

Use (from the project root):
    src/venv/bin/python src/experiments/channels/plot_channel_overlap.py
    (reproduce.py channels makes the report's version, with the arguments in config.json "figures")
    ... --images right/Water_C5_frames/c5_Right_80.png left/Water_C1_frames/c1_Left_100.png
    ... --images left/Water_L3_Frames/l3_Left_130.png --width 3.4 --name fig_channel_overlap_L3 --legend-right
Writes fig_channel_overlap.{pdf,png} to the study's out_dir.
"""
import argparse
import json
import os
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_study  # noqa: E402
from analyse_study import NAMES, STYLE  # noqa: E402
from run_study import Calibration  # noqa: E402

PARTIAL = STYLE["merge"]["color"]


def board_tags(corners_file, camera, image):
    """{tag id: 4x2 corners} of the board tags in one image (ids found twice are left out)."""
    det = corners_file["cameras"][camera][image]
    ids = [i for i in det["ids"] if 0 <= i < Calibration.N_TAGS]
    return {i: np.array(c) for i, c in zip(det["ids"], det["corners"]) if i in ids and ids.count(i) == 1}


def missed_tags(found, yaml_path, camera):
    """{tag id: 4x2 projected corners} for board tags no channel found, from a pose fitted to the found ones."""
    if len(found) < 4:
        return {}
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    K = fs.getNode(f"{camera}_camera").getNode("camera_matrix").mat()
    D = fs.getNode(f"{camera}_camera").getNode("distortion_coefficients").mat()
    board = Calibration.build_board()
    obj = board.getObjPoints()
    ok, rvec, tvec = cv2.solvePnP(np.concatenate([obj[i] for i in found]),
                                  np.concatenate([found[i] for i in found]).astype(np.float32), K, D)
    return {i: cv2.projectPoints(obj[i], rvec, tvec, K, D)[0].reshape(4, 2)
            for i in range(Calibration.N_TAGS) if i not in found}


def most_disagreeing(files, channels, n):
    """The n images (as camera/name) with the most tags not found by every channel."""
    scored = []
    ref = files[channels[0]]["cameras"]
    for camera in ref:
        for image in ref[camera]:
            sets = [set(board_tags(files[c], camera, image)) for c in channels]
            scored.append((len(set.union(*sets)) - len(set.intersection(*sets)), f"{camera}/{image}"))
    return [name for _, name in sorted(scored, reverse=True)[:n]]


def draw(ax, study, files, channels, key, scale=1.0, title=True):
    """One frame; scale shrinks lines and text with the figure (1 = the 6.2 in default width).
    Returns the frame's description lines; they become the axis title unless title=False."""
    camera, image = key.split("/", 1)
    medium = study["recall_medium"]
    per = {c: board_tags(files[c], camera, image) for c in channels}
    union = {}
    for c in channels:                      # corners of the first channel that found the tag
        for i, x in per[c].items():
            union.setdefault(i, x)
    yaml_path = run_study.yaml_path(study, medium, channels[0])
    missed = missed_tags(union, yaml_path, camera)

    frame_dir = study["media"][medium]
    sub, name = image.split("/", 1)
    bgr = cv2.imread(os.path.join(frame_dir, sub, camera, name), cv2.IMREAD_COLOR)
    ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), alpha=0.75)

    pts = np.concatenate([*union.values(), *missed.values()])
    inside = (pts[:, 0] > -50) & (pts[:, 0] < bgr.shape[1] + 50) & (pts[:, 1] > -50) & (pts[:, 1] < bgr.shape[0] + 50)
    x0, y0 = np.maximum(pts[inside].min(0) - 40, 0)
    x1, y1 = np.minimum(pts[inside].max(0) + 40, bgr.shape[1::-1])
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)

    for i, quad in missed.items():
        ax.add_patch(Polygon(quad, closed=True, fill=False, edgecolor="#ffffff", linestyle="--", linewidth=0.8 * scale))
    counts = {"all": 0, "some": 0, "added": 0}
    for i, quad in union.items():
        found_by = [c for c in channels if i in per[c]]
        every = len(found_by) == len(channels)
        counts["all" if every else "some"] += 1
        counts["added"] += channels[0] not in found_by
        ax.add_patch(Polygon(quad, closed=True, fill=False, linewidth=(1.6 if every else 2.2) * scale,
                             edgecolor="#111111" if every else PARTIAL))
        side = np.mean(np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1))
        centre, r = quad.mean(0), max(side * 0.13, 2.5)
        for k, c in enumerate(channels):
            ax.add_patch(Circle(centre + [(k - (len(channels) - 1) / 2) * 2.6 * r, 0], r,
                                facecolor=STYLE[c]["color"] if c in found_by else "white",
                                edgecolor=STYLE[c]["color"], linewidth=1.0 * scale))
    none = len([q for q in missed.values() if x0 <= q[:, 0].mean() <= x1 and y0 <= q[:, 1].mean() <= y1])
    lines = [f"{camera} / {image}", f"{counts['all']} found by all, {counts['some']} by only some",
             f"({counts['added']} missed by {NAMES[channels[0]].split(' ')[0]}: what merging adds)", f"{none} by none"]
    if title:
        ax.set_title(f"{lines[0]}\n{lines[1]}{chr(10) if scale < 0.8 else ' '}{lines[2]}, {lines[3]}",
                     loc="left", fontsize=max(8 * scale, 5.5), wrap=True)
    ax.axis("off")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    run_study.add_study_argument(ap)
    ap.add_argument("--width", type=float, default=6.2, help="figure width per frame, inches (default 6.2)")
    ap.add_argument("--name", default="fig_channel_overlap", help="output file name, without extension")
    ap.add_argument("--legend-right", action="store_true",
                    help="put the description and the key in a column to the right of the image(s)")
    ap.add_argument("--channels", nargs="+", default=["gray", "g", "r"], help="channels to compare (default gray g r)")
    ap.add_argument("--images", nargs="+", help="camera/subfolder/file.png to show (default: the two where the "
                                                "channels disagree most)")
    args = ap.parse_args(argv)
    study = run_study.load_study(args.study)
    medium = study["recall_medium"]
    files = {c: json.load(open(run_study.corners_path(study, medium, c), encoding="utf-8")) for c in args.channels}
    keys = args.images or most_disagreeing(files, args.channels, 2)

    scale = args.width / 6.2
    if args.legend_right:
        side = 2.6 * max(scale, 0.75)                     # inches for the description + key column
        fig = plt.figure(figsize=(args.width * len(keys) + side, 4.6 * scale + 0.3), constrained_layout=True)
        grid = fig.add_gridspec(1, len(keys) + 1, width_ratios=[args.width] * len(keys) + [side])
        axes = [fig.add_subplot(grid[0, k]) for k in range(len(keys))]
        info = [draw(ax, study, files, args.channels, key, scale, title=False) for ax, key in zip(axes, keys)]
    else:
        fig, axes = plt.subplots(1, len(keys), figsize=(args.width * len(keys), 4.6 * scale + 0.9), constrained_layout=True)
        for ax, key in zip(np.atleast_1d(axes), keys):
            draw(ax, study, files, args.channels, key, scale)
    handles = [Line2D([], [], marker="o", linestyle="", markerfacecolor=STYLE[c]["color"],
                      markeredgecolor=STYLE[c]["color"], markersize=6 * scale,
                      label=f"found by {NAMES[c]}") for c in args.channels]
    handles += [Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="#555555", markersize=6 * scale,
                       label="hollow: not found by that channel"),
                Line2D([], [], color="#111111", linewidth=1.6 * scale, label="found by every channel"),
                Line2D([], [], color=PARTIAL, linewidth=2.2 * scale, label="found by only some channels"),
                Line2D([], [], color="#888888", linestyle="--", linewidth=0.8, label="on the board, found by none")]
    heading = f"{medium}, preset {study['calibration']['preset']}:\nwhich channel found each tag"
    if args.legend_right:
        side_ax = fig.add_subplot(grid[0, -1])
        side_ax.axis("off")
        text = heading + "\n\n" + "\n\n".join("\n".join(lines) for lines in info)
        side_ax.text(0, 1, text, va="top", ha="left", fontsize=max(8 * scale, 6), transform=side_ax.transAxes)
        side_ax.legend(handles=handles, loc="lower left", fontsize=max(8 * scale, 6), frameon=False,
                       handletextpad=0.4, borderaxespad=0)
    else:
        small = len(keys) * args.width < 5
        fig.legend(handles=handles, loc="outside lower center", ncol=2 if small else 3, fontsize=max(8 * scale, 5.5),
                   frameon=False, handletextpad=0.4, columnspacing=1.0)
        fig.suptitle(heading.replace("\n", " "), x=0.01, ha="left", fontsize=max(10 * scale, 6.5))
    out = os.path.join(study["out_dir"], args.name)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    print(f"wrote {os.path.relpath(out, run_study.PROJECT_ROOT)}.{{pdf,png}}: {', '.join(keys)}")


if __name__ == "__main__":
    main()

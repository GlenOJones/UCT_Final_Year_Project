"""How well the AprilGrid was detected, from the detection section Calibration.py writes.

Reads one or more calibration YAMLs and, for each, draws:
  - tags found in every image through the recording, left and right camera, with the --min-tags
    threshold below which an image is not used;
  - the distribution of tags found per image;
  - the board as its 10 x 7 grid of tags, each coloured by the share of images it was found in.
    A pattern here (an edge, a corner, a band) points at a cause: the image periphery, glare,
    caustics, or the board leaving the frame.

    src/venv/bin/python src/01_Calibration/plot_detection.py results/30Sep/calibration/c1_wide_stereo.yaml

Writes <yaml without _stereo.yaml>_detection.png next to each YAML.
"""
import argparse
import os
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_paths import rel  # noqa: E402

CAMERAS = ("left", "right")
CAMERA_COLOURS = {"left": "#2a78d6", "right": "#e8743b"}
RATE_RAMP = ["#e9e8e4", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, MUTED = "#fcfcfb", "#2b2b2a", "#8a8a86"
TAGS_X, TAGS_Y = 10, 7


def read_detection(path):
    """{camera: (image names, tags found per image, images per tag id)} and the --min-tags used."""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    node = fs.getNode("detection")
    if node.isNone():
        raise SystemExit(f"{rel(path)} has no detection section: re-run it with the current Calibration.py")
    min_tags = int(fs.getNode("calibration_board").getNode("min_tags_per_image").real())
    out = {}
    for camera in CAMERAS:
        c = node.getNode(camera)
        seq = lambda key: [c.getNode(key).at(i) for i in range(c.getNode(key).size())]
        names = [n.string() for n in seq("images")]
        counts = np.array([int(n.real()) for n in seq("tags_found")])
        per_id = np.array([int(n.real()) for n in seq("images_per_tag_id")])
        out[camera] = (names, counts, per_id)
    fs.release()
    return out, min_tags


def plot(path):
    detection, min_tags = read_detection(path)
    name = os.path.basename(path).replace("_stereo.yaml", "")
    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": MUTED})
    fig = plt.figure(figsize=(15, 8.5), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.38, wspace=0.25)
    summary = []
    for camera in CAMERAS:
        _, counts, _ = detection[camera]
        summary.append(f"{camera}: median {np.median(counts):.0f}/70 tags, "
                       f"{np.mean(counts >= min_tags):.0%} of {len(counts)} images usable")
    fig.suptitle(f"{name}: AprilGrid detection.  " + ";  ".join(summary), x=0.01, ha="left", fontsize=11)

    # Tags found through the recording.
    ax = fig.add_subplot(grid[0, :], facecolor=SURFACE)
    for camera in CAMERAS:
        _, counts, _ = detection[camera]
        ax.plot(np.arange(len(counts)), counts, color=CAMERA_COLOURS[camera], lw=1.2, label=camera)
    ax.axhline(min_tags, color=MUTED, lw=0.8, ls="--")
    ax.text(0, min_tags + 1, f"used if ≥ {min_tags} tags", color=MUTED, fontsize=8)
    ax.set_xlim(0, max(len(detection[c][1]) for c in CAMERAS) - 1)
    ax.set_ylim(0, 72)
    ax.set_xlabel("image (in recording order)")
    ax.set_ylabel("tags found (of 70)")
    ax.set_title("Tags found per image", loc="left")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax.grid(color="#e4e3df", lw=0.6)

    # Distribution.
    ax = fig.add_subplot(grid[1, 0], facecolor=SURFACE)
    bins = np.arange(0, 72, 2)
    for camera in CAMERAS:
        _, counts, _ = detection[camera]
        ax.hist(counts, bins=bins, histtype="step", lw=2, color=CAMERA_COLOURS[camera], label=camera)
    ax.axvline(min_tags, color=MUTED, lw=0.8, ls="--")
    ax.set_xlabel("tags found per image")
    ax.set_ylabel("images")
    ax.set_title("Distribution", loc="left")
    ax.legend(frameon=False)

    # Per tag id, laid out as on the board: id 0 bottom-left, ids run right then up.
    cmap = LinearSegmentedColormap.from_list("rate", RATE_RAMP)
    for k, camera in enumerate(CAMERAS):
        _, counts, per_id = detection[camera]
        rate = per_id.reshape(TAGS_Y, TAGS_X) / max(len(counts), 1)
        ax = fig.add_subplot(grid[1, 1 + k], facecolor=SURFACE)
        image = ax.imshow(rate, origin="lower", cmap=cmap, vmin=0, vmax=1)
        for (y, x), r in np.ndenumerate(rate):
            ax.text(x, y, f"{100 * r:.0f}", ha="center", va="center", fontsize=7,
                    color=SURFACE if r > 0.55 else INK)
        ax.set_xticks(range(TAGS_X))
        ax.set_yticks(range(TAGS_Y))
        ax.set_title(f"{camera}: % of images each tag was found in\n(board layout, id 0 bottom-left)",
                     loc="left", fontsize=9)
        fig.colorbar(image, ax=ax, shrink=0.75, format=lambda v, _: f"{100 * v:.0f}%")

    out = path.replace("_stereo.yaml", "_detection.png")
    fig.savefig(out, dpi=140, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {rel(out)}")


def main():
    ap = argparse.ArgumentParser(description="Plot AprilGrid detection from Calibration.py output.")
    ap.add_argument("yamls", nargs="+", help="calibration YAMLs written by Calibration.py")
    for path in ap.parse_args().yamls:
        plot(path)


if __name__ == "__main__":
    main()

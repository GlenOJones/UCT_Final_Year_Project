"""One air frame beside one water frame, with the tags the study's detector found outlined.

The figure shows only the frames, unlabelled, in the order the config lists them (Air left, Water right);
the description is printed for the report's caption: the board's distance from the camera (solvePnP on the saved detections, with that medium's own
calibration), the camera's horizontal field of view from the same calibration, how many board tags were
found and their mean side in pixels. Nothing is detected here: the tags are the ones the study's run saved
in its corners file.

The frames, camera, channel and title come from config.json "figures" -> "air_vs_water".

Use (from the project root, after run_study.py; reproduce.py channels runs it):
    src/venv/bin/python src/experiments/channels/plot_air_vs_water.py
Writes <out_dir>/<name>.{pdf,png} for every figure listed.
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
from matplotlib.patches import Polygon  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_study  # noqa: E402
from run_study import PROJECT_ROOT, Calibration as C  # noqa: E402

FOUND_COLOUR = "#18c418"
# Board centre in board coordinates (mm): the distance quoted is to this point.
BOARD_CENTRE = np.array([(9 * C.PITCH + C.TAG_SIZE) / 2, (6 * C.PITCH + C.TAG_SIZE) / 2, 0.0])


def panel(ax, study, medium, image, camera, channel):
    """Draw one frame with its found tags; returns the numbers for the caption."""
    det = json.load(open(run_study.corners_path(study, medium, channel)))["cameras"][camera][image]
    fs = cv2.FileStorage(run_study.yaml_path(study, medium, channel), cv2.FILE_STORAGE_READ)
    node = fs.getNode(f"{camera}_camera")
    K, D = node.getNode("camera_matrix").mat(), node.getNode("distortion_coefficients").mat()
    hfov = node.getNode("field_of_view_horizontal_deg").real()
    obj = C.build_board().getObjPoints()
    tags = [(i, np.float32(c)) for i, c in zip(det["ids"], det["corners"]) if 0 <= i < C.N_TAGS]
    _, rvec, tvec = cv2.solvePnP(np.concatenate([obj[i] for i, _ in tags]), np.concatenate([c for _, c in tags]), K, D)
    distance_cm = (cv2.Rodrigues(rvec)[0] @ BOARD_CENTRE + tvec.ravel())[2] / 10

    folder, name = image.split("/")
    path = os.path.join(study["media"][medium], folder, camera, name)
    ax.imshow(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))
    sides = [np.mean(np.linalg.norm(c - np.roll(c, 1, axis=0), axis=1)) for _, c in tags]
    for _, c in tags:
        ax.add_patch(Polygon(c, closed=True, fill=False, edgecolor=FOUND_COLOUR, linewidth=1.0))
    ax.axis("off")
    return {"distance_cm": float(distance_cm), "hfov_deg": hfov, "tags": len(tags), "mean_side_px": float(np.mean(sides))}


def figure(study, spec):
    media = list(spec["images"])
    fig, axes = plt.subplots(1, len(media), figsize=(5 * len(media), 4.4), constrained_layout=True)
    numbers = {m: panel(ax, study, m, spec["images"][m], spec["camera"], spec["channel"])
               for ax, m in zip(np.atleast_1d(axes), media)}
    out = os.path.join(study["out_dir"], spec["name"])
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {os.path.relpath(out, PROJECT_ROOT)}.{{pdf,png}}; caption text:\n  {spec['title']}")
    for m, n in numbers.items():
        print(f"  {m}: board {n['distance_cm']:.0f} cm from camera, horizontal FOV {n['hfov_deg']:.0f}°, "
              f"{n['tags']} of {C.N_TAGS} tags found, mean tag side {n['mean_side_px']:.0f} px")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    run_study.add_study_argument(ap)
    study = run_study.load_study(ap.parse_args(argv).study)
    for spec in study.get("figures", {}).get("air_vs_water", []):
        figure(study, spec)


if __name__ == "__main__":
    main()

"""Cropped examples of how underwater images differ from in-air ones, beyond magnification."""
import argparse
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
from run_study import PROJECT_ROOT, Calibration as C  # noqa: E402

DATA = os.path.join(PROJECT_ROOT, "data", "Calibration", "FinalComparison")
CROP_HEIGHT = 420            # px, every crop is resized to this height for display

# (medium, scan folder, file) pairs at matched measured distance, right camera
CLOSE = (("Air", "Air_C1_Frames", "C1_Right_40.png"), ("Water", "Water_C1_frames", "c1_Right_40.png"))      # 21.8 / 19.7 cm
FAR = (("Air", "Air_C1_Frames", "C1_Right_140.png"), ("Water", "Water_C1_frames", "c1_Right_140.png"))      # 121.0 / 120.6 cm
# Board tag ids of the block to crop (rows run 0..6 bottom to top, columns 0..9 left to right).
CLOSE_BLOCK = (25, 26, 27, 35, 36, 37)    # in frame in both close-ups
FAR_BLOCK = (23, 24, 25, 26, 33, 34, 35, 36, 43, 44, 45, 46)
SINGLE_TAG = 26              # in frame in all four matched images (incl. the tight water close-up)
SINGLE_MARGIN = 0.45         # x tag side around the tag: covers the white gap (0.3) and the corner squares
# Single-image examples: (title, image path under DATA, crop x0, y0, x1, y1 in px)
EXAMPLES = (
    ("Bubbles on the board (water, C1, 20 cm)", "Water/Water_C1_frames/right/c1_Right_40.png", (300, 100, 900, 520)),
    ("Glints on the laminated sea target (water)", "WaterComparison/left/reference__Left_50.png", (500, 380, 1220, 880)),
    ("Objects between camera and board (water, C5)", "Water/Water_C5_frames/right/c5_Right_40.png", (200, 450, 1500, 1200)),
)


def pose(study, medium, folder, name):
    """(K, D, rvec, tvec, distance cm) of the board in one right-camera image, from the study's gray run."""
    corners = json.load(open(run_study.corners_path(study, medium, "gray")))["cameras"]["right"]
    d = corners[f"{folder}/{name}"]
    fs = cv2.FileStorage(run_study.yaml_path(study, medium, "gray"), cv2.FILE_STORAGE_READ)
    K, D = fs.getNode("right_camera").getNode("camera_matrix").mat(), fs.getNode("right_camera").getNode("distortion_coefficients").mat()
    obj = C.build_board().getObjPoints()
    tags = [(i, np.float32(c)) for i, c in zip(d["ids"], d["corners"]) if 0 <= i < C.N_TAGS and d["ids"].count(i) == 1]
    _, rvec, tvec, _ = cv2.solvePnPRansac(np.concatenate([obj[i] for i, _ in tags]), np.concatenate([c for _, c in tags]),
                                          K, D, flags=cv2.SOLVEPNP_IPPE)
    centre = np.array([(9 * C.PITCH + C.TAG_SIZE) / 2, (6 * C.PITCH + C.TAG_SIZE) / 2, 0.0])
    distance = float((cv2.Rodrigues(rvec)[0] @ centre + tvec.ravel())[2] / 10)
    return K, D, rvec, tvec, distance


def block_crop(study, medium, folder, name, block, margin_tags=0.25):
    """The image region covering these board tags plus margin_tags x a tag's width, and the distance."""
    K, D, rvec, tvec, distance = pose(study, medium, folder, name)
    obj = C.build_board().getObjPoints()
    pts = np.concatenate([cv2.projectPoints(obj[i], rvec, tvec, K, D)[0].reshape(4, 2) for i in block])
    margin = margin_tags * (pts[:, 0].max() - pts[:, 0].min()) / (len({i % 10 for i in block}))
    image = cv2.imread(os.path.join(DATA, medium, folder, "right", name))
    h, w = image.shape[:2]
    if pts.min() < 0 or pts[:, 0].max() > w or pts[:, 1].max() > h:
        raise SystemExit(f"{medium} {name}: tag block {block} is not fully in frame")
    x0, y0 = np.maximum(np.floor(pts.min(0) - margin), 0).astype(int)        # the margin may be clipped by the
    x1, y1 = np.minimum(np.ceil(pts.max(0) + margin), [w, h]).astype(int)   # frame edge; the tags may not
    return image[y0:y1, x0:x1], distance, (x1 - x0)


def show(img):
    """BGR crop resized to CROP_HEIGHT, as RGB for matplotlib."""
    s = CROP_HEIGHT / img.shape[0]
    return cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC),
                        cv2.COLOR_BGR2RGB)


def single_tag_figure(study, out):
    """Close-up and furthest matched pairs, one tag and its border each, enlarged with nearest-neighbour scaling so the
    real pixels are visible."""
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.4), constrained_layout=True)
    caption = [f"Tag {SINGLE_TAG} and its border, right camera, angle C1; same measured distance in air and water. "
               f"Top row: close-up pair, bottom row: furthest pair; enlarged with nearest-neighbour scaling."]
    for row, (pair, label) in enumerate(((CLOSE, "close-up"), (FAR, "furthest"))):
        for col, (medium, folder, name) in enumerate(pair):
            img, dist, width = block_crop(study, medium, folder, name, (SINGLE_TAG,), SINGLE_MARGIN)
            s = CROP_HEIGHT / img.shape[0]
            ax = axes[row, col]
            ax.imshow(cv2.cvtColor(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB))
            caption.append(f"{medium}, {label} ({name}): board {dist:.1f} cm, crop {img.shape[1]} x {img.shape[0]} px "
                           f"in the original")
            ax.axis("off")
            cv2.imwrite(os.path.join(out, f"single_tag_{label}_{medium.lower()}_{name}"), img)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_single_tag_close_far.{ext}"), dpi=200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"wrote {os.path.relpath(out, PROJECT_ROOT)}/fig_single_tag_close_far.{{pdf,png}}; caption text:\n  "
          + "\n  ".join(caption))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    run_study.add_study_argument(ap)
    study = run_study.load_study(ap.parse_args(argv).study)
    out = os.path.join(study["out_dir"], "air_vs_water_crops")
    os.makedirs(out, exist_ok=True)
    single_tag_figure(study, out)
    fig = plt.figure(figsize=(12, 10.5), constrained_layout=True)
    grid = fig.add_gridspec(3, 6)
    for row, (pair, block, label) in enumerate(((CLOSE, CLOSE_BLOCK, "close-up"), (FAR, FAR_BLOCK, "furthest"))):
        for col, (medium, folder, name) in enumerate(pair):
            img, dist, width = block_crop(study, medium, folder, name, block)
            ax = fig.add_subplot(grid[row, 3 * col:3 * col + 3])
            ax.imshow(show(img))
            ax.set_title(f"{medium}, {label}: board {dist:.1f} cm from camera (label {name.split('_')[-1][:-4]} cm)\n"
                         f"same {len(block)} tags in both; crop {width} px wide in the original", loc="left", fontsize=8.5)
            ax.axis("off")
            cv2.imwrite(os.path.join(out, f"{label}_{medium.lower()}_{name}"), img)
    for k, (title, path, (x0, y0, x1, y1)) in enumerate(EXAMPLES):
        img = cv2.imread(os.path.join(DATA, path))[y0:y1, x0:x1]
        ax = fig.add_subplot(grid[2, 2 * k:2 * k + 2])
        ax.imshow(show(img))
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.axis("off")
        cv2.imwrite(os.path.join(out, f"example_{os.path.basename(path)}"), img)
    fig.suptitle("Air vs water beyond magnification: right camera, matched pairs at the same measured distance, "
                 "cropped to the same tags and shown at the same size", x=0.01, ha="left", fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_air_vs_water_crops.{ext}"), dpi=200)
    print(f"wrote {os.path.relpath(out, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

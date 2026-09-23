"""
Image filtering to help AprilTag detection on hard (blurred, small, low-contrast) calibration frames.

Filters are applied for DETECTION ONLY. They shift edge positions, so corners found on a filtered
image must be refined on the original grayscale image (cv2.cornerSubPix), never on the filtered one.
detect_multipass() returns corners in original-image pixels for that reason.

Measured on the 18 Sep Rigframes set (208 images, usable = >=10 tags; 90 with no filtering):
    unsharp 104, upscale2x 102, flatfield 100, clahe 99; gamma 86 and denoising 57 made it worse.
More detections is not the same as a better calibration: check per-view reprojection error afterwards.

Use from Calibration.py:
    from filtering import detect_multipass
    corners, ids, rejected, used = detect_multipass(detector, gray, min_tags=MIN_TAGS)

Run this file to compare the filters on a folder of frames (from the project root):
    python src/01_Calibration/filtering.py data/Calibration/18_Sep/Rigframes
"""
import glob
import os
import sys

import cv2
import numpy as np


# ====== FILTERS (uint8 grayscale in, uint8 grayscale out, same size) ======
def unsharp(gray, sigma=3, amount=1.0):
    """Sharpen: original + amount * (original - blurred). Helps blurred tags."""
    return cv2.addWeighted(gray, 1 + amount, cv2.GaussianBlur(gray, (0, 0), sigma), -amount, 0)


def clahe(gray, clip=2.0, tiles=8):
    """Local contrast equalisation."""
    return cv2.createCLAHE(clip, (tiles, tiles)).apply(gray)


def flatfield(gray, sigma=40):
    """Divide by a heavy blur to remove uneven lighting / backscatter gradients."""
    background = cv2.GaussianBlur(gray, (0, 0), sigma)
    return cv2.normalize(cv2.divide(gray, background, scale=128), None, 0, 255, cv2.NORM_MINMAX)


FILTERS = {"unsharp": unsharp, "flatfield": flatfield, "clahe": clahe}
UPSCALE = 2   # factor for the "upscale" pass, which helps small tags


# ====== DETECTION ======
def detect(detector, gray, name):
    """detectMarkers on gray after the named pass ("none", "upscale" or a FILTERS key).
    Returns (corners, ids, rejected) with corners in ORIGINAL image pixels."""
    if name == "upscale":
        big = cv2.resize(gray, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
        corners, ids, rejected = detector.detectMarkers(big)
        return [c / UPSCALE for c in corners], ids, [r / UPSCALE for r in rejected]
    return detector.detectMarkers(gray if name == "none" else FILTERS[name](gray))


def detect_multipass(detector, gray, min_tags=10, passes=("none", "unsharp", "upscale", "flatfield", "clahe")):
    """Try each pass in order and stop at the first with >= min_tags tags; otherwise keep the pass
    with the most tags. Returns (corners, ids, rejected, pass_name). Corners are unrefined."""
    best = None
    for name in passes:
        corners, ids, rejected = detect(detector, gray, name)
        n = 0 if ids is None else len(ids)
        if best is None or n > best[0]:
            best = (n, corners, ids, rejected, name)
        if n >= min_tags:
            break
    return best[1:]


# ====== COMPARE FILTERS ON A FOLDER ======
if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "data/Calibration/18_Sep/Rigframes"
    files = sorted(glob.glob(os.path.join(folder, "**", "*.png"), recursive=True))
    if not files:
        sys.exit(f"no PNGs under {folder}")

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    params.markerBorderBits = 2
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
    params.adaptiveThreshWinSizeMin, params.adaptiveThreshWinSizeMax, params.adaptiveThreshWinSizeStep = 3, 53, 5
    params.errorCorrectionRate = 1.0
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    passes = ("none", "upscale") + tuple(FILTERS)
    counts = {p: [] for p in passes + ("multipass",)}
    for path in files:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        for p in passes:
            _, ids, _ = detect(detector, gray, p)
            counts[p].append(0 if ids is None else len(ids))
        _, ids, _, _ = detect_multipass(detector, gray)
        counts["multipass"].append(0 if ids is None else len(ids))

    print(f"{len(files)} images under {folder}; usable = 10 or more tags")
    for p, n in counts.items():
        n = np.array(n)
        print(f"  {p:10s} usable {np.sum(n >= 10):4d}   total tags {n.sum():6d}")

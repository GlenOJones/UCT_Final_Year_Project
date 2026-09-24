"""Check a folder of frames for blur, by the variance of the Laplacian.

The Laplacian is a second-derivative filter, so it responds to whatever changes quickly in the
image. A sharp frame is full of hard edges and its Laplacian swings widely, giving a large variance;
blur smooths those edges away and the variance collapses. Frames scoring below the threshold are
called blurry.

Noise is reported alongside, because it works against the score: sensor noise is high-frequency too,
so it also makes the Laplacian swing and can lift the score of a frame that is genuinely soft. The
estimate is Immerkaer's, from a 3x3 second-derivative kernel, measured over the flat parts of the
image only - the strongest 10% of Sobel gradients are masked out so that the board's own edges are
not counted as noise. Read it as a sanity check: a frame with high noise and a middling score is
softer than its score suggests.

The score is not absolute, so neither is the threshold. It measures how much fine detail is in the
frame, and DISTANCE changes that as much as focus does: on the 18 Sep set, C90_Left_50 (board close,
filling the frame) scores 41.8 while C90_Left_140 (board small and distant) scores 7.8, at the same
focus on the same camera. Across all of Rigframes the scores run 7 to 88 with a median of 12.7, so
the default of 100 - a number that suits a webcam photo - calls every frame blurry. Run this once on
a set you know is good, look at the spread, and set --threshold from that.

The input images are only ever read.

Usage (sub-folders are included, so a whole recording works too):
    src/venv/bin/python src/preproccessing/blur_check.py data/Calibration/18_Sep/Rigframes/90
    src/venv/bin/python src/preproccessing/blur_check.py data/Calibration/18_Sep/Rigframes --threshold 30

Tests: src/venv/bin/python -m unittest discover -s src/preproccessing
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_THRESHOLD = 100.0   # variance of the Laplacian below which a frame is called blurry
EDGE_PERCENTILE = 90        # gradients above this are real edges, excluded from the noise estimate
IMAGE_TYPES = (".png", ".jpg", ".jpeg")
MAX_NAMES_LISTED = 12       # a whole recording flags hundreds; listing them all scrolls the table away

# Immerkaer's 3x3 second-derivative kernel. Convolving with it cancels any locally linear ramp in
# intensity, so what is left over a flat patch of image is the noise on its own.
IMMERKAER_KERNEL = np.array([
    [1, -2, 1],
    [-2, 4, -2],
    [1, -2, 1],
], dtype=np.float64)


def blur_score(gray):
    """Variance of the Laplacian: large on a sharp frame, collapsing towards 0 as it blurs."""
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def noise_sigma(gray):
    """Immerkaer's estimate of the noise standard deviation, in grey levels.

    Measured only where the image is flat. Without that the board's own edges would be counted as
    noise, and the answer would say more about the scene than about the sensor.
    """
    image = gray.astype(np.float64)

    # Step 1: find the edges, using the first-order (Sobel) gradient.
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    gradient_magnitude = np.hypot(gradient_x, gradient_y)

    # Step 2: keep the flat pixels, which is everything below the 90th percentile of that gradient.
    # The comparison is "<=" rather than "<": on an image with large perfectly flat areas the
    # percentile is itself 0, and "< 0" would select no pixels at all and leave nothing to average.
    edge_threshold = np.percentile(gradient_magnitude, EDGE_PERCENTILE)
    is_flat = gradient_magnitude <= edge_threshold

    # Step 3: average the second-derivative response over those flat pixels. The constants are
    # Immerkaer's: 6 is the kernel's norm and sqrt(pi/2) converts a mean absolute value to a
    # standard deviation, assuming the noise is Gaussian.
    response = np.abs(cv2.filter2D(image, -1, IMMERKAER_KERNEL))
    mean_response = response[is_flat].mean()
    return np.sqrt(np.pi / 2) * mean_response / 6


def find_images(folder):
    """Every image file under folder, including sub-folders, sorted by name."""
    images = []
    for path in sorted(Path(folder).rglob("*")):
        if path.suffix.lower() in IMAGE_TYPES:
            images.append(path)
    if not images:
        raise SystemExit(f"no images found in {folder}")
    return images


def measure(folder):
    """Score every image under folder. Returns a list of {name, score, noise}. Read only."""
    results = []
    for path in find_images(folder):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise SystemExit(f"{path}: could not be read as an image")
        results.append({
            "name": str(path.relative_to(folder)),
            "score": blur_score(gray),
            "noise": noise_sigma(gray),
        })
    return results


def print_table(results, threshold):
    """One row per image, widest name setting the first column."""
    name_width = max(12, max(len(r["name"]) for r in results) + 2)
    print("image".ljust(name_width) + "score".rjust(10) + "noise".rjust(9) + "   verdict")
    print("-" * (name_width + 28))
    for result in results:
        verdict = "BLURRY" if result["score"] < threshold else "sharp"
        row = result["name"].ljust(name_width)
        row += f"{result['score']:.1f}".rjust(10)
        row += f"{result['noise']:.2f}".rjust(9)
        print(row + "   " + verdict)


def print_summary(results, threshold):
    """The spread of the scores, so the threshold can be judged, then the counts."""
    scores = [r["score"] for r in results]
    noises = [r["noise"] for r in results]
    blurry = [r["name"] for r in results if r["score"] < threshold]

    print(f"\nscores: lowest {min(scores):.1f}, median {np.median(scores):.1f}, "
          f"highest {max(scores):.1f}   median noise {np.median(noises):.2f}")
    print(f"{len(blurry)} blurry, {len(results) - len(blurry)} sharp")
    if blurry:
        listed = " ".join(blurry[:MAX_NAMES_LISTED])
        if len(blurry) > MAX_NAMES_LISTED:
            listed += f" ... and {len(blurry) - MAX_NAMES_LISTED} more"
        print("blurry: " + listed)
    return blurry


def report(folder, results, threshold):
    """Print the table and the summary. Returns the names called blurry."""
    print(f"\n{len(results)} images in {folder}, blurry below a score of {threshold:g}")
    print_table(results, threshold)
    return print_summary(results, threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="folder of frames to check (sub-folders included; read only)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"blurry below this variance of the Laplacian "
                             f"(default {DEFAULT_THRESHOLD:g}, but see the notes above)")
    args = parser.parse_args()

    results = measure(args.folder)
    report(args.folder, results, args.threshold)


if __name__ == "__main__":
    main()

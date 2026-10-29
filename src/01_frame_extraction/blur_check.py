"""Check a folder of frames for blur, by the variance of the Laplacian."""
import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_THRESHOLD = 100.0   # variance of the Laplacian below which a frame is called blurry
EDGE_PERCENTILE = 90        # gradients above this are real edges, excluded from the noise estimate
IMAGE_TYPES = (".png", ".jpg", ".jpeg")
MAX_NAMES_LISTED = 12       # a whole recording flags hundreds; listing them all scrolls the table away

# Immerkaer's 3x3 second-derivative kernel.
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
    """Immerkaer's estimate of the noise standard deviation, in grey levels."""
    image = gray.astype(np.float64)

    # Step 1: find the edges, using the first-order (Sobel) gradient.
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    gradient_magnitude = np.hypot(gradient_x, gradient_y)

    # Step 2: keep the flat pixels, below the 90th percentile (<= so all-flat images still work).
    edge_threshold = np.percentile(gradient_magnitude, EDGE_PERCENTILE)
    is_flat = gradient_magnitude <= edge_threshold

    # Step 3: mean response over the flat pixels (Immerkaer's constants).
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
    """Score every image under folder."""
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
    """Print the table and the summary."""
    print(f"\n{len(results)} images in {folder}, blurry below a score of {threshold:g}")
    print_table(results, threshold)
    return print_summary(results, threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", help="folder of frames to check (sub-folders included; read only)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"blurry below this variance of the Laplacian "
                             f"(default {DEFAULT_THRESHOLD:g})")
    args = parser.parse_args()

    results = measure(args.folder)
    report(args.folder, results, args.threshold)


if __name__ == "__main__":
    main()

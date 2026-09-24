"""Check blur_check.py against synthetic images whose blur and noise are known exactly.

Real frames cannot test this: there is no ground truth for how blurred they are. So a sharp textured
image is built, blurred by a known amount, and given a known amount of noise, and the two functions
are asked to recover what was done to it.

Run (from any directory):
    src/venv/bin/python -m unittest discover -s src/preproccessing
    src/venv/bin/python src/preproccessing/test_blur_check.py
"""
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blur_check  # noqa: E402

SIZE = 480
SQUARE = 40      # checkerboard square in px: plenty of hard edges for the Laplacian to find


def checkerboard():
    """A sharp, high-contrast test pattern, mid-grey so that added noise does not clip at 0 or 255."""
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    return np.where((ys // SQUARE + xs // SQUARE) % 2 == 0, 60, 200).astype(np.uint8)


class BlurScoreTest(unittest.TestCase):
    def setUp(self):
        self.sharp = checkerboard()
        self.blurred = cv2.GaussianBlur(self.sharp, (0, 0), 3.0)

    def test_blur_lowers_the_score(self):
        self.assertGreater(blur_check.blur_score(self.sharp), blur_check.blur_score(self.blurred))

    def test_more_blur_lowers_it_further(self):
        """The score has to fall monotonically, or a threshold on it means nothing."""
        scores = [blur_check.blur_score(cv2.GaussianBlur(self.sharp, (0, 0), s))
                  for s in (1.0, 2.0, 4.0, 8.0)]
        self.assertEqual(scores, sorted(scores, reverse=True), scores)

    def test_a_threshold_between_them_separates_them(self):
        sharp, blurred = blur_check.blur_score(self.sharp), blur_check.blur_score(self.blurred)
        threshold = (sharp + blurred) / 2
        self.assertGreaterEqual(sharp, threshold)
        self.assertLess(blurred, threshold)


class NoiseSigmaTest(unittest.TestCase):
    def test_recovers_a_known_noise_level(self):
        rng = np.random.default_rng(0)
        for sigma in (2.0, 5.0, 10.0):
            noisy = np.clip(checkerboard() + rng.normal(0, sigma, (SIZE, SIZE)), 0, 255).astype(np.uint8)
            self.assertAlmostEqual(blur_check.noise_sigma(noisy), sigma, delta=0.15 * sigma,
                                   msg=f"sigma {sigma}")

    def test_a_clean_image_reads_near_zero(self):
        """The edge mask has to work: without it the checkerboard's own edges read as noise."""
        self.assertLess(blur_check.noise_sigma(checkerboard()), 1.0)


class FolderTest(unittest.TestCase):
    """The whole path: read a folder, score every image, print the verdicts."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name)
        self.sharp_names = [f"{i:04d}.png" for i in range(3)]
        for name in self.sharp_names:
            cv2.imwrite(str(self.folder / name), checkerboard())
        self.blurred_name = "0003.png"
        cv2.imwrite(str(self.folder / self.blurred_name),
                    cv2.GaussianBlur(checkerboard(), (0, 0), 3.0))

    def tearDown(self):
        self._tmp.cleanup()

    def test_only_the_blurred_frame_is_flagged(self):
        results = blur_check.measure(self.folder)
        self.assertEqual(len(results), 4)
        scores = {result["name"]: result["score"] for result in results}
        threshold = (scores[self.sharp_names[0]] + scores[self.blurred_name]) / 2
        self.assertEqual(blur_check.report(self.folder, results, threshold), [self.blurred_name])

    def test_inputs_are_not_modified(self):
        blur_check.measure(self.folder)
        on_disk = cv2.imread(str(self.folder / self.sharp_names[0]), cv2.IMREAD_GRAYSCALE)
        self.assertTrue(np.array_equal(on_disk, checkerboard()))


if __name__ == "__main__":
    unittest.main()

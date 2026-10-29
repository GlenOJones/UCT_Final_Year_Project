"""Zone splitting, corner-to-centre ratios, binned profiles, grid statistics and the paired noise test, checked on
synthetic residuals whose errors are known exactly.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibmethods import metrics, stats  # noqa: E402

IMAGE_SIZE = (1600, 1200)
PP = (800.0, 600.0)
R_MAX = 1000.0                       # centre of a 1600 x 1200 image to its corner
RADII = (0.33, 0.66)


def points_at(radii_norm, n_angles=36):
    """Points on circles of the given normalised radii about PP, evenly spaced in angle."""
    t = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    r = np.repeat(np.asarray(radii_norm, float) * R_MAX, n_angles)
    t = np.tile(t, len(radii_norm))
    return PP[0] + r * np.cos(t), PP[1] + r * np.sin(t)


def residuals_by_zone(u, v, size_per_zone):
    """Residuals with length size_per_zone[zone] and random direction, so the RMS per zone is exact."""
    zone = metrics.zone_of(metrics.normalised_radius(u, v, PP, IMAGE_SIZE), RADII)
    mag = np.asarray(size_per_zone, float)[zone]
    angle = np.random.default_rng(0).uniform(0, 2 * np.pi, u.size)
    return mag * np.cos(angle), mag * np.sin(angle)


class NormalisedRadius(unittest.TestCase):
    def test_centre_and_corners(self):
        r = metrics.normalised_radius([800, 0, 1600, 1600], [600, 0, 0, 1200], PP, IMAGE_SIZE)
        np.testing.assert_allclose(r, [0, 1, 1, 1])

    def test_off_centre_principal_point_uses_farthest_corner(self):
        # farthest corner from (700, 500) is (1600, 1200): sqrt(900^2 + 700^2)
        r = metrics.normalised_radius([1600], [1200], (700, 500), IMAGE_SIZE)
        np.testing.assert_allclose(r, [1.0])
        r = metrics.normalised_radius([0], [0], (700, 500), IMAGE_SIZE)
        np.testing.assert_allclose(r, [np.hypot(700, 500) / np.hypot(900, 700)])


class Zones(unittest.TestCase):
    def test_boundaries_belong_to_the_outer_zone(self):
        np.testing.assert_array_equal(metrics.zone_of([0, 0.329, 0.33, 0.659, 0.66, 1.0], RADII),
                                      [0, 0, 1, 1, 2, 2])

    def test_bad_radii_rejected(self):
        for radii in [(0.5, 0.3), (0, 0.5), (0.4, 0.4)]:
            with self.assertRaises(ValueError):
                metrics.zone_of([0.1], radii)

    def test_known_errors_per_zone(self):
        u, v = points_at([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        du, dv = residuals_by_zone(u, v, [0.2, 0.5, 1.2])
        z = metrics.zone_rms(du, dv, metrics.normalised_radius(u, v, PP, IMAGE_SIZE), RADII)
        self.assertAlmostEqual(z["centre"]["rms"], 0.2)
        self.assertAlmostEqual(z["mid"]["rms"], 0.5)
        self.assertAlmostEqual(z["outer"]["rms"], 1.2)
        self.assertAlmostEqual(z["outer_to_centre"], 6.0)
        self.assertAlmostEqual(z["mid_to_centre"], 2.5)
        self.assertEqual([z[k]["n"] for k in metrics.ZONES], [3 * 36, 3 * 36, 3 * 36])

    def test_uniform_error_gives_ratio_one(self):
        u, v = points_at(np.linspace(0.05, 0.95, 10))
        du, dv = residuals_by_zone(u, v, [0.4, 0.4, 0.4])
        z = metrics.zone_rms(du, dv, metrics.normalised_radius(u, v, PP, IMAGE_SIZE), RADII)
        self.assertAlmostEqual(z["outer_to_centre"], 1.0)
        self.assertAlmostEqual(z["mid_to_centre"], 1.0)

    def test_ratio_follows_the_radii_parameter(self):
        # errors grow linearly with r; moving the boundaries outwards must raise the outer zone's RMS
        u, v = points_at(np.linspace(0.05, 0.95, 19))
        r = metrics.normalised_radius(u, v, PP, IMAGE_SIZE)
        du, dv = r.copy(), np.zeros_like(r)
        narrow = metrics.zone_rms(du, dv, r, (0.3, 0.6))
        wide = metrics.zone_rms(du, dv, r, (0.3, 0.8))
        self.assertGreater(wide["outer"]["rms"], narrow["outer"]["rms"])
        self.assertAlmostEqual(narrow["centre"]["rms"], wide["centre"]["rms"])

    def test_empty_zone_is_nan_not_an_error(self):
        u, v = points_at([0.1, 0.2])                     # nothing outside the centre zone
        z = metrics.zone_rms(np.ones(u.size), np.zeros(u.size),
                             metrics.normalised_radius(u, v, PP, IMAGE_SIZE), RADII)
        self.assertEqual(z["outer"]["n"], 0)
        self.assertTrue(np.isnan(z["outer"]["rms"]))
        self.assertTrue(np.isnan(z["outer_to_centre"]))


class BinnedRms(unittest.TestCase):
    def test_known_profile(self):
        x = np.repeat([0.05, 0.15, 0.25], 10)
        du = np.repeat([1.0, 2.0, 3.0], 10)
        centres, value, n = metrics.binned_rms(x, du, np.zeros_like(du), [0, 0.1, 0.2, 0.3])
        np.testing.assert_allclose(centres, [0.05, 0.15, 0.25])
        np.testing.assert_allclose(value, [1, 2, 3])
        np.testing.assert_array_equal(n, [10, 10, 10])

    def test_sparse_and_out_of_range_bins(self):
        x = np.array([0.05, 0.05, 0.15, 5.0])            # 5.0 is beyond the last edge and ignored
        _, value, n = metrics.binned_rms(x, np.ones(4), np.zeros(4), [0, 0.1, 0.2], min_count=2)
        np.testing.assert_array_equal(n, [2, 1])
        self.assertAlmostEqual(value[0], 1.0)
        self.assertTrue(np.isnan(value[1]))


class Grid(unittest.TestCase):
    def test_constant_field_and_greyed_cells(self):
        # 4 x 3 grid of 400 x 400 px cells; 10 points in every cell except one with 3
        rng = np.random.default_rng(1)
        u, v = [], []
        for row in range(3):
            for col in range(4):
                k = 3 if (row, col) == (2, 3) else 10
                u.append(rng.uniform(col * 400, (col + 1) * 400, k))
                v.append(rng.uniform(row * 400, (row + 1) * 400, k))
        u, v = np.concatenate(u), np.concatenate(v)
        du, dv = np.full(u.size, 0.3), np.full(u.size, -0.4)        # length 0.5 everywhere
        g = metrics.grid_stats(u, v, du, dv, IMAGE_SIZE, (4, 3), min_count=5)
        self.assertEqual(g["n"].shape, (3, 4))
        self.assertEqual(g["n"][2, 3], 3)
        self.assertFalse(g["valid"][2, 3])
        self.assertEqual(int(g["valid"].sum()), 11)
        self.assertTrue(np.isnan(g["mean_error"][2, 3]))
        np.testing.assert_allclose(g["mean_error"][g["valid"]], 0.5)
        np.testing.assert_allclose(g["mean_du"][g["valid"]], 0.3)
        np.testing.assert_allclose(g["mean_dv"][g["valid"]], -0.4)

    def test_mean_error_is_length_but_vector_cancels(self):
        # opposite residuals: random error 1 px, no systematic shift
        u, v = np.full(10, 100.0), np.full(10, 100.0)
        du = np.tile([1.0, -1.0], 5)
        g = metrics.grid_stats(u, v, du, np.zeros(10), IMAGE_SIZE, (4, 3))
        self.assertAlmostEqual(g["mean_error"][0, 0], 1.0)
        self.assertAlmostEqual(g["mean_du"][0, 0], 0.0)

    def test_points_on_the_far_edge_and_outside(self):
        g = metrics.grid_stats([1599.9, 1600.0, -1], [1199.9, 0, 0], np.ones(3), np.zeros(3),
                               IMAGE_SIZE, (4, 3), min_count=1)
        self.assertEqual(int(g["n"].sum()), 1)
        self.assertEqual(g["n"][2, 3], 1)


class Stats(unittest.TestCase):
    def test_summary_ignores_nan(self):
        s = stats.summarise([1.0, 2.0, np.nan, 3.0])
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["mean"], 2.0)

    def test_consistent_difference_is_not_noise(self):
        a = 0.30 + 0.05 * np.random.default_rng(2).standard_normal(28)
        self.assertFalse(stats.paired_difference(a + 0.02, a)["within_noise"])

    def test_difference_straddling_zero_is_noise(self):
        a = np.linspace(0.2, 0.4, 28)
        b = a + np.tile([0.01, -0.01], 14)
        self.assertTrue(stats.paired_difference(a, b)["within_noise"])


if __name__ == "__main__":
    unittest.main()

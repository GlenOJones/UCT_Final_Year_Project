"""The two camera models on synthetic board views made with known parameters: projection and
undistortion invert each other, posing recovers a known pose, fitting recovers known intrinsics, and
triangulation recovers known lengths.

Run from the project root:
    src/venv/bin/python -m unittest discover -s src/experiments/compensation/tests -v
"""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibmethods import data, models  # noqa: E402

IMAGE_SIZE = (1600, 1200)
K = np.array([[955.0, 0, 792.0], [0, 954.0, 642.0], [0, 0, 1]])
TRUE = {"brown_conrady": np.array([0.06, -0.08, 0.0003, -0.0003, 0.02]),
        "kannala_brandt": np.array([0.02, -0.01, 0.004, -0.001])}
N_WATER = 1.333
K_AIR = np.array([[718.0, 0, 792.0], [0, 717.0, 642.0], [0, 0, 1]])
D_AIR = np.array([-0.02, 0.01, 0.0002, -0.0002, -0.003])


def pinax():
    m = models.Pinax(n_water=N_WATER)
    m.K, m.D = K_AIR.copy(), D_AIR.copy()
    return m


def views(model, n=25, seed=0):
    """Synthetic board Frames seen by `model` from varied poses, distances 300-1300 mm."""
    rng = np.random.default_rng(seed)
    obj = data.board_points().reshape(-1, 3)
    centre = data.board_centre()
    frames = []
    while len(frames) < n:
        rvec = np.radians(rng.uniform(-35, 35, 3)) * [1, 1, 0.3]
        R = cv2.Rodrigues(rvec)[0]
        t = np.array([rng.uniform(-150, 150), rng.uniform(-100, 100), rng.uniform(300, 1300)]) - R @ centre
        img = model.project(obj, rvec, t)
        inside = (img[:, 0] >= 0) & (img[:, 0] < IMAGE_SIZE[0]) & (img[:, 1] >= 0) & (img[:, 1] < IMAGE_SIZE[1])
        inside = inside.reshape(-1, 4).all(axis=1)
        if inside.sum() < 20:
            continue
        ids = np.flatnonzero(inside)
        k4 = np.repeat(inside, 4)
        frames.append(data.Frame(f"v{len(frames)}", "g", ids, obj[k4], img[k4]))
    return frames


class Models(unittest.TestCase):
    def each_model(self, fittable_only=False):
        for key, cls in models.MODELS.items():
            if key == "pinax" and fittable_only:
                continue
            with self.subTest(model=key):
                yield pinax() if key == "pinax" else cls(K, TRUE[key])

    def test_normalise_inverts_project(self):
        for m in self.each_model():
            f = views(m, n=3)[0]
            rvec, tvec = np.zeros(3), np.array([-200.0, -150, 400])
            pts = m.project(f.obj, rvec, tvec)
            cam = f.obj + tvec
            np.testing.assert_allclose(m.normalise(pts), cam[:, :2] / cam[:, 2:], atol=1e-9)

    def test_pose_recovers_known_pose(self):
        for m in self.each_model():
            f = views(m, n=1, seed=3)[0]
            rvec, tvec = m.pose(f.obj, f.img)
            np.testing.assert_allclose(m.project(f.obj, rvec, tvec), f.img, atol=1e-6)

    def test_fit_recovers_intrinsics(self):
        for m in self.each_model(fittable_only=True):
            frames = views(m)
            fitted = type(m)().fit(frames, IMAGE_SIZE)
            self.assertLess(fitted.fit_rms, 1e-3)
            np.testing.assert_allclose(fitted.K, K, atol=0.05)
            np.testing.assert_allclose(fitted.D, m.D, atol=1e-3)

    def test_pinax_obeys_snell(self):
        """The air-side ray of a water ray at angle theta_w makes sin(theta_a) = n sin(theta_w)."""
        m = pinax()
        theta_w = np.radians([1, 10, 25, 40])
        X = np.column_stack([np.tan(theta_w), np.zeros(4), np.ones(4)]) * 1000.0
        px = m.project(X, np.zeros(3), np.zeros(3))
        xa = cv2.undistortPoints(px.reshape(-1, 1, 2), K_AIR, D_AIR,
                                 criteria=models.UNDISTORT_CRITERIA).reshape(-1, 2)
        theta_a = np.arctan(np.hypot(xa[:, 0], xa[:, 1]))
        np.testing.assert_allclose(np.sin(theta_a), N_WATER * np.sin(theta_w), atol=1e-9)

    def test_pinax_paraxial_focal_is_n_times_air(self):
        """Near the axis a flat interface only scales the image: water focal length = n x air focal length."""
        m = pinax()
        X = np.array([[0.5, 0.0, 1000.0], [0.0, 0.5, 1000.0]])
        px = m.project(X, np.zeros(3), np.zeros(3))
        np.testing.assert_allclose((px[0, 0] - K_AIR[0, 2]) / 0.0005, N_WATER * K_AIR[0, 0], rtol=1e-5)
        np.testing.assert_allclose((px[1, 1] - K_AIR[1, 2]) / 0.0005, N_WATER * K_AIR[1, 1], rtol=1e-5)

    def test_stereo_lm_matches_opencv(self):
        """stereo_lm (used for Kannala-Brandt) must give cv2.stereoCalibrate's answer on Brown-Conrady."""
        rng = np.random.default_rng(7)
        R = cv2.Rodrigues(np.radians([0.1, -0.8, 0.12]))[0]
        T = np.array([-150.0, 0.5, -0.2])
        left = models.BrownConrady(K, TRUE["brown_conrady"])
        right = models.BrownConrady(K + [[1, 0, -20], [0, 1, -10], [0, 0, 0]], TRUE["brown_conrady"] * 1.2)
        pairs = []
        for f in views(left, n=12, seed=11):
            rvec, tvec = left.pose(f.obj, f.img)
            X = (cv2.Rodrigues(rvec)[0] @ f.obj.T).T + tvec
            r_img = right.project(X, cv2.Rodrigues(R)[0].ravel(), T)
            noise = lambda n: rng.normal(0, 0.2, (n, 2))     # noqa: E731
            pairs.append((data.Frame(f.name, "g", f.ids, f.obj, f.img + noise(len(f.img))),
                          data.Frame(f.name, "g", f.ids, f.obj, r_img + noise(len(f.img)))))
        R_cv, T_cv, rms_cv = left.stereo(pairs, right, IMAGE_SIZE)
        R_lm, T_lm, rms_lm = models.stereo_lm(left, right, pairs)
        self.assertAlmostEqual(rms_lm, rms_cv, delta=1e-4)
        np.testing.assert_allclose(T_lm, T_cv, atol=1e-3)
        np.testing.assert_allclose(R_lm, R_cv, atol=1e-6)

    def test_triangulation_recovers_lengths(self):
        R = cv2.Rodrigues(np.radians([0, -0.8, 0]))[0]
        T = np.array([-150.0, 0, 0])
        for m in self.each_model():
            f = views(m, n=1, seed=5)[0]
            rvec, tvec = m.pose(f.obj, f.img)
            X_left = (cv2.Rodrigues(rvec)[0] @ f.obj.T).T + tvec
            right_img = m.project(X_left, cv2.Rodrigues(R)[0].ravel(), T)
            X = models.triangulate(m, m, R, T, f.img, right_img)
            np.testing.assert_allclose(X, X_left, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

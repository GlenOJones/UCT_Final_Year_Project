"""The two camera models on synthetic board views made with known parameters: projection and undistortion invert each
other, posing recovers a known pose, fitting recovers known intrinsics, and triangulation recovers known lengths.
"""
import os
import sys
import unittest

import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calibmethods import covariance, data, models  # noqa: E402

IMAGE_SIZE = (1600, 1200)
K = np.array([[955.0, 0, 792.0], [0, 954.0, 642.0], [0, 0, 1]])
TRUE = {"brown_conrady": np.array([0.06, -0.08, 0.0003, -0.0003, 0.02]),
        "kannala_brandt": np.array([0.02, -0.01, 0.004, -0.001])}
TRUE["brown_conrady_axial"] = TRUE["brown_conrady"]
SHIFT = np.array([-12.0, 6.0])     # mm: dz = -2.9 mm at r = 0.5, -5.2 mm at r = 0.8
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
            if key in ("pinax", "air_direct") and fittable_only:
                continue
            with self.subTest(model=key):
                if key == "pinax":
                    yield pinax()
                elif key == "air_direct":
                    m = cls()
                    m.K, m.D = K.copy(), TRUE["brown_conrady"].copy()
                    yield m
                elif key == "brown_conrady_axial":
                    yield cls(K, TRUE[key], SHIFT)
                else:
                    yield cls(K, TRUE[key])

    def test_normalise_inverts_project(self):
        for m in self.each_model():
            if not getattr(m, "central", True):
                continue            # a non-central ray's direction is not X/Z; see Axial.test_ray_hits_point
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
            if hasattr(m, "shift"):
                np.testing.assert_allclose(fitted.shift, m.shift, atol=0.05)

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


class Axial(unittest.TestCase):
    """BrownConradyAxial: rays, nesting with Brown-Conrady, and recovering a known shift."""

    def test_ray_hits_point(self):
        """Each pixel's ray, from its axis crossing along its direction, passes through the 3-D point."""
        m = models.BrownConradyAxial(K, TRUE["brown_conrady"], SHIFT)
        X = np.array([[0, 0, 500.0], [200, -100, 400], [-450, 300, 700], [800, 500, 1200]])
        px = m.project(X, np.zeros(3), np.zeros(3))
        o, xy = m.ray_origins(px), m.normalise(px)
        lam = X[:, 2] - o[:, 2]
        np.testing.assert_allclose(o + lam[:, None] * np.column_stack([xy, np.ones(len(xy))]), X, atol=1e-6)

    def test_zero_shift_is_brown_conrady(self):
        f = views(models.BrownConrady(K, TRUE["brown_conrady"]), n=1, seed=2)[0]
        rvec, tvec = np.array([0.1, -0.2, 0.05]), np.array([-100.0, -80, 700])
        np.testing.assert_allclose(models.BrownConradyAxial(K, TRUE["brown_conrady"]).project(f.obj, rvec, tvec),
                                   models.BrownConrady(K, TRUE["brown_conrady"]).project(f.obj, rvec, tvec),
                                   atol=1e-9)

    def test_shift_is_depth_dependent(self):
        """The same ray angle at two depths lands on different pixels than Brown-Conrady puts it, by ~f dz r/Z."""
        bc, ax = models.BrownConrady(K, TRUE["brown_conrady"]), models.BrownConradyAxial(K, TRUE["brown_conrady"], SHIFT)
        diffs = []
        for Z in (300.0, 1200.0):
            X = np.array([[0.5 * Z, 0, Z]])
            diffs.append(np.linalg.norm(ax.project(X, np.zeros(3), np.zeros(3)) - bc.project(X, np.zeros(3), np.zeros(3))))
        self.assertAlmostEqual(diffs[0] / diffs[1], 4.0, delta=0.1)

    def test_midpoint_matches_opencv_for_central(self):
        """With central models the midpoint triangulation equals cv2.triangulatePoints on exact data."""
        R, T = cv2.Rodrigues(np.radians([0.2, -0.8, 0.1]))[0], np.array([-150.0, 0.5, 0])
        m = models.BrownConrady(K, TRUE["brown_conrady"])
        f = views(m, n=1, seed=5)[0]
        rvec, tvec = m.pose(f.obj, f.img)
        X_left = (cv2.Rodrigues(rvec)[0] @ f.obj.T).T + tvec
        right_img = m.project(X_left, cv2.Rodrigues(R)[0].ravel(), T)
        np.testing.assert_allclose(models.midpoint_triangulate(m, m, R, T, f.img, right_img),
                                   models.triangulate(m, m, R, T, f.img, right_img), atol=1e-6)

    def test_brown_conrady_data_fits_zero_shift(self):
        """Single-viewpoint data: the axial fit finds dz ~ 0 and the same intrinsics as Brown-Conrady."""
        frames = views(models.BrownConrady(K, TRUE["brown_conrady"]))
        fitted = models.BrownConradyAxial().fit(frames, IMAGE_SIZE)
        np.testing.assert_allclose(fitted.shift, 0, atol=0.05)
        np.testing.assert_allclose(fitted.K, K, atol=0.05)


def write_air_yaml(path, R, T):
    """A minimal in-air Calibration.py YAML: both cameras' K, D, the image size and the stereo extrinsics."""
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    fs.startWriteStruct("image", cv2.FILE_NODE_MAP)
    fs.write("width_px", IMAGE_SIZE[0])
    fs.write("height_px", IMAGE_SIZE[1])
    fs.endWriteStruct()
    for camera in ("left", "right"):
        fs.startWriteStruct(f"{camera}_camera", cv2.FILE_NODE_MAP)
        fs.write("camera_matrix", K_AIR)
        fs.write("distortion_coefficients", D_AIR.reshape(1, -1))
        fs.endWriteStruct()
    fs.startWriteStruct("stereo", cv2.FILE_NODE_MAP)
    fs.write("rotation_matrix", R)
    fs.write("translation_mm", T.reshape(3, 1))
    fs.endWriteStruct()
    fs.release()


class InAirAndCovariance(unittest.TestCase):
    def test_air_direct_uses_air_calibration_unchanged(self):
        """Fitting reads the in-air K, D; stereo returns the in-air R, T, and scores them at ~0 px on data made with
        exactly that rig."""
        R, T = cv2.Rodrigues(np.radians([0.2, -0.8, 0.1]))[0], np.array([-150.0, 0.5, -0.2])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "air_stereo.yaml")
            write_air_yaml(path, R, T)
            left = models.AirDirect(path).fit([], IMAGE_SIZE, "left")
            right = models.AirDirect(path).fit([], IMAGE_SIZE, "right")
        np.testing.assert_allclose(left.K, K_AIR)
        np.testing.assert_allclose(left.D, D_AIR)
        pairs = []
        for f in views(left, n=4, seed=13):
            rvec, tvec = left.pose(f.obj, f.img)
            X = (cv2.Rodrigues(rvec)[0] @ f.obj.T).T + tvec
            pairs.append((f, data.Frame(f.name, "g", f.ids, f.obj, right.project(X, cv2.Rodrigues(R)[0].ravel(), T))))
        R_out, T_out, rms = left.stereo(pairs, right, IMAGE_SIZE)
        np.testing.assert_allclose(R_out, R)
        np.testing.assert_allclose(T_out, T)
        self.assertLess(rms, 1e-6)

    def test_covariance_matches_opencv(self):
        """The Schur-complement covariance gives OpenCV's calibrateCameraExtended standard deviations."""
        rng = np.random.default_rng(21)
        frames = [data.Frame(f.name, f.group, f.ids, f.obj, f.img + rng.normal(0, 0.3, f.img.shape))
                  for f in views(models.BrownConrady(K, TRUE["brown_conrady"]), n=20, seed=17)]
        m = models.BrownConrady().fit(frames, IMAGE_SIZE)
        res = covariance.intrinsic_covariance(m, frames)
        out = cv2.calibrateCameraExtended([f.obj.astype(np.float32) for f in frames],
                                          [f.img.astype(np.float32) for f in frames], IMAGE_SIZE, None, None,
                                          criteria=models.FIT_CRITERIA)
        np.testing.assert_allclose(res["sigma"], out[5].ravel()[:9], rtol=2e-3)
        self.assertAlmostEqual(res["sigma_px"], 0.3, delta=0.02)
        C = np.array(res["corr"])
        np.testing.assert_allclose(np.diag(C), 1, atol=1e-9)
        np.testing.assert_allclose(C, C.T, atol=1e-9)

    def test_param_vector_round_trip(self):
        for m in (models.BrownConrady(K, TRUE["brown_conrady"]), models.KannalaBrandt(K, TRUE["kannala_brandt"]),
                  models.BrownConradyAxial(K, TRUE["brown_conrady"], SHIFT)):
            with self.subTest(model=type(m).__name__):
                p = models.get_params(m)
                self.assertEqual(len(p), len(models.param_names(m)))
                back = models.with_params(m, p)
                np.testing.assert_allclose(models.get_params(back), p)
                np.testing.assert_allclose(back.project(np.array([[100.0, 50, 600]]), np.zeros(3), np.zeros(3)),
                                           m.project(np.array([[100.0, 50, 600]]), np.zeros(3), np.zeros(3)))


if __name__ == "__main__":
    unittest.main()

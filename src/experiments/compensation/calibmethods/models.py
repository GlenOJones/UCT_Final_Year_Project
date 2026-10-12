"""The camera models being compared, behind one interface so that fitting, posing, projecting and
triangulating are written once and differ only where the model does.

    BrownConrady   pinhole + k1 k2 k3 radial, p1 p2 tangential       cv2.calibrateCamera
    KannalaBrandt  pinhole + k1..k4 on theta (no tangential terms)   cv2.fisheye.calibrate
    Pinax          in-air Brown-Conrady calibration + Snell refraction into water; nothing fitted in water

Both fits are Levenberg-Marquardt on pixel reprojection error over intrinsics and one board pose per
image. They differ in initialisation: calibrateCamera starts from Zhang's closed-form estimate from
the board homographies; fisheye.calibrate puts the principal point at the image centre, estimates the
focal length by least squares from the homographies, and starts the distortion at zero.

Held-out frames are posed with the intrinsics fixed, and the pose minimises PIXEL error for both models:
    BrownConrady   cv2.solvePnP (SOLVEPNP_ITERATIVE: homography start, then LM on pixel error)
    KannalaBrandt  cv2.fisheye.solvePnP for the start, then LM on pixel error (scipy least_squares).
                   cv2.fisheye.solvePnP alone fits in undistorted normalised coordinates, which weights
                   points differently from the Brown-Conrady pose and would make the two models'
                   held-out errors incomparable.
    Pinax          cv2.solvePnP on the refracted normalised rays for the start, then LM on pixel error.
"""
import cv2
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares

from .refraction import refract_radial, unrefract_radial  # noqa: E402  (one definition of the physics)

# Same termination for every fit, so neither model is stopped early relative to the other. The
# pipeline (Calibration.py) uses calibrateCameraExtended's default (30 iterations); 100 lets both
# solvers converge on the larger training sets without changing the result on the pipeline's own set.
FIT_CRITERIA = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-9)
UNDISTORT_CRITERIA = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-12)


class CameraModel:
    name = ""
    function = ""          # the OpenCV call that fits it, for the report
    distortion_names = ()

    def __init__(self, K=None, D=None):
        self.K = None if K is None else np.asarray(K, np.float64)
        self.D = None if D is None else np.asarray(D, np.float64).ravel()
        self.fit_rms = float("nan")
        self.flags = 0

    @property
    def principal_point(self):
        return float(self.K[0, 2]), float(self.K[1, 2])

    @property
    def n_intrinsics(self):
        return 4 + len(self.distortion_names)

    def fit(self, frames, image_size, camera=None):
        raise NotImplementedError

    def project(self, obj, rvec, tvec):
        return self.project_jac(obj, rvec, tvec)[0]

    def project_jac(self, obj, rvec, tvec):
        """(pixel points (N, 2), d pixels / d (rvec, tvec) as (2N, 6))."""
        raise NotImplementedError

    def normalise(self, img):
        """Undistorted normalised image coordinates (x/z, y/z) of pixel points, (N, 2)."""
        raise NotImplementedError

    def initial_pose(self, obj, img):
        raise NotImplementedError

    def pose(self, obj, img):
        """(rvec, tvec) of a board view with the intrinsics fixed, minimising pixel reprojection error:
        the model's initial pose refined by Levenberg-Marquardt on pixel error."""
        x0 = np.concatenate(self.initial_pose(obj, img))
        fit = least_squares(lambda x: (self.project(obj, x[:3], x[3:]) - img).ravel(), x0,
                            jac=lambda x: self.project_jac(obj, x[:3], x[3:])[1], method="lm",
                            xtol=1e-12, ftol=1e-12)
        return fit.x[:3], fit.x[3:]

    def stereo(self, pairs, right, image_size):
        """R, T (left to right, mm) and RMS px with both cameras' intrinsics held fixed."""
        raise NotImplementedError

    def to_dict(self):
        return {"model": self.name, "opencv_function": self.function,
                "camera_matrix": self.K.tolist(),
                "distortion": dict(zip(self.distortion_names, self.D.tolist())),
                "n_intrinsic_parameters": self.n_intrinsics, "flags": int(self.flags),
                "fit_rms_px": self.fit_rms}


class BrownConrady(CameraModel):
    name = "Brown-Conrady"
    function = "cv2.calibrateCamera"
    distortion_names = ("k1", "k2", "p1", "p2", "k3")

    def fit(self, frames, image_size, camera=None):
        obj = [f.obj.astype(np.float32) for f in frames]
        img = [f.img.astype(np.float32) for f in frames]
        self.flags = 0                         # fx, fy, cx, cy, k1, k2, p1, p2, k3 all free
        self.fit_rms, self.K, D, _, _ = cv2.calibrateCamera(obj, img, image_size, None, None,
                                                            flags=self.flags, criteria=FIT_CRITERIA)
        self.D = D.ravel()
        return self

    def project_jac(self, obj, rvec, tvec):
        img, jac = cv2.projectPoints(np.asarray(obj, np.float64), np.asarray(rvec, np.float64),
                                     np.asarray(tvec, np.float64), self.K, self.D)
        return img.reshape(-1, 2), jac[:, :6]            # columns: rvec, tvec, f, c, distortion

    def normalise(self, img):
        return cv2.undistortPoints(img.reshape(-1, 1, 2), self.K, self.D,
                                   criteria=UNDISTORT_CRITERIA).reshape(-1, 2)

    def initial_pose(self, obj, img):
        ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, self.D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise RuntimeError("solvePnP failed")
        return rvec.ravel(), tvec.ravel()

    def pose(self, obj, img):
        return self.initial_pose(obj, img)      # SOLVEPNP_ITERATIVE already minimises pixel error

    def stereo(self, pairs, right, image_size):
        obj = [lf.obj.astype(np.float32) for lf, _ in pairs]
        result = cv2.stereoCalibrate(obj, [lf.img.astype(np.float32) for lf, _ in pairs],
                                     [rf.img.astype(np.float32) for _, rf in pairs],
                                     self.K, self.D, right.K, right.D, image_size,
                                     flags=cv2.CALIB_FIX_INTRINSIC, criteria=FIT_CRITERIA)
        return result[5], result[6].ravel(), float(result[0])


class KannalaBrandt(CameraModel):
    name = "Kannala-Brandt"
    function = "cv2.fisheye.calibrate"
    distortion_names = ("k1", "k2", "k3", "k4")

    def fit(self, frames, image_size, camera=None):
        obj = [f.obj.reshape(1, -1, 3).astype(np.float64) for f in frames]
        img = [f.img.reshape(1, -1, 2).astype(np.float64) for f in frames]
        # RECOMPUTE_EXTRINSIC re-poses every board after each intrinsic step (OpenCV's recommended use);
        # FIX_SKEW matches the Brown-Conrady model, which has no skew term. CHECK_COND is left off: it
        # throws on any ill-conditioned view instead of fitting, and the views are already the pipeline's
        # cleaned set.
        self.flags = cv2.CALIB_RECOMPUTE_EXTRINSIC | cv2.CALIB_FIX_SKEW
        self.fit_rms, self.K, D, _, _ = cv2.fisheye.calibrate(obj, img, image_size, None, None,
                                                              flags=self.flags, criteria=FIT_CRITERIA)
        self.D = D.ravel()
        return self

    def project_jac(self, obj, rvec, tvec):
        img, jac = cv2.fisheye.projectPoints(np.asarray(obj, np.float64).reshape(-1, 1, 3),
                                             np.asarray(rvec, np.float64).reshape(3, 1),
                                             np.asarray(tvec, np.float64).reshape(3, 1), self.K, self.D)
        return img.reshape(-1, 2), jac[:, 8:14]          # columns: f, c, k1..k4, rvec, tvec, skew

    def normalise(self, img):
        return cv2.fisheye.undistortPoints(img.reshape(-1, 1, 2).astype(np.float64), self.K, self.D,
                                           criteria=UNDISTORT_CRITERIA).reshape(-1, 2)

    def initial_pose(self, obj, img):
        ok, rvec, tvec = cv2.fisheye.solvePnP(obj.reshape(-1, 1, 3).astype(np.float64),
                                              img.reshape(-1, 1, 2).astype(np.float64), self.K, self.D)
        if not ok:
            raise RuntimeError("fisheye.solvePnP failed")
        return rvec.ravel(), tvec.ravel()

    def stereo(self, pairs, right, image_size):
        # cv2.fisheye.stereoCalibrate (OpenCV 5.0.0) fails with "number of matrix elements is not
        # divisible" unless every pair has the same number of points, and ours cannot: a pair keeps the
        # tags both cameras saw. So the same fit is done by stereo_lm, which minimises the same pixel error
        # with the intrinsics fixed, and reproduces cv2.stereoCalibrate on Brown-Conrady to 1e-4 px
        # (tests/test_models.py).
        return stereo_lm(self, right, pairs)


def _mean_rotation(rotations):
    """Chordal mean of rotation matrices: the nearest rotation to their sum."""
    u, _, vt = np.linalg.svd(np.sum(rotations, axis=0))
    return u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt


def stereo_lm(left, right, pairs):
    """R, T (left to right) and RMS px over both cameras, intrinsics fixed: Levenberg-Marquardt-style
    least squares on pixel error over R, T and one board pose per pair, as cv2.stereoCalibrate does with
    CALIB_FIX_INTRINSIC. Started from each pair's separately posed boards."""
    poses_l = [left.pose(lf.obj, lf.img) for lf, _ in pairs]
    poses_r = [right.pose(rf.obj, rf.img) for _, rf in pairs]
    rel = [cv2.Rodrigues(rr)[0] @ cv2.Rodrigues(rl)[0].T for (rl, _), (rr, _) in zip(poses_l, poses_r)]
    R0 = _mean_rotation(rel)
    T0 = np.median([tr - R0 @ tl for (_, tl), (_, tr) in zip(poses_l, poses_r)], axis=0)
    x0 = np.concatenate([cv2.Rodrigues(R0)[0].ravel(), T0] + [np.concatenate(p) for p in poses_l])

    def right_pose(x, k):
        """Right camera's board pose for pair k and its derivatives w.r.t. (pair pose, (om, T))."""
        rv, tv = x[6 + 6 * k: 9 + 6 * k], x[9 + 6 * k: 12 + 6 * k]
        r3, t3, dr3dr1, dr3dt1, dr3dr2, dr3dt2, dt3dr1, dt3dt1, dt3dr2, dt3dt2 = cv2.composeRT(rv, tv, x[:3], x[3:6])
        return (rv, tv, r3.ravel(), t3.ravel(),
                np.block([[dr3dr1, dr3dt1], [dt3dr1, dt3dt1]]), np.block([[dr3dr2, dr3dt2], [dt3dr2, dt3dt2]]))

    def residuals(x):
        out = []
        for k, (lf, rf) in enumerate(pairs):
            rv, tv, r3, t3, _, _ = right_pose(x, k)
            out.append((left.project(lf.obj, rv, tv) - lf.img).ravel())
            out.append((right.project(rf.obj, r3, t3) - rf.img).ravel())
        return np.concatenate(out)

    def jacobian(x):
        rows, cols, vals = [], [], []

        def put(block, row0, col0):
            r, c = np.indices(block.shape)
            rows.append((r + row0).ravel()), cols.append((c + col0).ravel()), vals.append(block.ravel())

        row = 0
        for k, (lf, rf) in enumerate(pairs):
            rv, tv, r3, t3, d_pose, d_rt = right_pose(x, k)
            Jl = left.project_jac(lf.obj, rv, tv)[1]
            Jr = right.project_jac(rf.obj, r3, t3)[1]
            put(Jl, row, 6 + 6 * k)
            row += len(Jl)
            put(Jr @ d_pose, row, 6 + 6 * k)
            put(Jr @ d_rt, row, 0)
            row += len(Jr)
        return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(row, len(x)))

    fit = least_squares(residuals, x0, jac=jacobian, method="trf", x_scale="jac",
                        xtol=1e-12, ftol=1e-12, gtol=1e-12)
    rms = float(np.sqrt(np.mean(fit.fun ** 2) * 2))          # per point, as OpenCV reports it
    return cv2.Rodrigues(fit.x[:3])[0], fit.x[3:6], rms


class Pinax(CameraModel):
    """Pinax (Luczynski, Pfingsthorn & Birk 2017): the camera's IN-AIR calibration plus Snell's law at a flat
    interface, with nothing fitted to the underwater data.

    A pixel is undistorted with the in-air Brown-Conrady model to recover its ray in air, and the ray is
    refracted into water, sin(theta_air) = n_water sin(theta_water) (compensation.refract_radial); the glass
    index cancels out of the direction. Projection runs the same chain backwards. The rays are taken to
    leave from Pinax's single virtual centre on the optical axis. Where that centre sits (and how far real
    rays miss it) depends on the lens-to-port distance d0 and port thickness d1, but only as a constant shift
    along the axis that each board pose absorbs; the residual spread is not modelled. For a water-contact
    lens d0 and d1 have no physical value, so this far-field form is the only one that can be applied.

    The question it answers: does the in-air lens plus flat-interface physics predict the underwater image?
    If the Aquagon behaved as an in-air lens behind a flat port it would, to the accuracy of the air
    calibration. Its optics are designed for water, so differences, if any, will be largest off-axis.
    """
    name = "Pinax"
    function = "in-air cv2.calibrateCamera + Snell refraction (compensation.refract_radial)"
    distortion_names = BrownConrady.distortion_names
    POSE_STEP = 1e-7         # rad and mm, for the finite-difference pose Jacobian

    def __init__(self, air_calibration=None, n_water=1.333):
        super().__init__()
        self.air_calibration = air_calibration
        self.n_water = float(n_water)

    @property
    def n_intrinsics(self):
        return "9 from air, 0 fitted in water"

    def fit(self, frames, image_size, camera=None):
        fs = cv2.FileStorage(self.air_calibration, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise SystemExit(f"cannot read the in-air calibration {self.air_calibration}")
        node = fs.getNode(f"{camera}_camera")
        self.K = node.getNode("camera_matrix").mat()
        self.D = node.getNode("distortion_coefficients").mat().ravel()
        size = (int(fs.getNode("image").getNode("width_px").real()), int(fs.getNode("image").getNode("height_px").real()))
        fs.release()
        if size != tuple(image_size):
            raise SystemExit(f"in-air calibration is {size}, underwater images are {tuple(image_size)}")
        return self

    def _air_scale(self, r_water):
        """r_air / r_water for normalised water-side radii (n_water at the axis); NaN past the critical angle."""
        r = np.asarray(r_water, float)
        safe = np.where(r > 1e-12, r, 1.0)
        return np.where(r > 1e-12, unrefract_radial(safe, self.n_water) / safe, self.n_water)

    def _project(self, obj, rvec, tvec):
        X = obj @ cv2.Rodrigues(np.asarray(rvec, np.float64))[0].T + np.asarray(tvec, np.float64)
        xw = X[:, :2] / X[:, 2:]
        xa = xw * self._air_scale(np.hypot(xw[:, 0], xw[:, 1]))[:, None]
        rays = np.hstack([xa, np.ones((len(xa), 1))]).reshape(-1, 1, 3)
        return cv2.projectPoints(rays, np.zeros(3), np.zeros(3), self.K, self.D)[0].reshape(-1, 2)

    def project_jac(self, obj, rvec, tvec):
        x = np.concatenate([np.ravel(rvec), np.ravel(tvec)]).astype(np.float64)
        img = self._project(obj, x[:3], x[3:])
        jac = np.empty((img.size, 6))
        for k in range(6):
            step = np.zeros(6)
            step[k] = self.POSE_STEP * max(1.0, abs(x[k]))
            jac[:, k] = ((self._project(obj, (x + step)[:3], (x + step)[3:])
                          - self._project(obj, (x - step)[:3], (x - step)[3:])) / (2 * step[k])).ravel()
        return img, jac

    def normalise(self, img):
        xa = cv2.undistortPoints(img.reshape(-1, 1, 2).astype(np.float64), self.K, self.D,
                                 criteria=UNDISTORT_CRITERIA).reshape(-1, 2)
        r = np.hypot(xa[:, 0], xa[:, 1])
        safe = np.where(r > 1e-12, r, 1.0)
        return xa * np.where(r > 1e-12, refract_radial(safe, self.n_water) / safe, 1 / self.n_water)[:, None]

    def initial_pose(self, obj, img):
        ok, rvec, tvec = cv2.solvePnP(obj, self.normalise(img), np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise RuntimeError("solvePnP failed")
        return rvec.ravel(), tvec.ravel()

    def stereo(self, pairs, right, image_size):
        return stereo_lm(self, right, pairs)

    def to_dict(self):
        return {**super().to_dict(), "air_calibration": self.air_calibration, "n_water": self.n_water}


MODELS = {"brown_conrady": BrownConrady, "kannala_brandt": KannalaBrandt, "pinax": Pinax}


def triangulate(left_model, right_model, R, T, left_img, right_img):
    """3-D points (N, 3) in the left camera frame, mm, from matched pixel points in the two cameras."""
    xl = left_model.normalise(left_img)
    xr = right_model.normalise(right_img)
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, np.asarray(T, np.float64).reshape(3, 1)])
    X = cv2.triangulatePoints(P1, P2, xl.T, xr.T)
    return (X[:3] / X[3]).T

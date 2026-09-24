"""Shared pieces of the stereo reconstruction: the calibrated rig, triangulation and rigid fits.

tag_poses.py and dense_stereo.py both work from the calibration YAML that
src/01_Calibration/Calibration.py writes. Keeping the loading and the geometry here means the two
stages cannot disagree about what the rig looks like or which way a transform points.

Transform naming: T_a_b is a 4x4 matrix that maps a point in frame b into frame a,
    X_a = T_a_b @ X_b
so T_cam_board takes board coordinates into the left camera, and inverting it gives T_board_cam.
"""
import os
from dataclasses import dataclass

import cv2
import numpy as np

# Where everything lives is defined once, in src/project_paths.py.
import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from project_paths import PROJECT_ROOT, rel  # noqa: E402,F401

CAMERAS = ("left", "right")


@dataclass
class StereoRig:
    """The calibrated pair, as read from Calibration.py's YAML. Units are mm and px."""
    path: str
    image_size: tuple      # (width, height)
    K: dict                # camera -> 3x3
    D: dict                # camera -> distortion coefficients
    R: np.ndarray          # X_right = R @ X_left + T
    T: np.ndarray
    R1: np.ndarray         # rectification (stereoRectify, alpha=0)
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray

    @property
    def baseline_mm(self):
        return float(np.linalg.norm(self.T))

    def triangulate(self, left_px, right_px):
        """3D points in the LEFT camera frame (mm) from matching raw (distorted) pixel positions.

        Points are undistorted to normalised coordinates first, so the projection matrices are
        just [I|0] and [R|T] and no rectification is involved.
        """
        left = cv2.undistortPoints(np.asarray(left_px, np.float64).reshape(-1, 1, 2), self.K["left"], self.D["left"])
        right = cv2.undistortPoints(np.asarray(right_px, np.float64).reshape(-1, 1, 2), self.K["right"], self.D["right"])
        X = cv2.triangulatePoints(np.hstack([np.eye(3), np.zeros((3, 1))]), np.hstack([self.R, self.T]),
                                  left.reshape(-1, 2).T, right.reshape(-1, 2).T)
        return (X[:3] / X[3]).T

    def reprojection_error(self, X, left_px, right_px):
        """RMS distance (px) between the given pixels and where points X (left frame) project."""
        errors = []
        for camera, rvec, tvec, px in (("left", np.zeros(3), np.zeros(3), left_px),
                                       ("right", cv2.Rodrigues(self.R)[0], self.T, right_px)):
            projected = cv2.projectPoints(np.asarray(X, np.float64), rvec, tvec, self.K[camera], self.D[camera])[0]
            errors.append(projected.reshape(-1, 2) - np.asarray(px).reshape(-1, 2))
        return float(np.sqrt(np.mean(np.concatenate(errors) ** 2) * 2))


def load_rig(calibration_path):
    """Read the stereo calibration YAML written by src/01_Calibration/Calibration.py."""
    if not os.path.exists(calibration_path):
        raise SystemExit(f"no calibration at {rel(calibration_path)} - run "
                         f"src/01_Calibration/Calibration.py first, or pass --calibration")
    fs = cv2.FileStorage(calibration_path, cv2.FILE_STORAGE_READ)
    image = fs.getNode("image")
    stereo = fs.getNode("stereo")
    if stereo.getNode("status").string() != "computed":
        raise SystemExit(f"{rel(calibration_path)}: stereo extrinsics were not computed, "
                         f"so it cannot be used for reconstruction")
    rect = stereo.getNode("rectification")
    rig = StereoRig(
        path=calibration_path,
        image_size=(int(image.getNode("width_px").real()), int(image.getNode("height_px").real())),
        K={camera: fs.getNode(f"{camera}_camera").getNode("camera_matrix").mat() for camera in CAMERAS},
        D={camera: fs.getNode(f"{camera}_camera").getNode("distortion_coefficients").mat() for camera in CAMERAS},
        R=stereo.getNode("rotation_matrix").mat(),
        T=stereo.getNode("translation_mm").mat().reshape(3, 1),
        **{key: rect.getNode(key).mat() for key in ("R1", "R2", "P1", "P2", "Q")})
    fs.release()
    return rig


def find_frame_pairs(frames_dir):
    """(name, left_path, right_path) for every PNG present in both left/ and right/ of a recording.

    extract_stereo_frames.py writes a timestamp-matched pair under the same filename in each folder,
    so matching on the filename is matching on the pair.
    """
    names = {camera: sorted(os.listdir(os.path.join(frames_dir, camera)))
             for camera in CAMERAS if os.path.isdir(os.path.join(frames_dir, camera))}
    if len(names) < 2:
        raise SystemExit(f"{rel(frames_dir)} needs left/ and right/ folders of frames - check --frames-dir")
    shared = sorted(set(n for n in names["left"] if n.endswith(".png")) & set(names["right"]))
    if not shared:
        raise SystemExit(f"no PNG present in both {rel(frames_dir)}/left and /right")
    return [(n, os.path.join(frames_dir, "left", n), os.path.join(frames_dir, "right", n)) for n in shared]


# ====== RIGID GEOMETRY ======
def rigid_fit(source, target):
    """Least-squares rotation and translation (Kabsch) taking source points onto target points.

    Returns (T, rms): T is the 4x4 with target = T @ source, rms the residual in the points' units.
    No scale is fitted: both point sets are already metric, and a scale here would hide exactly the
    calibration scale error the residual is meant to show.
    """
    source, target = np.asarray(source, np.float64), np.asarray(target, np.float64)
    source_mean, target_mean = source.mean(0), target.mean(0)
    U, _, Vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    flip = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])   # keep it a rotation, not a reflection
    R = Vt.T @ flip @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, target_mean - R @ source_mean
    rms = float(np.sqrt(np.mean(np.sum((transform(T, source) - target) ** 2, axis=1))))
    return T, rms


def transform(T, points):
    """Apply a 4x4 rigid transform to an (N, 3) array of points."""
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


def invert(T):
    """Inverse of a rigid 4x4 transform, without a general matrix inverse."""
    inverse = np.eye(4)
    inverse[:3, :3] = T[:3, :3].T
    inverse[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return inverse


def mean_rotation(rotations):
    """Rotation closest (Frobenius) to the element-wise mean of several rotations.

    Good enough for averaging many noisy estimates of the same rotation, which is all it is used for.
    """
    U, _, Vt = np.linalg.svd(np.mean(rotations, axis=0))
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt

"""Pinax flat-port refraction correction, applied to the 18 Sep underwater calibration frames and evaluated.

THE MODEL
Luczynski, Pfingsthorn & Birk (2017), "The Pinax-model for accurate and efficient refraction correction of
underwater cameras in flat-pane housings", Ocean Engineering 133, 9-22. A camera sits a distance d0 behind a
flat pane of thickness d1; light crosses water -> glass -> air, refracting at both faces. The rays no longer
meet at one point, so the true camera is an AXIAL camera (all rays cross the optical axis, assumed normal to
the pane) and no pinhole model fits it exactly. Pinax replaces the axial camera with a single virtual pinhole
placed at the point that minimises the spread of the ray/axis crossings, which makes the correction a fixed
image warp that can be baked into a lookup table.

Two facts drive the implementation:
  * Snell's law at both faces gives  sin(theta_air) = n_w sin(theta_water); the glass index cancels, so the
    pane's material and thickness do NOT change the ray DIRECTION in water. They only shift the ray sideways,
    which is what moves the axis crossing and therefore sets the error of the pinhole approximation.
  * The geometry is rotationally symmetric about the pane normal, so the direction correction is a 1-D radial
    function of image radius. That is refract_radial() below; ray_trace() is the full 3-medium trace, used for
    the virtual centre and for the axial-spread diagnostic.
Total internal reflection caps the correctable half-angle in air at asin(1/n_w) ~ 48.3 deg.

WHAT THIS SCRIPT TESTS
Pinax is meant to be driven by an IN-AIR calibration. There is no in-air calibration of this rig in the repo,
so by default the script sweeps the in-air focal length, applies the correction, recalibrates, and compares
against the plain pinhole baseline that Calibration.py produces. Pass --air-calib to use the proper workflow.

The discriminating metric is NOT overall RMS. A pinhole model with radial distortion can absorb a fixed image
warp, so refraction that is the same at every distance is already inside k1..k3. What a pinhole CANNOT absorb
is the distance dependence: because the flat port is an axial camera, its effective distortion changes with
object distance. On this dataset the baseline already shows that signature (median per-view error 4.29 px
below 350 mm against 0.23 px at 700-900 mm for the left camera). So the script reports error per distance bin
and asks whether the correction flattens the curve, alongside held-out and object-space (mm) errors.

CAVEAT: the Aquagon lenses are water-corrected (7 elements, 2 aspherical), and their quoted 78 deg horizontal
FOV is an in-water figure that the baseline calibration reproduces (78.9 / 79.6 deg). A plain flat port in
front of an in-air lens would instead imply a ~95 deg in-air FOV. If the lens already compensates the port
optically, Pinax has little left to remove; that is a result worth reporting either way, not a bug.

Usage (project root):
    src/venv/bin/python src/01_Calibration/compensation.py data/Calibration/18_Sep/Rigframes
    src/venv/bin/python src/01_Calibration/compensation.py data/Calibration/18_Sep/Rigframes \
        --air-calib results/calibration/InAir_stereo.yaml --d0 8 --d1 6 --n-water 1.339
Outputs (default results/pinax/): results.json, fig_pinax.png, and a corrected sample image per camera.
"""
import argparse
import glob
import json
import os
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ====== PHYSICAL DEFAULTS (mm; override on the command line) ======
N_AIR, N_GLASS, N_WATER = 1.0, 1.49, 1.339   # acrylic pane; seawater near 0 degC is about 1.34
D0_MM, D1_MM = 8.0, 6.0                      # camera-to-pane distance and pane thickness: MEASURE THESE
DIST_BINS = [(0, 350), (350, 500), (500, 700), (700, 900), (900, 1300)]   # mm, for the distance-dependence test

# ====== BOARD AND DETECTOR: identical to src/01_Calibration/Calibration.py ======
TAG_SIZE = 35.36
PITCH = TAG_SIZE * 1.3
MIN_TAGS = 10
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
_params = cv2.aruco.DetectorParameters()
_params.markerBorderBits = 2
_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
_params.adaptiveThreshWinSizeMin, _params.adaptiveThreshWinSizeMax, _params.adaptiveThreshWinSizeStep = 3, 53, 5
_params.errorCorrectionRate = 1.0
detector = cv2.aruco.ArucoDetector(dictionary, _params)
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
_tags = []
for tag_id in range(70):
    x, y = (tag_id % 10) * PITCH, (tag_id // 10) * PITCH
    _tags.append(np.array([[x + TAG_SIZE, y, 0], [x, y, 0], [x, y + TAG_SIZE, 0], [x + TAG_SIZE, y + TAG_SIZE, 0]],
                          np.float32))
board = cv2.aruco.Board(_tags, dictionary, np.arange(70))


# ====== PINAX CORE ======
def refract_radial(r, n_w):
    """Normalised image radius in air -> normalised radius of the same ray in water (tan of its angle).

    sin(theta_air) = n_w sin(theta_water), so theta_water = asin(sin(theta_air) / n_w). Independent of the
    pane's index and thickness. r' < r: the water-side field of view is narrower, by about 1/n_w near the axis.
    """
    return np.tan(np.arcsin(np.clip(np.sin(np.arctan(r)) / n_w, -1.0, 1.0)))


def unrefract_radial(r_w, n_w):
    """Inverse of refract_radial: water-side radius -> air-side radius. NaN past total internal reflection."""
    s = n_w * np.sin(np.arctan(r_w))
    return np.where(np.abs(s) <= 1.0, np.tan(np.arcsin(np.clip(s, -1.0, 1.0))), np.nan)


def ray_trace(v0, d0, d1, n_g, n_w):
    """Trace unit air-side rays v0 (N,3) through a pane at z=d0 of thickness d1, normal +z.

    Vector Snell's law, as in the reference implementation's RayTrace.m. Returns (po, v2): the exit point on
    the pane's outer face and the unit direction in water.
    """
    v0 = v0 / np.linalg.norm(v0, axis=1, keepdims=True)
    n = np.array([0.0, 0.0, 1.0])
    pi = d0 * v0 / (v0 @ n)[:, None]                       # entry point on the inner face
    c = -(v0 @ -n)[:, None]                                # cos of the incidence angle
    out = []
    for mu, v_in in ((N_AIR / n_g, v0), (N_AIR / n_w, v0)):  # air->glass, and air->water overall
        k = 1 - mu ** 2 * (1 - c ** 2)
        out.append(mu * v_in + (mu * c - np.sqrt(np.clip(k, 0, None))) * -n)
    v1, v2 = (v / np.linalg.norm(v, axis=1, keepdims=True) for v in out)
    po = pi + d1 * v1 / (v1 @ n)[:, None]                  # exit point after crossing the glass
    return po, v2


def virtual_center(K, size, d0, d1, n_g, n_w, grid=24):
    """Pinax virtual pinhole: where the water rays, extended back, cross the optical axis.

    Returns (z_v, spread_mm). Each back-extended water ray crosses z at po_z - r_po / (r_v2 / v2_z); a true
    pinhole would give one z for all rays, so the spread of those crossings is the error the model cannot
    remove. z_v is the midpoint. This is the Pinax approximation's own accuracy limit.
    """
    u, v = np.meshgrid(np.linspace(0, size[0] - 1, grid), np.linspace(0, size[1] - 1, grid))
    pts = np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    rays = (np.linalg.inv(K) @ pts).T
    po, v2 = ray_trace(rays, d0, d1, n_g, n_w)
    r_po, r_v2 = np.hypot(po[:, 0], po[:, 1]), np.hypot(v2[:, 0], v2[:, 1])
    ok = r_v2 > 1e-9
    z = po[ok, 2] - r_po[ok] * v2[ok, 2] / r_v2[ok]
    return float(0.5 * (z.min() + z.max())), float(z.max() - z.min())


def correct_points(pts, K, D, n_w, K_new):
    """Pinax point correction: observed pixels -> pixels of an ideal pinhole looking into the water.

    Undistort with the in-air model to recover the air-side ray, refract it into water, then reproject with
    K_new. pts is (N,1,2) or (N,2); the shape is preserved.
    """
    shape = pts.shape
    xy = cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), K, D).reshape(-1, 2)
    r = np.linalg.norm(xy, axis=1)
    scale = np.where(r > 1e-12, refract_radial(r, n_w) / np.where(r > 1e-12, r, 1.0), 1.0 / n_w)
    xy = xy * scale[:, None]
    out = xy @ K_new[:2, :2].T + K_new[:2, 2]
    return out.reshape(shape).astype(np.float32)


def correction_maps(K, D, size, n_w, K_new):
    """(mapx, mapy) for cv2.remap: for each pixel of the corrected image, where to sample the original.

    Inverse of correct_points. Pixels past total internal reflection are left outside the image so remap
    fills them with the border value.
    """
    u, v = np.meshgrid(np.arange(size[0], dtype=np.float64), np.arange(size[1], dtype=np.float64))
    xy = np.stack([(u - K_new[0, 2]) / K_new[0, 0], (v - K_new[1, 2]) / K_new[1, 1]], -1).reshape(-1, 2)
    r = np.linalg.norm(xy, axis=1)
    scale = np.where(r > 1e-12, unrefract_radial(r, n_w) / np.where(r > 1e-12, r, 1.0), n_w)
    air = (xy * scale[:, None]).astype(np.float64)
    px = cv2.projectPoints(np.hstack([air, np.ones((len(air), 1))]).reshape(-1, 1, 3),
                           np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
    px[~np.isfinite(px)] = -1
    return px[:, 0].reshape(size[1], size[0]).astype(np.float32), px[:, 1].reshape(size[1], size[0]).astype(np.float32)


# ====== DETECTION ======
def detect_all(frames_dir, camera):
    """{'folder/file.png': (corners, ids)} for the usable images of one camera, as Calibration.py selects them."""
    out = {}
    for path in sorted(glob.glob(os.path.join(frames_dir, "*", camera, "*.png"))):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) < MIN_TAGS:
            continue
        corners = tuple(cv2.cornerSubPix(gray, c.reshape(4, 1, 2).copy(), (5, 5), (-1, -1), SUBPIX_CRITERIA)
                        .reshape(1, 4, 2) for c in corners)
        name = os.path.relpath(path, frames_dir).replace(os.sep + camera + os.sep, "/")
        out[name] = (corners, ids)
    return out


def match_points(det, names):
    """Object/image point arrays per view, via the board model."""
    obj, img = [], []
    for n in names:
        o, p = board.matchImagePoints(*det[n])
        obj.append(o), img.append(p)
    return obj, img


# ====== EVALUATION METRICS ======
def fit(obj, img, size, flags=0):
    rms, K, D, rvecs, tvecs, _, _, per_view = cv2.calibrateCameraExtended(obj, img, size, None, None, flags=flags)
    return dict(rms=float(rms), K=K, D=D, rvecs=rvecs, tvecs=tvecs, err=per_view.ravel())


def distances(tvecs):
    return np.array([float(t[2, 0]) for t in tvecs])


def bin_errors(z, err):
    """Median per-view error inside each distance bin, and the spread across bins."""
    rows, vals = [], []
    for lo, hi in DIST_BINS:
        m = (z >= lo) & (z < hi)
        med = float(np.median(err[m])) if m.any() else None
        rows.append(dict(lo=lo, hi=hi, n=int(m.sum()), median_px=med))
        if med is not None:
            vals.append(med)
    return rows, (max(vals) - min(vals) if len(vals) > 1 else float("nan"))


def held_out(obj, img, names, size, flags=0):
    """Leave-one-recording-out: fit without a folder, then measure that folder's views by PnP. RMS over views."""
    folders = sorted({n.split("/")[0] for n in names})
    errs = []
    for f in folders:
        tr = [i for i, n in enumerate(names) if not n.startswith(f + "/")]
        te = [i for i, n in enumerate(names) if n.startswith(f + "/")]
        if len(tr) < 6 or not te:
            continue
        _, K, D, *_ = cv2.calibrateCamera([obj[i] for i in tr], [img[i] for i in tr], size, None, None, flags=flags)
        for i in te:
            ok, rv, tv = cv2.solvePnP(obj[i], img[i], K, D)
            if not ok:
                continue
            pr = cv2.projectPoints(obj[i], rv, tv, K, D)[0].reshape(-1, 2)
            errs.append(np.sqrt(np.mean(np.sum((pr - img[i].reshape(-1, 2)) ** 2, 1))))
    e = np.array(errs)
    return dict(rms_px=float(np.sqrt(np.mean(e ** 2))), median_px=float(np.median(e)), n=len(e))


def intersection_error(cal, obj, img):
    """Object-space error (mm): back-project each corner, intersect the board plane, compare to the true corner.

    The metric recommended over reprojection error by Beyond Reprojection Error (arXiv 2608.05066): it is in
    millimetres, in the space the reconstruction actually lives in. Rays start at the origin of the fitted
    camera frame, which for a Pinax-corrected set IS the virtual centre (calibrateCamera absorbs the offset
    into every tvec), so z_v must not be applied again here.
    """
    d = []
    for o, p, rv, tv in zip(obj, img, cal["rvecs"], cal["tvecs"]):
        R = cv2.Rodrigues(rv)[0]
        truth = (R @ o.reshape(-1, 3).T + tv).T                 # true corner positions in camera coords
        n, q = R[:, 2], tv.ravel()                              # board plane: normal n through point q
        xy = cv2.undistortPoints(p.reshape(-1, 1, 2), cal["K"], cal["D"]).reshape(-1, 2)
        rays = np.hstack([xy, np.ones((len(xy), 1))])
        s = (q @ n) / (rays @ n)
        d.append(np.linalg.norm(rays * s[:, None] - truth, axis=1))
    d = np.concatenate(d)
    return dict(rms_mm=float(np.sqrt(np.mean(d ** 2))), median_mm=float(np.median(d)), p95_mm=float(np.percentile(d, 95)))


def evaluate(obj, img, names, size, label, flags=0):
    cal = fit(obj, img, size, flags)
    z = distances(cal["tvecs"])
    rows, spread = bin_errors(z, cal["err"])
    return dict(label=label, rms_px=cal["rms"], median_view_px=float(np.median(cal["err"])),
                fx=float(cal["K"][0, 0]), fy=float(cal["K"][1, 1]), cx=float(cal["K"][0, 2]),
                cy=float(cal["K"][1, 2]), k1=float(cal["D"].ravel()[0]), k2=float(cal["D"].ravel()[1]),
                bins=rows, bin_spread_px=spread, held_out=held_out(obj, img, names, size, flags),
                object_space=intersection_error(cal, obj, img), n_views=len(names)), cal


# ====== MAIN ======
def read_air_calib(path, camera):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
    node = fs.getNode(f"{camera}_camera")
    if node.empty():
        sys.exit(f"{path}: no {camera}_camera section")
    return node.getNode("camera_matrix").mat(), node.getNode("distortion_coefficients").mat()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frames_dir", help="folder of extracted pairs, e.g. data/Calibration/18_Sep/Rigframes")
    ap.add_argument("--out", default="results/pinax")
    ap.add_argument("--air-calib", help="in-air stereo YAML (proper Pinax workflow); omit to sweep f_air")
    ap.add_argument("--d0", type=float, default=D0_MM, help=f"camera-to-pane distance mm (default {D0_MM})")
    ap.add_argument("--d1", type=float, default=D1_MM, help=f"pane thickness mm (default {D1_MM})")
    ap.add_argument("--n-glass", type=float, default=N_GLASS)
    ap.add_argument("--n-water", type=float, default=N_WATER)
    ap.add_argument("--sweep", type=float, nargs=3, metavar=("LO", "HI", "N"), default=(0.3, 1.5, 13),
                    help="in-air focal sweep as a fraction of fx_baseline/n_water (default 0.3 1.5 13)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    res = dict(settings=vars(args), cameras={})
    for camera in ("left", "right"):
        det = detect_all(args.frames_dir, camera)
        names = sorted(det)
        first = cv2.imread(sorted(glob.glob(os.path.join(args.frames_dir, "*", camera, "*.png")))[0], 0)
        size = first.shape[::-1]
        obj, img = match_points(det, names)
        print(f"\n{'=' * 78}\n{camera.upper()}: {len(names)} usable views, image {size[0]}x{size[1]}")

        base, base_cal = evaluate(obj, img, names, size, "pinhole (baseline)")
        print(f"  baseline: RMS {base['rms_px']:.3f} px, held-out {base['held_out']['rms_px']:.3f} px, "
              f"bin spread {base['bin_spread_px']:.3f} px, object-space median {base['object_space']['median_mm']:.2f} mm")

        z_v, spread = virtual_center(base_cal["K"], size, args.d0, args.d1, args.n_glass, args.n_water)
        print(f"  Pinax virtual centre z_v = {z_v:.3f} mm, axis-crossing spread {spread:.3f} mm "
              f"(the pinhole approximation's own limit, from d0={args.d0} d1={args.d1} mm)")

        # Control: a pure 1/n_water scale, i.e. refraction with its nonlinearity removed. A scale is absorbed
        # exactly by fx, fy, so this MUST reproduce the baseline. Anything Pinax gains over it comes from the
        # nonlinear part of Snell's law, which is the only part a pinhole+polynomial model cannot already fit.
        cands = [("control scale-only", base_cal["K"], base_cal["D"], "scale")]
        if args.air_calib:
            K_air, D_air = read_air_calib(args.air_calib, camera)
            cands.append(("in-air calib", K_air, D_air, "pinax"))
        else:
            f0 = base_cal["K"][0, 0] / args.n_water
            for s in np.linspace(args.sweep[0], args.sweep[1], int(args.sweep[2])):
                K_air = np.array([[f0 * s, 0, base_cal["K"][0, 2]], [0, f0 * s, base_cal["K"][1, 2]], [0, 0, 1]])
                cands.append((f"f_air={f0 * s:.0f}", K_air, np.zeros(5), "pinax"))

        runs = []
        for label, K_air, D_air, kind in cands:
            K_new = K_air.copy()
            K_new[0, 0] *= args.n_water
            K_new[1, 1] *= args.n_water   # keep the corrected image at roughly the baseline scale
            if kind == "scale":
                xy = [cv2.undistortPoints(p.reshape(-1, 1, 2).astype(np.float64), K_air, D_air).reshape(-1, 2)
                      for p in img]
                cimg = [((x / args.n_water) @ K_new[:2, :2].T + K_new[:2, 2]).reshape(p.shape).astype(np.float32)
                        for x, p in zip(xy, img)]
            else:
                cimg = [correct_points(p, K_air, D_air, args.n_water, K_new) for p in img]
            if not all(np.isfinite(c).all() for c in cimg):
                print(f"  {label}: skipped (rays past total internal reflection)")
                continue
            # no distortion: the strong test, that refraction alone explains the warp
            r_nd, _ = evaluate(obj, cimg, names, size, f"{label} (no distortion)",
                               flags=cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
                               | cv2.CALIB_FIX_K3)
            r_d, _ = evaluate(obj, cimg, names, size, f"{label} + lens distortion")
            fov = float(np.degrees(2 * np.arctan(size[0] / (2 * K_air[0, 0]))))
            r_nd["f_air"] = r_d["f_air"] = float(K_air[0, 0])
            r_nd["in_air_fov_deg"] = r_d["in_air_fov_deg"] = fov
            r_nd["kind"] = r_d["kind"] = kind
            runs += [r_nd, r_d]
            print(f"  {label:20s} (in-air FOV {fov:5.1f} deg) no-dist RMS {r_nd['rms_px']:6.3f} | "
                  f"+dist RMS {r_d['rms_px']:6.3f} held-out {r_d['held_out']['rms_px']:6.3f} "
                  f"spread {r_d['bin_spread_px']:5.3f} obj {r_d['object_space']['median_mm']:5.2f} mm")

        best = min((r for r in runs if "+ lens distortion" in r["label"] and r["kind"] == "pinax"),
                   key=lambda r: r["held_out"]["rms_px"], default=None)
        if best:
            print(f"  best by held-out error: {best['label']}  "
                  f"{best['held_out']['rms_px']:.3f} px vs baseline {base['held_out']['rms_px']:.3f} px "
                  f"({100 * (best['held_out']['rms_px'] / base['held_out']['rms_px'] - 1):+.1f}%)")
            print(f"  PLAUSIBILITY: that implies an in-air FOV of {best['in_air_fov_deg']:.1f} deg. A flat port in "
                  f"front of this lens implies about {np.degrees(2 * np.arctan(size[0] / (2 * base_cal['K'][0, 0] / args.n_water))):.1f} deg. "
                  f"A best-fit value far from that means the correction is acting as a flexible radial basis, "
                  f"not identifying real port geometry.")
            K_air = next(k for lbl, k, _, _ in cands if lbl in best["label"])
            K_new = K_air.copy()
            K_new[0, 0] *= args.n_water
            K_new[1, 1] *= args.n_water
            mx, my = correction_maps(K_air, np.zeros(5) if not args.air_calib else read_air_calib(args.air_calib, camera)[1],
                                     size, args.n_water, K_new)
            sample = sorted(glob.glob(os.path.join(args.frames_dir, "*", camera, "*.png")))[len(names) // 2]
            cv2.imwrite(os.path.join(args.out, f"sample_{camera}_corrected.png"),
                        cv2.remap(cv2.imread(sample, 0), mx, my, cv2.INTER_LINEAR, borderValue=0))
        res["cameras"][camera] = dict(baseline=base, virtual_center_mm=z_v, axial_spread_mm=spread,
                                      runs=runs, best=best, sample=os.path.basename(sample) if best else None)

    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, camera in zip(axes, ("left", "right")):
        c = res["cameras"][camera]
        x = [0.5 * (b["lo"] + b["hi"]) for b in c["baseline"]["bins"]]
        ax.plot(x, [b["median_px"] for b in c["baseline"]["bins"]], "o-", label="pinhole (baseline)", lw=2)
        if c["best"]:
            ax.plot(x, [b["median_px"] for b in c["best"]["bins"]], "s--", label=c["best"]["label"])
        ax.set_xlabel("board distance (mm)"), ax.set_ylabel("median per-view error (px)")
        ax.set_title(f"{camera}: distance dependence", loc="left", fontsize=10)
        ax.grid(alpha=.3), ax.legend(fontsize=8)
    fig.savefig(os.path.join(args.out, "fig_pinax.png"), dpi=160)
    print(f"\nwrote {args.out}/results.json and fig_pinax.png")


if __name__ == "__main__":
    main()

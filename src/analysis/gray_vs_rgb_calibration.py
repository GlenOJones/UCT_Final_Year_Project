"""Quantitative comparison of grayscale sources for stereo calibration.

Question: is calibrating from the recorded luma plane (Y, what
extract_stereo_frames.py --gray saves) better than from the grayscale a
calibration tool computes from the colour PNG (RGB-gray)?

Image conditions, all derived from the same raw YUYV frames:
    Y         recorded luma, untouched (levels 16-235)             reference
    Y-full    Y stretched to 0-255 and rounded                     negative control
    RGB-gray  cv2.cvtColor(colour PNG, BGR2GRAY)                   default colour output
    JPEG95    colour frame -> JPEG q95 -> BGR2GRAY                 positive control
    JPEG75    colour frame -> JPEG q75 -> BGR2GRAY                 positive control

Stages:
    1 pixel fidelity  error against the ideal luma Yref = (Y - 16) * 255 / 219
                      over the full frame, the board, and the cornerSubPix windows
    2 corners         ChArUco detection + cornerSubPix; displacement against Y
    3 calibration     time-blocked K-fold cross-validation: held-out mono
                      reprojection error and stereo rectification (epipolar)
                      error, paired per frame, moving-block bootstrap 95% CIs
                      and Wilcoxon signed-rank tests
    4 precision       moving-block bootstrap of the whole calibration: spread
                      of intrinsics and baseline

Usage:
    src/venv/bin/python src/analysis/gray_vs_rgb_calibration.py data/Calibration/LabTesting/26Aug
Outputs (default results/gray_vs_rgb/): results.json, per_frame_cv.csv,
table_*.tex (booktabs), fig_*.pdf (+ .png previews).
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "preproccessing"))
import extract_stereo_frames as esf  # noqa: E402

CONDS = ["Y", "Y-full", "RGB-gray", "JPEG95", "JPEG75"]
TESTED = CONDS[1:]  # everything compared against Y
REGIONS = {"frame": "Full frame", "board": "Board", "win": "Corner windows"}
WIN = (5, 5)  # cornerSubPix half-window -> 11x11 px
CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
EDGES = np.arange(-64, 64.25, 0.25)  # signed pixel-error histogram bins (grey levels)
MODELS = {"rational": cv2.CALIB_RATIONAL_MODEL, "standard": 0}
BLUE, ORANGE, INK, INK2, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5",
                                                 "#256abf", "#184f95", "#0d366b"])


# ---------------------------------------------------------------- images and corners

def conditions(bgr, y):
    """Return (Yref float, {cond: uint8 gray}, {cond: BGR image it came from})."""
    yref = (y.astype(np.float32) - 16) * 255 / 219
    gray = {"Y": y, "Y-full": np.clip(np.rint(yref), 0, 255).astype(np.uint8),
            "RGB-gray": cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)}
    src = {"RGB-gray": bgr}
    for q in (95, 75):
        dec = cv2.imdecode(cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
        gray[f"JPEG{q}"], src[f"JPEG{q}"] = cv2.cvtColor(dec, cv2.COLOR_BGR2GRAY), dec
    return yref, gray, src


def detect(det, g, n_corners):
    """Refined ChArUco corners (n, 2) ordered by id if the full board is found, else None."""
    cc, ci, _, _ = det.detectBoard(g)
    if ci is None or len(ci) < n_corners:
        return None, 0 if ci is None else len(ci)
    cc = cc[np.argsort(ci.ravel())].astype(np.float32)
    return cv2.cornerSubPix(g, cc, WIN, (-1, -1), CRIT).reshape(-1, 2), len(ci)


def region_masks(corners, shape, board_scale):
    """Full-frame, board (inner-corner hull scaled to the outer edge) and corner-window masks."""
    masks = {"frame": np.ones(shape, bool)}
    if corners is None:
        return masks
    hull = cv2.convexHull(corners)
    c = hull.reshape(-1, 2).mean(0)
    board = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(board, np.rint(c + (hull.reshape(-1, 2) - c) * board_scale).astype(np.int32), 1)
    win = np.zeros(shape, np.uint8)
    for x, y in np.rint(corners).astype(int):
        win[max(y - WIN[1], 0):y + WIN[1] + 1, max(x - WIN[0], 0):x + WIN[0] + 1] = 1
    masks.update(board=board.astype(bool), win=win.astype(bool))
    return masks


def scan(rec, det, n_corners, board_scale):
    """Stages 1-2 over every frame of both cameras.

    Returns corners[cam][cond] (list per frame, None if board incomplete),
    detection counts, and pixel-error accumulators keyed by (cond, region).
    """
    corners = {cam: {c: [] for c in CONDS} for cam in esf.CAMS}
    counts = {cam: {c: [] for c in CONDS} for cam in esf.CAMS}
    acc = {(c, r): dict(hist=np.zeros(len(EDGES) - 1), n=0, abs=0.0, sq=0.0, max=0.0, gt1=0, clip=0)
           for c in TESTED for r in REGIONS}
    for cam in esf.CAMS:
        n = len(rec[cam]["pts"])
        for i, bgr in esf.ffmpeg_frames(rec[cam], range(n)):
            yref, gray, src = conditions(bgr, np.ascontiguousarray(esf.yuyv(rec[cam], i)[..., 0]))
            for c in CONDS:
                cc, k = detect(det, gray[c], n_corners)
                corners[cam][c].append(cc)
                counts[cam][c].append(k)
            masks = region_masks(corners[cam]["Y"][-1], yref.shape, board_scale)
            for c in TESTED:
                e = gray[c] - yref
                clip = ((src[c] == 0) | (src[c] == 255)).any(2) if c in src else None
                for r, m in masks.items():
                    a, em = acc[(c, r)], e[m]
                    a["hist"] += np.histogram(em, EDGES)[0]
                    a["n"] += em.size
                    a["abs"] += np.abs(em).sum()
                    a["sq"] += (em.astype(np.float64) ** 2).sum()
                    a["max"] = max(a["max"], np.abs(em).max())
                    a["gt1"] += (np.abs(em) > 1).sum()
                    a["clip"] += -1 if clip is None else clip[m].sum()
            if i % 50 == 0:
                print(f"  scanned {cam} frame {i}/{n - 1}")
    return corners, counts, acc


def pixel_summary(acc):
    """MAE, RMSE, P99 |e|, max |e|, % |e| > 1 and % RGB-clipped per (cond, region)."""
    out = {}
    for (c, r), a in acc.items():
        centres = (EDGES[:-1] + EDGES[1:]) / 2
        order = np.argsort(np.abs(centres))
        cdf = np.cumsum(a["hist"][order]) / a["n"]
        out[f"{c}|{r}"] = dict(
            mae=a["abs"] / a["n"], rmse=np.sqrt(a["sq"] / a["n"]),
            p99=float(np.abs(centres[order])[np.searchsorted(cdf, 0.99)]) + 0.125,
            max=float(a["max"]), pct_gt1=100 * a["gt1"] / a["n"],
            pct_clip=None if a["clip"] < 0 else 100 * a["clip"] / a["n"], n=int(a["n"]))
    return out


# ---------------------------------------------------------------- calibration

def calibrate(obj, ipl, ipr, size, flags):
    """Mono-calibrate both cameras, then stereo with intrinsics fixed."""
    objs = [obj] * len(ipl)
    rl, Kl, Dl, _, _ = cv2.calibrateCamera(objs, ipl, size, None, None, flags=flags)
    rr, Kr, Dr, _, _ = cv2.calibrateCamera(objs, ipr, size, None, None, flags=flags)
    st = cv2.stereoCalibrate(objs, ipl, ipr, Kl, Dl, Kr, Dr, size, flags=flags | cv2.CALIB_FIX_INTRINSIC)
    return dict(Kl=Kl, Dl=Dl, Kr=Kr, Dr=Dr, R=st[5], T=st[6], rms_l=rl, rms_r=rr, rms_s=st[0])


def reproj_rms(obj, K, D, ip):
    """Held-out reprojection RMS for one view: planar PnP (IPPE + LM refine), intrinsics fixed."""
    _, rv, tv = cv2.solvePnP(obj, ip, K, D, flags=cv2.SOLVEPNP_IPPE)
    rv, tv = cv2.solvePnPRefineLM(obj, ip, K, D, rv, tv)
    pr = cv2.projectPoints(obj, rv, tv, K, D)[0].reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((pr - ip.reshape(-1, 2)) ** 2, 1))))


def rectifier(c, size):
    """Function mapping (left pts, right pts) -> RMS vertical disparity after rectification."""
    R1, R2, P1, P2, *_ = cv2.stereoRectify(c["Kl"], c["Dl"], c["Kr"], c["Dr"], size, c["R"], c["T"], alpha=0)
    crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-10)

    def err(ipl, ipr):
        ul = cv2.undistortPoints(ipl.reshape(-1, 1, 2), c["Kl"], c["Dl"], R=R1, P=P1, criteria=crit).reshape(-1, 2)
        ur = cv2.undistortPoints(ipr.reshape(-1, 1, 2), c["Kr"], c["Dr"], R=R2, P=P2, criteria=crit).reshape(-1, 2)
        return float(np.sqrt(np.mean((ul[:, 1] - ur[:, 1]) ** 2)))
    return err


def cross_validate(obj, ip, times, size, flags, k, guard_s, block):
    """Interleaved time-blocked K-fold CV. Returns rows (frame_pos, fold, cond, err_l, err_r, err_rect).

    Consecutive blocks of `block` frames go to folds round-robin, so every
    training set spans the whole recording (all board poses); training frames
    within guard_s of any test frame are dropped, so no near-duplicate of a
    test frame is ever trained on.
    """
    rows, fold_of = [], (np.arange(len(times)) // block) % k
    for fold in range(k):
        test = np.flatnonzero(fold_of == fold)
        train = np.flatnonzero(np.abs(times[:, None] - times[test][None]).min(1) > guard_s)
        for c in CONDS:
            cal = calibrate(obj, [ip[c]["left"][j] for j in train], [ip[c]["right"][j] for j in train], size, flags)
            rect = rectifier(cal, size)
            for j in test:
                l, r = ip[c]["left"][j], ip[c]["right"][j]
                rows.append((int(j), fold, c, reproj_rms(obj, cal["Kl"], cal["Dl"], l),
                             reproj_rms(obj, cal["Kr"], cal["Dr"], r), rect(l, r)))
    return rows


def block_boot_ci(x, block, reps, rng):
    """Moving-block bootstrap 95% CI of mean(x) for a time-ordered series."""
    n = len(x)
    starts = rng.integers(0, n - block + 1, size=(reps, -(-n // block)))
    idx = (starts[..., None] + np.arange(block)).reshape(reps, -1)[:, :n]
    return np.percentile(x[idx].mean(1), [2.5, 97.5])


def compare(rows, block, reps, rng):
    """Per metric and condition: mean error, paired difference to Y, its CI and Wilcoxon p."""
    metrics = {"err_l": "Left reprojection", "err_r": "Right reprojection", "err_rect": "Stereo rectification"}
    table = {}
    for col, (m, _) in enumerate(metrics.items(), start=3):
        by = {c: np.array([r[col] for r in sorted(rows, key=lambda r: r[0]) if r[2] == c]) for c in CONDS}
        for c in CONDS:
            d = by[c] - by["Y"]
            entry = dict(mean=by[c].mean(), sd=by[c].std(ddof=1), n=len(d))
            if c != "Y":
                try:
                    p = float(wilcoxon(d).pvalue)
                except ValueError:  # all differences zero
                    p = float("nan")
                lo, hi = block_boot_ci(d, block, reps, rng)
                entry.update(delta=d.mean(), lo=lo, hi=hi, pct=100 * d.mean() / by["Y"].mean(), p=p)
            table[f"{m}|{c}"] = entry
    return metrics, table


_BOOT = {}


def _boot_init(obj, ip, size, flags):
    cv2.setNumThreads(1)
    _BOOT.update(obj=obj, ip=ip, size=size, flags=flags)


def _boot_one(idx):
    """Parameters from one bootstrap replicate (same resampled frames for every condition)."""
    out = {}
    for c in CONDS:
        cal = calibrate(_BOOT["obj"], [_BOOT["ip"][c]["left"][j] for j in idx],
                        [_BOOT["ip"][c]["right"][j] for j in idx], _BOOT["size"], _BOOT["flags"])
        out[c] = params(cal)
    return out


def params(cal):
    """Scalar calibration parameters reported in the precision analysis."""
    p = {}
    for s in ("l", "r"):
        K = cal[f"K{s}"]
        p.update({f"fx_{s}": K[0, 0], f"fy_{s}": K[1, 1], f"cx_{s}": K[0, 2], f"cy_{s}": K[1, 2]})
    p["baseline"] = float(np.linalg.norm(cal["T"]))
    p.update(rms_l=cal["rms_l"], rms_r=cal["rms_r"], rms_s=cal["rms_s"])
    return {k: float(v) for k, v in p.items()}


# ---------------------------------------------------------------- output: tables

def f(x, nd=4):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def ptex(p):
    return "--" if np.isnan(p) else ("$<$0.001" if p < 0.001 else f"{p:.3f}")


def write_table(path, caption, label, cols, header, rows, note=""):
    """Write a booktabs LaTeX table."""
    body = "\n".join(" & ".join(r) + r" \\" if isinstance(r, list) else r for r in rows)
    path.write_text(
        f"% generated by src/analysis/gray_vs_rgb_calibration.py\n\\begin{{table}}[htbp]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\footnotesize\n\\begin{{tabular}}{{{cols}}}\n"
        f"\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"
        + (f"\\par\\smallskip\\parbox{{0.95\\linewidth}}{{\\scriptsize {note}}}\n" if note else "")
        + "\\end{table}\n")


def tables(out, pix, det_rows, cv, cv_std, prec, unit):
    rows = []
    for r, name in REGIONS.items():
        rows += [r"\addlinespace"] if rows else []
        for c in TESTED:
            p = pix[f"{c}|{r}"]
            rows.append([name if c == TESTED[0] else "", c, f(p["mae"], 3), f(p["rmse"], 3), f(p["p99"], 2),
                         f(p["max"], 1), f(p["pct_gt1"], 2), f(p["pct_clip"], 2)])
    write_table(
        out / "table_pixel_fidelity.tex",
        "Pixel fidelity of each grayscale source against the ideal luma "
        r"$Y_\mathrm{ref} = (Y - 16)\cdot 255/219$, in 8-bit grey levels.",
        "tab:gray-pixel", "llrrrrrr",
        r"Region & Condition & MAE & RMSE & P99 $|e|$ & Max $|e|$ & $|e|>1$ (\%) & RGB clipped (\%)",
        rows,
        "Corner windows are the 11$\\times$11 px cornerSubPix windows. RGB clipped: pixels with any "
        "channel at 0 or 255 in the colour image the gray was derived from.")
    write_table(
        out / "table_corners.tex",
        "ChArUco detection and sub-pixel corner displacement relative to the recorded luma $Y$.",
        "tab:gray-corners", "lrrrrr",
        r"Condition & Full boards L / R & Median (px) & P95 (px) & Max (px) & Corners",
        det_rows)
    rows = []
    for m, name in cv["metrics"].items():
        rows += ([r"\addlinespace"] if rows else []) + [r"\multicolumn{6}{l}{\textit{" + name + r"}} \\"]
        for c in CONDS:
            e = cv["table"][f"{m}|{c}"]
            rows.append([f"\\quad {c}", f(e["mean"]), f(e.get("delta"), 4) if c != "Y" else "--",
                         f"[{f(e['lo'])}, {f(e['hi'])}]" if c != "Y" else "--",
                         f(e.get("pct"), 1) if c != "Y" else "--", ptex(e["p"]) if c != "Y" else "--"])
    note = (f"{cv['k']}-fold interleaved time-blocked cross-validation over {cv['n']} stereo pairs "
            f"(blocks of {cv['block']} frames, {cv['guard_s']:.1f} s guard band), {cv['model']} "
            "distortion model. Errors are per-frame "
            r"RMS in pixels on held-out frames; $\Delta$ is the paired mean difference to $Y$ "
            f"(positive = worse), CI from a moving-block bootstrap (block {cv['block']} frames), "
            "$p$ from a Wilcoxon signed-rank test (optimistic under temporal correlation).")
    write_table(out / "table_cv.tex", "Held-out calibration error by grayscale source.", "tab:gray-cv",
                "lrrrrr", r"Condition & Mean (px) & $\Delta$ vs $Y$ (px) & 95\% CI & $\Delta$ (\%) & $p$", rows, note)
    rows = [[c] + [f(cv_std["table"][f"{m}|{c}"]["mean"]) for m in cv_std["metrics"]]
            + [f(cv_std["table"][f"err_rect|{c}"].get("delta"), 4) if c != "Y" else "--"] for c in CONDS]
    write_table(out / "table_cv_standard.tex",
                "Sensitivity check: held-out error with the standard 5-coefficient distortion model.",
                "tab:gray-cv-std", "lrrrr",
                r"Condition & Left reproj. & Right reproj. & Rectification & $\Delta$ rect. vs $Y$", rows)
    boot_note = (f"{prec['reps']} moving-block bootstrap replicates (block {prec['block']} frames); "
                 "each replicate uses the same resampled frames for every condition")
    rows = []
    for c in CONDS:
        row = [c]
        for k in ("rms_l", "rms_r", "rms_s"):
            v = prec["full"][c][k]
            if c == "Y":
                row += [f(v), "--"]
                continue
            lo, hi = np.percentile([b[c][k] - b["Y"][k] for b in prec["boot"]], [2.5, 97.5])
            row += [f(v), f"{v - prec['full']['Y'][k]:+.4f} [{lo:+.4f}, {hi:+.4f}]"]
        rows.append(row)
    write_table(
        out / "table_insample.tex",
        "In-sample RMS reprojection error of the calibration on all stereo pairs, and its paired "
        r"difference to $Y$ (px).", "tab:gray-insample", "lrrrrrr",
        r"& \multicolumn{2}{c}{Left} & \multicolumn{2}{c}{Right} & \multicolumn{2}{c}{Stereo} \\"
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}" "\n"
        r"Condition & RMS & $\Delta$ [95\% CI] & RMS & $\Delta$ [95\% CI] & RMS & $\Delta$ [95\% CI]",
        rows, boot_note + "; the CI is the 2.5--97.5 percentile of the per-replicate difference.")
    keys = ["fx_l", "cx_l", "cy_l", "fx_r", "cx_r", "cy_r", "baseline"]
    rows = [[c] + [f(prec["sd"][c][k], 3 if k != "baseline" else 5)
                   + ("" if c == "Y" else f" ({prec['sd'][c][k] / prec['sd']['Y'][k]:.2f})") for k in keys]
            for c in CONDS]
    write_table(
        out / "table_precision.tex",
        "Bootstrap standard deviation of the calibrated parameters (ratio to $Y$ in brackets).",
        "tab:gray-precision", "l" + "r" * 7,
        r"Condition & $f_x^L$ & $c_x^L$ & $c_y^L$ & $f_x^R$ & $c_x^R$ & $c_y^R$ & " f"Baseline ({unit})",
        rows, boot_note + ". Intrinsics in px. Monte Carlo error of each SD is about "
              f"{100 / np.sqrt(2 * (prec['reps'] - 1)):.0f}\\%.")
    rows = []
    for k, name in PARAMS:
        nd = 4 if k == "baseline" else 2
        cells = []
        for c in TESTED:
            d = param_shift(prec, c, k)
            lo, hi = np.percentile(d, [2.5, 97.5])
            cells.append(f"{d.mean():+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]")
        rows.append([f"{name} ({'px' if k != 'baseline' else unit})"] + cells)
    write_table(
        out / "table_param_shift.tex",
        r"Paired shift of each calibrated parameter relative to $Y$: bootstrap mean [95\% interval].",
        "tab:gray-shift", "l" + "r" * len(TESTED), "Parameter & " + " & ".join(TESTED), rows, boot_note + ".")


# ---------------------------------------------------------------- output: figures

def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white", "pdf.fonttype": 42})


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_example(out, rec, det, n_corners, board_scale, cam="right", frame=150):
    """Crop of one frame: luma, |RGB-gray - Yref| and clipped-RGB mask."""
    bgr = dict(esf.ffmpeg_frames(rec[cam], [frame]))[frame]
    y = np.ascontiguousarray(esf.yuyv(rec[cam], frame)[..., 0])
    yref, gray, _ = conditions(bgr, y)
    cc, _ = detect(det, y, n_corners)
    x0, y0 = np.maximum(cc.min(0) - 90, 0).astype(int)
    x1, y1 = np.minimum(cc.max(0) + 90, [y.shape[1], y.shape[0]]).astype(int)
    crop = np.s_[y0:y1, x0:x1]
    err = np.abs(gray["RGB-gray"] - yref)[crop]
    clip = ((bgr == 0) | (bgr == 255)).any(2)[crop]
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.6), constrained_layout=True)
    ax[0].imshow(y[crop], cmap="gray", vmin=0, vmax=255)
    ax[0].plot(cc[:, 0] - x0, cc[:, 1] - y0, "o", ms=3, mfc="none", mec=ORANGE, mew=0.8)
    im = ax[1].imshow(err, cmap=SEQ, vmin=0, vmax=8)
    fig.colorbar(im, ax=ax[1], shrink=0.8, label="grey levels", extend="max")
    ax[2].imshow(y[crop], cmap="gray", vmin=0, vmax=255)
    ax[2].imshow(np.ma.masked_where(~clip, clip), cmap=LinearSegmentedColormap.from_list("o", [ORANGE, ORANGE]),
                 alpha=0.85)
    for a, t in zip(ax, ["(a) Luma $Y$, ChArUco corners", r"(b) $|$RGB-gray $- Y_\mathrm{ref}|$",
                         "(c) RGB channel clipped"]):
        a.set_title(t, loc="left", color=INK)
        a.set_xticks([]), a.set_yticks([]), a.grid(False)
    save(fig, out, "fig_example_error")


def fig_pixel_hist(out, acc):
    """Small multiples: signed error distribution per condition, corner windows vs full frame."""
    group = 8  # 0.25-level bins -> 2-level bins, wide enough to hide the 255/219 quantisation comb
    centres = EDGES[:-1].reshape(-1, group).mean(1) + (EDGES[1] - EDGES[0]) / 2
    fig, ax = plt.subplots(1, len(TESTED), figsize=(7.0, 2.1), sharey=True, constrained_layout=True)
    for a, c in zip(ax, TESTED):
        for r, col, lw in (("frame", MUTED, 1.0), ("win", BLUE, 1.6)):
            h = acc[(c, r)]["hist"].reshape(-1, group).sum(1) / acc[(c, r)]["n"]
            a.step(centres, np.where(h > 0, h, np.nan), where="mid", color=col, lw=lw, label=REGIONS[r])
        a.set_yscale("log"), a.set_xlim(-30, 30), a.set_title(c, loc="left", color=INK)
        a.set_xlabel(r"$e = g - Y_\mathrm{ref}$ (grey levels)")
    ax[0].set_ylabel("fraction of pixels")
    ax[0].legend(loc="lower left", fontsize=7)
    save(fig, out, "fig_pixel_error_hist")


def fig_displacement(out, disp):
    """Horizontal box plot of corner displacement against Y, per condition."""
    fig, ax = plt.subplots(figsize=(3.4, 1.9), constrained_layout=True)
    data = [np.maximum(disp[c], 1e-5) for c in TESTED]
    ax.boxplot(data, orientation="horizontal", tick_labels=TESTED, whis=(5, 95), widths=0.5,
               showfliers=True, flierprops=dict(marker="o", ms=2, mfc=BLUE, mec="none", alpha=0.4),
               boxprops=dict(color=BLUE), whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE),
               medianprops=dict(color=INK, lw=1.2))
    ax.set_xscale("log"), ax.invert_yaxis()
    ax.set_xlabel("corner displacement vs $Y$ (px, log scale)")
    ax.grid(axis="y", visible=False)
    save(fig, out, "fig_corner_displacement")


def fig_cv(out, cv):
    """Forest plot: paired held-out error difference to Y with 95% CI, one panel per metric."""
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 1.9), sharey=True, constrained_layout=True)
    for a, (m, name) in zip(ax, cv["metrics"].items()):
        for k, c in enumerate(TESTED):
            e = cv["table"][f"{m}|{c}"]
            a.plot([e["lo"], e["hi"]], [k, k], color=BLUE, lw=2, solid_capstyle="round")
            a.plot(e["delta"], k, "o", ms=5, color=BLUE, mec="white", mew=1)
        a.axvline(0, color=INK2, lw=0.8)
        a.set_title(name, loc="left", color=INK)
        a.set_xlabel(r"$\Delta$ held-out RMS vs $Y$ (px)")
        a.grid(axis="y", visible=False)
    ax[0].set_yticks(range(len(TESTED)), TESTED), ax[0].invert_yaxis()
    save(fig, out, "fig_cv_forest")


PARAMS = [("fx_l", "$f_x^L$"), ("cx_l", "$c_x^L$"), ("cy_l", "$c_y^L$"), ("baseline", "baseline"),
          ("fx_r", "$f_x^R$"), ("cx_r", "$c_x^R$"), ("cy_r", "$c_y^R$")]


def param_shift(prec, c, k):
    """Per-replicate paired parameter difference, condition minus Y."""
    return np.array([b[c][k] - b["Y"][k] for b in prec["boot"]])


def fig_precision(out, prec, unit):
    """Small multiples: paired bootstrap shift of each parameter relative to Y (box 25-75%, whiskers 95%)."""
    fig, axs = plt.subplots(2, 4, figsize=(7.0, 3.2), sharey=True, constrained_layout=True)
    for a, (k, name) in zip(axs.flat, PARAMS):
        a.boxplot([param_shift(prec, c, k) for c in TESTED], orientation="horizontal", tick_labels=TESTED,
                  whis=(2.5, 97.5), widths=0.55, showfliers=False, boxprops=dict(color=BLUE),
                  whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE), medianprops=dict(color=INK, lw=1.2))
        a.axvline(0, color=INK2, lw=0.6)
        a.set_title(name, loc="left", color=INK)
        a.set_xlabel(r"$\Delta$ vs $Y$ (" + ("px" if k != "baseline" else unit) + ")")
        a.grid(axis="y", visible=False)
    axs[0, 0].invert_yaxis()
    axs.flat[-1].axis("off")
    save(fig, out, "fig_bootstrap_params")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="recording folder (as for extract_stereo_frames.py)")
    ap.add_argument("--out", type=Path, default=Path("results/gray_vs_rgb"))
    ap.add_argument("--board", default="5x5", help="ChArUco squares WxH (default 5x5)")
    ap.add_argument("--dict", default="DICT_6X6_250")
    ap.add_argument("--first-id", type=int, default=24, help="id of the board's first marker (default 24)")
    ap.add_argument("--marker-ratio", type=float, default=0.75, help="marker/square side (corners are "
                    "cornerSubPix-refined, so this barely matters)")
    ap.add_argument("--square-mm", type=float, help="square side in mm (default: report in squares)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--guard-s", type=float, default=1.0, help="CV guard band around each test block")
    ap.add_argument("--block", type=int, default=10, help="bootstrap block length in frames")
    ap.add_argument("--boot", type=int, default=500, help="bootstrap replicates for parameter precision")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    style()

    sq = args.square_mm or 1.0
    unit = "mm" if args.square_mm else "squares"
    bw, bh = map(int, args.board.split("x"))
    n_markers = (bw * bh) // 2
    board = cv2.aruco.CharucoBoard((bw, bh), sq, sq * args.marker_ratio,
                                   cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict)),
                                   np.arange(args.first_id, args.first_id + n_markers))
    det = cv2.aruco.CharucoDetector(board)
    obj = board.getChessboardCorners().astype(np.float32)
    n_corners, board_scale = len(obj), bw / (bw - 2)

    rec = esf.load_recording(args.folder)
    size = (rec["right"]["w"], rec["right"]["h"])
    print(f"stage 1-2: scanning {len(rec['right']['pts'])} + {len(rec['left']['pts'])} frames, 5 conditions")
    corners, counts, acc = scan(rec, det, n_corners, board_scale)
    pix = pixel_summary(acc)

    disp, det_rows = {}, []
    for c in CONDS:
        d = [np.linalg.norm(a - b, axis=1) for cam in esf.CAMS
             for a, b in zip(corners[cam][c], corners[cam]["Y"]) if a is not None and b is not None]
        disp[c] = np.concatenate(d)
        full = [sum(x is not None for x in corners[cam][c]) for cam in esf.CAMS]
        tot = sum(sum(counts[cam][c]) for cam in esf.CAMS)
        det_rows.append([c, f"{full[0]} / {full[1]}"] + (["--"] * 3 if c == "Y" else
                        [f(np.median(disp[c])), f(np.percentile(disp[c], 95)), f(disp[c].max(), 3)]) + [str(tot)])

    # stage 3: stereo pairs (timestamp-matched) with a full board in both cameras for every condition
    max_dt = 0.25 * np.median(np.diff(rec["right"]["pts"])) / 1e6
    pairs = [(r, l) for r, l, _ in esf.pair_up(rec, range(len(rec["right"]["pts"])), max_dt)
             if all(corners["right"][c][r] is not None and corners["left"][c][l] is not None for c in CONDS)]
    times = np.array([rec["right"]["pts"][r] / 1e9 for r, _ in pairs])
    ip = {c: {"left": [corners["left"][c][l] for _, l in pairs], "right": [corners["right"][c][r] for r, _ in pairs]}
          for c in CONDS}
    print(f"stage 3: {len(pairs)} usable stereo pairs, {args.folds}-fold CV")
    cv = {}
    for model, flags in MODELS.items():
        rows = cross_validate(obj, ip, times, size, flags, args.folds, args.guard_s, args.block)
        metrics, table = compare(rows, args.block, 10000, rng)
        cv[model] = dict(metrics=metrics, table=table, rows=rows, k=args.folds, n=len(pairs),
                         guard_s=args.guard_s, block=args.block, model=model)
    with open(args.out / "per_frame_cv.csv", "w") as fh:
        fh.write("model,pair,right_frame,left_frame,fold,condition,err_left_px,err_right_px,err_rect_px\n")
        for model in MODELS:
            for j, fold, c, el, er, ex in cv[model]["rows"]:
                fh.write(f"{model},{j},{pairs[j][0]},{pairs[j][1]},{fold},{c},{el:.6f},{er:.6f},{ex:.6f}\n")

    print(f"stage 4: {args.boot} bootstrap calibrations x {len(CONDS)} conditions")
    flags = MODELS["rational"]
    full = {}
    for c in CONDS:
        cal = calibrate(obj, ip[c]["left"], ip[c]["right"], size, flags)
        full[c] = dict(rms_l=cal["rms_l"], rms_r=cal["rms_r"], rms_s=cal["rms_s"], params=params(cal))
    n, L = len(pairs), args.block
    samples = [(s[:, None] + np.arange(L)).ravel()[:n]
               for s in rng.integers(0, n - L + 1, size=(args.boot, -(-n // L)))]
    with ProcessPoolExecutor(initializer=_boot_init, initargs=(obj, ip, size, flags)) as pool:
        boot = list(pool.map(_boot_one, samples, chunksize=4))
    sd = {c: {k: float(np.std([b[c][k] for b in boot], ddof=1)) for k in full["Y"]["params"]} for c in CONDS}
    prec = dict(full=full, boot=boot, sd=sd, reps=args.boot, block=L)

    tables(args.out, pix, det_rows, cv["rational"], cv["standard"], prec, unit)
    fig_example(args.out, rec, det, n_corners, board_scale)
    fig_pixel_hist(args.out, acc)
    fig_displacement(args.out, disp)
    fig_cv(args.out, cv["rational"])
    fig_precision(args.out, prec, unit)
    results = dict(
        recording=str(args.folder), versions=dict(opencv=cv2.__version__, numpy=np.__version__),
        settings={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        pixel=pix, displacement={c: dict(median=float(np.median(disp[c])), p95=float(np.percentile(disp[c], 95)),
                                          max=float(disp[c].max()), n=int(disp[c].size)) for c in TESTED},
        cv={m: dict(table=v["table"], n=v["n"]) for m, v in cv.items()},
        precision=dict(full=full, sd=sd, shift={
            c: {k: [float(param_shift(prec, c, k).mean())] + list(np.percentile(param_shift(prec, c, k), [2.5, 97.5]))
                for k in full["Y"]["params"]} for c in TESTED}))
    (args.out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print(f"wrote results to {args.out}")


if __name__ == "__main__":
    main()

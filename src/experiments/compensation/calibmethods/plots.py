"""Figures. Every figure puts all techniques (and both cameras) on one scale, so a difference in colour,
height or arrow length is a difference in the data, not in the axes.

Curves are drawn from the residuals of all splits pooled (each held-out frame appears once per split
that held it out); the shaded band around each curve is the 2.5-97.5 percentile range of the same
curve computed split by split, i.e. how much it moves with the choice of held-out recordings.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402


def _plain_log(axis):
    """Log axis labelled 0.2, 0.5, 1, 2, 5 ... instead of 2x10^-1, with no minor labels to collide."""
    axis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())

from . import data, metrics, stats  # noqa: E402

# Categorical slots 1-8 of the reference palette, in its validated order (adjacent pairs stay apart under
# colour-vision deficiency); each technique also has its own marker, so colour is never the only cue.
# Techniques keep their colour in every figure.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
INK, INK_2, GRID, MISSING = "#0b0b0b", "#52514e", "#e4e3df", "#d6d5d0"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
# Correlation (-1 .. 1): blue for negative, orange for positive, neutral grey at 0.
DIVERGING = LinearSegmentedColormap.from_list(
    "corr", ["#184f95", "#6da7ec", "#f2f1ee", "#f0a07c", "#b8461b"])

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
    "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})


# The cameras the figures show, as indices into data.CAMERAS (points rows store the index). Every camera is
# still fitted and scored; this only chooses which appear (show_cameras(["left"]) for the reference camera).
SHOWN = list(range(len(data.CAMERAS)))


def show_cameras(names):
    SHOWN[:] = [data.CAMERAS.index(n) for n in names]


def save(fig, path_stem):
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}")
    plt.close(fig)


def _select(points, t, c):
    return points[(points[:, 1] == t) & (points[:, 2] == c)]      # column 0 is the split


def _profile(points, t, c, column, edges, min_count):
    """Pooled binned RMS plus its per-split interval: (centres, pooled, lo, hi)."""
    sel = _select(points, t, c)
    centres, pooled, _ = metrics.binned_rms(sel[:, column], sel[:, 6], sel[:, 7], edges, min_count)
    per_split = []
    for s in np.unique(points[:, 0]):
        q = sel[sel[:, 0] == s]
        per_split.append(metrics.binned_rms(q[:, column], q[:, 6], q[:, 7], edges, max(1, min_count // 4))[1])
    per_split = np.array(per_split)
    lo, hi = np.full(len(centres), np.nan), np.full(len(centres), np.nan)
    for k in range(len(centres)):
        s = stats.summarise(per_split[:, k])
        lo[k], hi[k] = s["lo"], s["hi"]
    ok = np.isfinite(pooled)
    lo[~ok] = hi[~ok] = np.nan
    return centres, pooled, lo, hi


def _legend(fig, techniques, shown=None, **kw):
    """Legend of the techniques (or only those at the indices in `shown`), each in its own colour and marker."""
    shown = range(len(techniques)) if shown is None else shown
    handles = [plt.Line2D([], [], color=SERIES[i], marker=MARKERS[i], markersize=5, label=techniques[i].label)
               for i in shown]
    # At most 4 per row, so a long list of techniques wraps instead of widening (and shrinking) the figure.
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), frameon=False, **kw)


# ====== RESIDUAL HEATMAP ======
def _nice(x):
    """Largest of 1, 2, 5 x 10^n not above x."""
    p = 10 ** np.floor(np.log10(x))
    return max(m * p for m in (1, 2, 5) if m * p <= x)


def heatmaps(points, techniques, params_all, cfg, image_size, path_stem, subset=None, log=False):
    """Rows: cameras. Columns: techniques (all, or the indices in `subset`). Cells coloured by mean residual
    length, one colour scale for all panels (logarithmic if `log`, for techniques whose errors differ by more
    than a factor of 10); arrows are the mean residual vector, one gain for all panels; grey cells have fewer
    than min_points_per_cell points. Dashed circles are the zone boundaries about the principal point."""
    w, h = image_size
    nx, ny = cfg["grid_cells"]
    shown = list(range(len(techniques))) if subset is None else list(subset)
    grids = {(col, c): metrics.grid_stats(*_select(points, t, c)[:, 4:8].T, image_size, (nx, ny),
                                          cfg["min_points_per_cell"])
             for col, t in enumerate(shown) for c in SHOWN}
    vmax = max(np.nanmax(g["mean_error"]) for g in grids.values())
    vmin = min(np.nanmin(g["mean_error"]) for g in grids.values())
    norm = LogNorm(vmin=_nice(vmin), vmax=vmax) if log else Normalize(vmin=0, vmax=vmax)
    longest = max(np.nanmax(np.hypot(g["mean_du"], g["mean_dv"])) for g in grids.values())
    cell = w / nx
    gain = _nice(0.9 * cell / longest)                      # px of image per px of residual
    ref = _nice(longest / 2)

    fig, axes = plt.subplots(len(SHOWN), len(shown), figsize=(3.2 * len(shown), 2.65 * len(SHOWN) + 0.4),
                             squeeze=False, constrained_layout=True)
    cmap = SEQUENTIAL.copy()
    cmap.set_bad(MISSING)
    xc = (np.arange(nx) + 0.5) * w / nx
    yc = (np.arange(ny) + 0.5) * h / ny
    X, Y = np.meshgrid(xc, yc)
    for (col, c), g in grids.items():
        t = shown[col]
        row = SHOWN.index(c)
        ax = axes[row, col]
        im = ax.imshow(np.ma.masked_invalid(g["mean_error"]), extent=(0, w, h, 0), cmap=cmap, norm=norm,
                       interpolation="nearest")
        v = g["valid"]
        q = ax.quiver(X[v], Y[v], g["mean_du"][v] * gain, g["mean_dv"][v] * gain, angles="xy", scale_units="xy",
                      scale=1, color=INK, width=0.006, headwidth=3.5)
        K = np.array(params_all[techniques[t].key][data.CAMERAS[c]]["camera_matrix"])
        cx, cy = K[0, 2], K[1, 2]
        r_max = max(np.hypot(x - cx, y - cy) for x in (0, w) for y in (0, h))
        for r in cfg["zone_radii"]:
            ax.add_patch(Circle((cx, cy), r * r_max, fill=False, ls="--", lw=0.8, ec=INK_2))
        ax.plot(cx, cy, "+", color=INK, ms=6)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_xticks([0, w // 2, w])
        ax.set_yticks([0, h // 2, h])
        ax.grid(False)
        if row == 0:
            ax.set_title(techniques[t].label)
        if col == 0:
            ax.set_ylabel(f"{data.CAMERAS[c]} camera\nv (px)")
        if row == len(SHOWN) - 1:
            ax.set_xlabel("u (px)")
    fig.colorbar(im, ax=axes, shrink=0.7, label="mean residual length (px)" + (", log scale" if log else ""))
    axes[-1, 0].quiverkey(q, 0.02, 0.01, ref * gain, f"arrow key: {ref:g} px mean residual", labelpos="E",
                          coordinates="figure", color=INK, labelcolor=INK)
    fig.suptitle(f"Held-out residuals on a {nx} x {ny} grid; arrows = mean residual x {gain:g}; "
                 f"grey = fewer than {cfg['min_points_per_cell']} points", fontsize=8, color=INK_2)
    save(fig, path_stem)
    return {"vmax_px": float(vmax), "arrow_gain": float(gain), "arrow_key_px": float(ref), "log_colour": log,
            "techniques": [techniques[t].key for t in shown]}


# ====== PROFILES ======
def _profile_figure(points, techniques, column, edges, cfg, xlabel, path_stem, title, decorate=None, shown=None):
    shown = list(range(len(techniques))) if shown is None else list(shown)
    fig, axes = plt.subplots(1, len(SHOWN), figsize=(3.6 * len(SHOWN) + (1.2 if len(SHOWN) == 1 else 0), 3.0),
                             sharey=True, squeeze=False, constrained_layout=True)
    axes = axes[0]
    values = []
    for c, ax in zip(SHOWN, axes):
        if decorate:
            decorate(ax)
        for t in shown:
            x, y, lo, hi = _profile(points, t, c, column, edges, cfg["min_points_per_bin"])
            ax.fill_between(x, lo, hi, color=SERIES[t], alpha=0.15, lw=0)
            ax.plot(x, y, color=SERIES[t], marker=MARKERS[t], ms=3.5)
            values += list(y[np.isfinite(y)])
        if len(SHOWN) > 1:
            ax.set_title(f"{data.CAMERAS[c]} camera")
        ax.set_xlabel(xlabel)
    log = max(values) / min(values) > 10        # one shared scale either way; log when one model is far off
    for ax in axes:
        if log:
            ax.set_yscale("log")
            _plain_log(ax.yaxis)
        else:
            ax.set_ylim(bottom=0)
    axes[0].set_ylabel("held-out RMS residual (px)" + (", log scale" if log else ""))
    _legend(fig, techniques, shown, bbox_to_anchor=(0.5, -0.12))
    if title:
        fig.suptitle(title, fontsize=8, color=INK_2)
    save(fig, path_stem)


def radial(points, techniques, cfg, path_stem, subset=None):
    """Binned held-out RMS against normalised radius; all techniques, or only those at the indices in `subset`."""
    edges = np.arange(0, 1 + 1e-9, cfg["radial_bin_width"])
    # No title or zone lines: the report caption says what r is and what the bands are.
    _profile_figure(points, techniques, 8, edges, cfg, "normalised radius r", path_stem, None, shown=subset)


def distance(points, techniques, cfg, path_stem):
    width = cfg["distance_bin_width_mm"]
    d = points[:, 10]
    edges = np.arange(np.floor(d.min() / width) * width, d.max() + width, width)
    close = [t for t in techniques if t.train_distance_mm]

    def bands(ax):
        for t in close:
            ax.axvspan(*t.train_distance_mm, color=GRID, alpha=0.6, lw=0)
            ax.text(np.mean(t.train_distance_mm), 0.02, "close-range\ntraining band", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7, color=INK_2)

    _profile_figure(points, techniques, 10, edges, cfg, "board distance (mm)", path_stem,
                    "Error vs distance (camera to board centre); band: range over splits", bands)


# ====== SCALE BARS ======
def scale(bars, techniques, cfg, path_stem, subset=None):
    """Mean signed and RMS length error of the triangulated bars, binned by board distance; all techniques, or
    only those at the indices in `subset` (a technique tens of mm off would flatten the others)."""
    shown = list(range(len(techniques))) if subset is None else list(subset)
    width = 2 * cfg["distance_bin_width_mm"]
    d = bars[:, 4]
    edges = np.arange(np.floor(d.min() / width) * width, d.max() + width, width)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    for t in shown:
        b = bars[bars[:, 1] == t]
        k = np.digitize(b[:, 4], edges) - 1
        x, bias, rms, lo, hi = [], [], [], [], []
        for i in range(len(edges) - 1):
            sel = k == i
            if sel.sum() < 3:
                continue
            per_split = [b[sel & (b[:, 0] == s), 5].mean() for s in np.unique(b[sel, 0])]
            s = stats.summarise(per_split)
            x.append(0.5 * (edges[i] + edges[i + 1]))
            bias.append(b[sel, 5].mean())
            rms.append(np.sqrt(np.mean(b[sel, 6] ** 2)))
            lo.append(s["lo"])
            hi.append(s["hi"])
        axes[0].fill_between(x, lo, hi, color=SERIES[t], alpha=0.15, lw=0)
        axes[0].plot(x, bias, color=SERIES[t], marker=MARKERS[t], ms=3.5)
        axes[1].plot(x, rms, color=SERIES[t], marker=MARKERS[t], ms=3.5)
    axes[0].axhline(0, color=INK_2, lw=0.8)
    axes[0].set_ylabel("mean length error (mm)")
    axes[1].set_ylabel("RMS length error (mm)")
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.set_xlabel("board distance (mm)")
    lo_mm, hi_mm = cfg["scale_bar_length_mm"]
    fig.suptitle(f"Triangulated board corners, every pair {lo_mm}-{hi_mm} mm apart, held-out stereo pairs",
                 fontsize=8, color=INK_2)
    _legend(fig, techniques, shown, bbox_to_anchor=(0.5, -0.12))
    save(fig, path_stem)


# ====== SUMMARY ======
def summary(table, techniques, metric_list, path_stem):
    """Dot + interval per technique and camera for the headline metrics; one panel per metric."""
    fig, axes = plt.subplots(1, len(metric_list), figsize=(2.3 * len(metric_list), 2.6), constrained_layout=True)
    for ax, (camera_set, key, label) in zip(axes, metric_list):
        ticks, names, values = [], [], []
        y = 0
        for camera in camera_set:
            for t, tech in enumerate(techniques):
                s = stats.summarise(table[(tech.key, camera)][key])
                values += [s["lo"], s["hi"]]
                ax.plot([s["lo"], s["hi"]], [y, y], color=SERIES[t], lw=2, solid_capstyle="round")
                ax.plot(s["mean"], y, marker=MARKERS[t], color=SERIES[t], ms=6, mec="white", mew=1)
                y -= 1
            ticks.append(y + len(techniques) / 2 + 0.5)
            names.append(camera)
            y -= 0.8
        ax.set_yticks(ticks, names)
        ax.grid(axis="y", visible=False)
        values = [v for v in values if np.isfinite(v)]
        if values and min(values) > 0 and max(values) / min(values) > 8:
            ax.set_xscale("log")
            _plain_log(ax.xaxis)
            label += ", log"
        ax.set_title(label)
        if "to" in key:
            ax.axvline(1, color=INK_2, lw=0.8, ls="--")
    _legend(fig, techniques, bbox_to_anchor=(0.5, -0.16))
    save(fig, path_stem)


# ====== PINAX VS THE IN-WATER CALIBRATION ======
def ray_mapping(reference, other, image_size, path_stem, label_ref, label_other, observed, reach_px=30):
    """The angle of the underwater ray each model assigns to the same pixel, against the pixel's radius from
    the principal point. Both models go pixel -> ray, so each stays inside the image it was calibrated on
    (for Pinax: the in-air calibration's own image, then Snell's law). No board pose is involved.

    Top row: the two angle curves. Bottom row: their difference, with the part a focal-length difference
    alone would explain (dashed); the rest is a difference in the SHAPE of the mapping. Only pixels within
    reach_px of an observed corner are used, where the in-water calibration is not extrapolating.
    reference, other: {camera: model}; observed: {camera: (N, 2) detected corners}.
    """
    from scipy.spatial import cKDTree
    fig, axes = plt.subplots(2, len(reference), figsize=(3.6 * len(reference) + (1.2 if len(reference) == 1 else 0), 4.6),
                             squeeze=False, sharex=True, sharey="row",
                             constrained_layout=True)
    angles = np.radians(np.arange(0, 360, 5))
    r = np.arange(12.5, 1000, 25.0)
    for c, camera in enumerate(reference):
        ref, oth = reference[camera], other[camera]
        cx, cy = ref.principal_point
        tree = cKDTree(observed[camera])
        rows = []
        for a in angles:
            px = np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])
            keep = tree.query(px)[0] <= reach_px
            if keep.any():
                t_ref = np.degrees(np.arctan(np.hypot(*ref.normalise(px[keep]).T)))
                t_oth = np.degrees(np.arctan(np.hypot(*oth.normalise(px[keep]).T)))
                rows += list(zip(r[keep], t_ref, t_oth))
        rows = np.array(rows)
        xs = np.unique(rows[:, 0])
        ref_mean = np.array([rows[rows[:, 0] == x, 1].mean() for x in xs])
        oth_mean = np.array([rows[rows[:, 0] == x, 2].mean() for x in xs])
        diff = [rows[rows[:, 0] == x, 2] - rows[rows[:, 0] == x, 1] for x in xs]
        top, bottom = axes[0, c], axes[1, c]
        top.plot(xs, ref_mean, color=SERIES[0], marker=MARKERS[0], ms=3, label=f"{label_ref} (fitted in water)")
        top.plot(xs, oth_mean, color=SERIES[3], marker=MARKERS[3], ms=3, label=f"{label_other} (air + Snell)")
        top.set_title(f"{camera} camera")
        bottom.fill_between(xs, [d.min() for d in diff], [d.max() for d in diff], color=SERIES[3], alpha=0.2, lw=0)
        bottom.plot(xs, [d.mean() for d in diff], color=SERIES[3], marker=MARKERS[3], ms=3,
                    label=f"{label_other} - {label_ref}")
        f_ref, f_oth = ref.K[0, 0], oth.K[0, 0] * getattr(oth, "n_water", 1.0)
        bottom.plot(xs, np.degrees(np.arctan(xs / f_oth) - np.arctan(xs / f_ref)), color=INK_2, ls="--", lw=1.2,
                    label="focal-length difference alone")
        bottom.axhline(0, color=INK_2, lw=0.8)
        bottom.set_xlabel("radius from principal point (px)")
    axes[0, 0].set_ylabel("underwater ray angle (deg)")
    axes[1, 0].set_ylabel("difference (deg)")
    axes[0, 0].legend(loc="upper left", frameon=False)
    axes[1, 0].legend(loc="lower left", frameon=False)
    fig.suptitle("Underwater ray angle per pixel, where corners were observed; shaded = range over directions",
                 fontsize=8, color=INK_2)
    save(fig, path_stem)


# ====== RESIDUAL VECTORS (QUIVER) ======
def quiver(points, techniques, cfg, image_size, params_all, path_stem, subset=None, gains=None):
    """Mean held-out residual vector (observed - reprojected) per image cell, one panel per technique and
    camera. Systematic error shows as coherent arrow fields (radial: distortion/focal; uniform: principal
    point); noise averages out. Each panel is titled with its RMS and arrow gain. Cells with fewer than
    min_points_per_cell points are empty.

    gains: {technique index: arrow gain}, for techniques whose errors differ by orders of magnitude (the
    in-air ones against those fitted in water). There is no arrow key: the caption must state the gain.
    Better still, draw each gain's techniques as a figure of their own (`subset`), so arrows at
    different gains are never side by side. Without it, ONE gain is chosen for the whole figure, set by the
    90th-percentile cell so a few large cells do not shrink every other arrow to a dot."""
    w, h = image_size
    nx, ny = cfg.get("quiver_cells", cfg["grid_cells"])
    shown = list(range(len(techniques))) if subset is None else list(subset)
    grids = {(col, c): metrics.grid_stats(*_select(points, t, c)[:, 4:8].T, image_size, (nx, ny),
                                          cfg["min_points_per_cell"])
             for col, t in enumerate(shown) for c in SHOWN}
    # The gain is set by the 90th-percentile cell, so a few large cells (drawn longer than a cell) do not
    # shrink every other arrow to a dot.
    if gains is None:
        lengths = np.concatenate([np.hypot(g["mean_du"], g["mean_dv"])[g["valid"]] for g in grids.values()])
        gains = dict.fromkeys(shown, _nice(0.9 * (w / nx) / float(np.percentile(lengths, 90))))
    X, Y = np.meshgrid((np.arange(nx) + 0.5) * w / nx, (np.arange(ny) + 0.5) * h / ny)
    # At most 4 techniques across; each technique is a column with its cameras stacked (left above right).
    ncol = min(4, len(shown))
    per = len(SHOWN)
    nrow = per * int(np.ceil(len(shown) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.45 * nrow), squeeze=False, constrained_layout=True)
    for ax in axes.ravel():
        ax.axis("off")
    for (col, c), g in grids.items():
        t = shown[col]
        row = SHOWN.index(c)
        ax = axes[per * (col // ncol) + row, col % ncol]
        ax.axis("on")
        sel = _select(points, t, c)
        v = g["valid"]
        gain = gains[t]
        ax.quiver(X[v], Y[v], g["mean_du"][v] * gain, g["mean_dv"][v] * gain, angles="xy", scale_units="xy",
                  scale=1, color=SERIES[t], width=0.011, headwidth=3, headlength=3.5, headaxislength=3,
                  minlength=0.5)
        K = np.array(params_all[techniques[t].key][data.CAMERAS[c]]["camera_matrix"])
        ax.plot(K[0, 2], K[1, 2], "+", color=INK, ms=7)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect("equal")
        ax.set_xticks([0, w // 2, w])
        ax.set_yticks([0, h // 2, h])
        ax.grid(False)
        if col % ncol == 0:
            ax.set_ylabel("v (px)")
        if row == per - 1:
            ax.set_xlabel("u (px)")
        ax.set_title((techniques[t].label + "\n" if row == 0 else "")
                     + f"{data.CAMERAS[c]}: RMS {metrics.rms(sel[:, 6], sel[:, 7]):.2f} px", fontsize=7.5)
    save(fig, path_stem)
    return {"arrow_gains": {techniques[t].key: float(gains[t]) for t in shown}, "cells": [nx, ny]}


# ====== SEPARATE TARGET (WaterComparison tags) ======
def target_errors(tag_rows, techniques, size_mm, calib_range_mm, path_stem, subset=None):
    """Tag error (median edge - printed size) against distance, one point per tag, per technique; the
    shaded band is the distance range of the calibration boards. Points of a technique are nudged
    sideways a little so overlapping techniques stay visible."""
    shown = list(range(len(techniques))) if subset is None else list(subset)
    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    ax.axvspan(calib_range_mm[0] / 1000, calib_range_mm[1] / 1000, color=GRID, alpha=0.7, lw=0)
    ax.text(calib_range_mm[1] / 1000, 0.02, "calibration boards ", transform=ax.get_xaxis_transform(),
            ha="right", va="bottom", fontsize=7, color=INK_2)
    ax.axhline(0, color=INK_2, lw=0.8)
    for k, t in enumerate(shown):
        rows = tag_rows[techniques[t].key]
        if not rows:
            continue
        d = np.array([r["distance_mm"] for r in rows]) / 1000 + (k - (len(shown) - 1) / 2) * 0.015
        e = np.array([r["error_mm"] for r in rows])
        ax.scatter(d, e, s=22, color=SERIES[t], marker=MARKERS[t], edgecolor="white", linewidth=0.6, zorder=3)
        labels = sorted({r["label_cm"] for r in rows})
        med = [(np.mean([r["distance_mm"] for r in rows if r["label_cm"] == lab]) / 1000,
                np.median([r["error_mm"] for r in rows if r["label_cm"] == lab])) for lab in labels]
        ax.plot(*zip(*med), color=SERIES[t], lw=1.2, alpha=0.8)
    ax.set_xlabel("distance from left camera to tag (m)")
    ax.set_ylabel(f"tag error (mm): median edge - {size_mm:g} mm")
    ax.grid(alpha=0.6)
    handles = [plt.Line2D([], [], color=SERIES[t], marker=MARKERS[t], ms=5, label=techniques[t].label) for t in shown]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(shown)), frameon=False, bbox_to_anchor=(0.5, -0.14))
    fig.suptitle("WaterComparison target, the same detected corners for every technique; line = median per distance label",
                 fontsize=8, color=INK_2)
    save(fig, path_stem)


# ====== PARAMETER CORRELATION ======
def correlation(cov_results, path_stem):
    """Correlation matrices of the intrinsics, one panel per (technique, camera). cov_results:
    [(label, camera, covariance.intrinsic_covariance result)]. Values printed in each cell."""
    n = len(cov_results)
    cols = min(2, n)          # two across, so the figure fills a portrait page
    rows_ = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(3.0 * cols, 3.0 * rows_), squeeze=False, constrained_layout=True)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, (label, camera, res) in zip(axes.ravel(), cov_results):
        C = np.array(res["corr"])
        im = ax.imshow(C, cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
        m = len(res["names"])
        for i in range(m):
            for j in range(m):
                ax.text(j, i, f"{C[i, j]:.2f}".replace("0.", ".").replace("-.", "-."), ha="center", va="center",
                        fontsize=5 if m > 9 else 5.5, color="white" if abs(C[i, j]) > 0.7 else INK)
        ax.set_xticks(range(m), res["names"], rotation=90, fontsize=6)
        ax.set_yticks(range(m), res["names"], fontsize=6)
        ax.grid(False)
        ax.set_title(f"{label}\n{camera} camera", fontsize=7.5)
    fig.colorbar(im, ax=axes, shrink=0.6, label="correlation")
    save(fig, path_stem)

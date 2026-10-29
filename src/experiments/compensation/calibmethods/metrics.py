"""Error metrics computed from reprojection residuals. Pure NumPy, no OpenCV, so each can be tested on synthetic
residuals with known errors (tests/test_metrics.py).
"""
import numpy as np


def error_norm(du, dv):
    return np.hypot(np.asarray(du, float), np.asarray(dv, float))


def rms(du, dv):
    """Root mean square of the residual length; nan for no points."""
    e2 = np.asarray(du, float) ** 2 + np.asarray(dv, float) ** 2
    return float(np.sqrt(e2.mean())) if e2.size else float("nan")


def normalised_radius(u, v, principal_point, image_size):
    """Distance of each point from the principal point, divided by the distance from the principal point to the farthest
    image corner, so 0 is the principal point and 1 the farthest corner of the frame."""
    cx, cy = principal_point
    w, h = image_size
    r_max = max(np.hypot(x - cx, y - cy) for x in (0, w) for y in (0, h))
    return np.hypot(np.asarray(u, float) - cx, np.asarray(v, float) - cy) / r_max


# ====== ZONES ======
ZONES = ("centre", "mid", "outer")


def zone_of(r, radii):
    """Zone index per point: 0 centre (r < r1), 1 mid (r1 <= r < r2), 2 outer (r >= r2)."""
    r1, r2 = radii
    if not 0 < r1 < r2:
        raise ValueError(f"zone radii must satisfy 0 < r1 < r2, got {radii}")
    return np.digitize(np.asarray(r, float), [r1, r2])


def zone_rms(du, dv, r, radii):
    """{"centre"|"mid"|"outer": {"rms", "n"}} plus "outer_to_centre" and "mid_to_centre" ratios."""
    du, dv = np.asarray(du, float), np.asarray(dv, float)
    zone = zone_of(r, radii)
    out = {}
    for k, name in enumerate(ZONES):
        sel = zone == k
        out[name] = {"rms": rms(du[sel], dv[sel]), "n": int(sel.sum())}
    centre = out["centre"]["rms"]
    for name in ("outer", "mid"):
        out[f"{name}_to_centre"] = out[name]["rms"] / centre if centre > 0 else float("nan")
    return out


# ====== BINNED PROFILES ======
def binned_rms(x, du, dv, edges, min_count=1):
    """RMS residual in each bin of x (radius, distance, ...): (centres, rms, n)."""
    x = np.asarray(x, float)
    e2 = np.asarray(du, float) ** 2 + np.asarray(dv, float) ** 2
    edges = np.asarray(edges, float)
    k = np.digitize(x, edges) - 1                   # bin index; out of range is -1 or len(edges)-1
    n = np.bincount(k[(k >= 0) & (k < len(edges) - 1)], minlength=len(edges) - 1)
    s = np.bincount(k[(k >= 0) & (k < len(edges) - 1)], weights=e2[(k >= 0) & (k < len(edges) - 1)],
                    minlength=len(edges) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        value = np.sqrt(s / n)
    value[n < max(min_count, 1)] = np.nan
    return 0.5 * (edges[:-1] + edges[1:]), value, n


# ====== IMAGE GRID ======
def grid_stats(u, v, du, dv, image_size, cells, min_count=5):
    """Residuals gathered into a cells[0] x cells[1] (across x down) grid over the image."""
    nx, ny = cells
    w, h = image_size
    u, v = np.asarray(u, float), np.asarray(v, float)
    du, dv = np.asarray(du, float), np.asarray(dv, float)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    col = np.minimum((u[inside] * nx / w).astype(int), nx - 1)
    row = np.minimum((v[inside] * ny / h).astype(int), ny - 1)
    flat = row * nx + col

    def total(weights):
        return np.bincount(flat, weights=weights, minlength=nx * ny).reshape(ny, nx)

    n = np.bincount(flat, minlength=nx * ny).reshape(ny, nx)
    with np.errstate(invalid="ignore", divide="ignore"):
        stats = {"mean_error": total(error_norm(du[inside], dv[inside])) / n,
                 "mean_du": total(du[inside]) / n,
                 "mean_dv": total(dv[inside]) / n}
    valid = n >= min_count
    for a in stats.values():
        a[~valid] = np.nan
    return {**stats, "n": n, "valid": valid}

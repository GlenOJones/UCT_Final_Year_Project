"""Intervals over repeated train/validation splits, and the test for whether two models differ."""
import numpy as np

INTERVAL = (2.5, 97.5)


def summarise(values):
    """{"mean", "lo", "hi", "n"} over the finite values (one per split)."""
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    lo, hi = np.percentile(x, INTERVAL)
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": int(x.size)}


def paired_difference(a, b):
    """Summary of a - b per split (splits where either is nan are skipped), plus "within_noise"."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired values need one entry per split for both models")
    s = summarise(a - b)
    s["within_noise"] = bool(s["n"] == 0 or s["lo"] <= 0 <= s["hi"])
    return s

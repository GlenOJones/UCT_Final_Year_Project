"""Intervals over repeated train/validation splits, and the test for whether two models differ.

Each split holds out different recordings, so a metric has one value per split. Its interval is the
2.5-97.5 percentile range of those values: how much the number depends on WHICH recordings were held
out. The splits share training data, so they are not independent resamples; this range is wider
(more cautious) than a bootstrap of their mean would be, and it is the honest question here.

Two models are compared PAIRED: on each split both were trained on the same recordings and scored on
the same held-out frames, so their per-split difference removes the variation that comes from the
frames themselves. A difference is "within noise" when its interval contains zero.
"""
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

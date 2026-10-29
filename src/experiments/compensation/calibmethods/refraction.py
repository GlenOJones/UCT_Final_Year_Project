"""Flat-port refraction as a radial mapping of normalised image coordinates (Pinax: Luczynski, Pfingsthorn & Birk 2017,
Ocean Engineering 133, 9-22).
"""
import numpy as np


def refract_radial(r, n_w):
    """Normalised image radius in air -> normalised radius of the same ray in water (tan of its angle)."""
    return np.tan(np.arcsin(np.clip(np.sin(np.arctan(r)) / n_w, -1.0, 1.0)))


def unrefract_radial(r_w, n_w):
    """Inverse of refract_radial: water-side radius -> air-side radius."""
    s = n_w * np.sin(np.arctan(r_w))
    return np.where(np.abs(s) <= 1.0, np.tan(np.arcsin(np.clip(s, -1.0, 1.0))), np.nan)

"""Flat-port refraction as a radial mapping of normalised image coordinates (Pinax: Luczynski,
Pfingsthorn & Birk 2017, Ocean Engineering 133, 9-22).

A camera behind a flat pane, with the optical axis normal to it, sees every ray bent by Snell's law at
the pane: sin(theta_air) = n_w sin(theta_water). The glass index cancels, so the ray DIRECTION in water
depends only on n_w, and by symmetry about the axis the correction is a function of image radius alone.
These are the only Pinax pieces calib_methods uses; the full 3-medium ray trace and the virtual-centre
search are in src/archive/pinax_18sep.py, which imports these two from here.
"""
import numpy as np


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

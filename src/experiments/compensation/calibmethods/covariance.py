"""Covariance and correlation of a fitted model's intrinsic parameters.

    A = sum_k [ Jp_k^T Jp_k - Jp_k^T Jq_k (Jq_k^T Jq_k)^-1 Jq_k^T Jp_k ]
"""
import numpy as np

from . import models


def intrinsic_covariance(model, frames):
    """{"names", "values", "sigma", "cov", "corr", "dof", "sigma_px"} of the model's intrinsics, at its current
    parameters, over the given frames (each posed with the intrinsics fixed)."""
    names, p0 = models.param_names(model), models.get_params(model)
    m = len(p0)
    steps = 1e-6 * np.maximum(1.0, np.abs(p0))
    plus = [models.with_params(model, p0 + np.eye(m)[i] * steps[i]) for i in range(m)]
    minus = [models.with_params(model, p0 - np.eye(m)[i] * steps[i]) for i in range(m)]

    A = np.zeros((m, m))
    ssr, n_res = 0.0, 0
    for f in frames:
        rvec, tvec = model.pose(f.obj, f.img)
        img, Jq = model.project_jac(f.obj, rvec, tvec)
        r = (img - f.img).ravel()
        ssr += float(r @ r)
        n_res += r.size
        Jp = np.column_stack([(plus[i].project(f.obj, rvec, tvec) - minus[i].project(f.obj, rvec, tvec)).ravel()
                              / (2 * steps[i]) for i in range(m)])
        Hqq = Jq.T @ Jq
        Hpq = Jp.T @ Jq
        A += Jp.T @ Jp - Hpq @ np.linalg.solve(Hqq, Hpq.T)
    dof = n_res - m - 6 * len(frames)
    s2 = ssr / dof
    cov = s2 * np.linalg.inv(A)
    sigma = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sigma, sigma)
    return {"names": names, "values": p0.tolist(), "sigma": sigma.tolist(), "cov": cov.tolist(),
            "corr": corr.tolist(), "dof": int(dof), "sigma_px": float(np.sqrt(s2)), "n_frames": len(frames)}

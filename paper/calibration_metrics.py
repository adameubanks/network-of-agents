#!/usr/bin/env python3
import numpy as np
from typing import Dict


def fit_centered_calibration(opinions: np.ndarray, ratings: np.ndarray) -> Dict[str, float]:
    x = np.asarray(opinions, dtype=float).reshape(-1) - 0.5
    y = np.asarray(ratings, dtype=float).reshape(-1) - 0.5
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]
    if x.size == 0:
        return {"alpha_c": np.nan, "beta_c": np.nan, "r2": np.nan, "resid_std": np.nan}
    x_mat = np.vstack([np.ones_like(x), x]).T
    coef, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
    alpha_c, beta_c = float(coef[0]), float(coef[1])
    y_hat = x_mat @ coef
    resid = y - y_hat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    resid_std = float(np.sqrt(ss_res / max(len(y) - 2, 1)))
    return {"alpha_c": alpha_c, "beta_c": beta_c, "r2": r2, "resid_std": resid_std}


def rmse_calibration(opinions: np.ndarray, ratings: np.ndarray) -> float:
    o = np.asarray(opinions, dtype=float).reshape(-1)
    r = np.asarray(ratings, dtype=float).reshape(-1)
    valid = np.isfinite(o) & np.isfinite(r)
    o = o[valid]
    r = r[valid]
    if o.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((r - o) ** 2)))


def summarize_calibration(opinions: np.ndarray, ratings: np.ndarray) -> Dict[str, float]:
    lin = fit_centered_calibration(opinions, ratings)
    rmse = rmse_calibration(opinions, ratings)
    return {
        "alpha_c": lin["alpha_c"],
        "beta_c": lin["beta_c"],
        "r2_centered": lin["r2"],
        "rmse_calib": rmse,
    }



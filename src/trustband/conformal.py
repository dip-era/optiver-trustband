"""Conformal prediction bands: static split, adaptive (ACI) and per-trust-bucket (Mondrian).

All three are one function: observations arrive in daily batches; each group keeps its own buffer
of normalised scores s = |y - y_hat| / sigma_hat and its own running alpha_t.

  split conformal : adaptive=False      (quantile of the calibration scores, never updated)
  ACI             : adaptive=True       (Gibbs & Candes 2021: alpha_t += gamma * (alpha - miss_t))
  Mondrian        : groups=<bucket ids> (separate buffer + alpha_t per bucket)
"""
from __future__ import annotations

from collections import deque

import numpy as np


def online_conformal(dates, y, yhat, sigma, cal_scores, cal_groups=None, groups=None, *,
                     alpha: float = 0.10, gamma: float = 0.02, window: int = 4000,
                     adaptive: bool = True, min_group: int = 50) -> np.ndarray:
    """Return the half-width of the band for each test observation (in units of y).

    dates, y, yhat, sigma : test arrays, aligned; dates define the daily batches (processed in order)
    cal_scores            : normalised calibration scores used to initialise every buffer
    cal_groups, groups    : optional bucket ids for calibration / test rows (Mondrian)
    """
    dates, y, yhat, sigma = map(np.asarray, (dates, y, yhat, sigma))
    cal_scores = np.asarray(cal_scores, float)
    groups = np.zeros(len(y), int) if groups is None else np.asarray(groups)
    cal_groups = np.zeros(len(cal_scores), int) if cal_groups is None else np.asarray(cal_groups)

    buf, a_t = {}, {}
    for g in np.unique(np.concatenate([groups, cal_groups])):
        own = cal_scores[cal_groups == g]
        # a bucket with too few calibration scores borrows the pooled set
        buf[g] = deque((own if len(own) >= min_group else cal_scores)[-window:], maxlen=window)
        a_t[g] = alpha

    half = np.empty(len(y))
    for d in np.unique(dates):
        idx = np.where(dates == d)[0]
        for g in np.unique(groups[idx]):
            j = idx[groups[idx] == g]
            b = np.fromiter(buf[g], float)
            level = 1 - a_t[g]
            if level >= 1:
                q = np.inf
            elif level <= 0:
                q = 0.0
            else:
                q = np.quantile(b, min(level * (1 + 1 / len(b)), 1.0))
            half[j] = q * sigma[j]
            if adaptive:
                s = np.abs(y[j] - yhat[j]) / sigma[j]
                a_t[g] += gamma * (alpha - np.mean(s > q))
                buf[g].extend(s)
    return half

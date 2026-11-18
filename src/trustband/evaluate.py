"""Metrics: coverage (overall, rolling, stress windows), band width, rank correlations."""
from __future__ import annotations

import numpy as np
import pandas as pd


def coverage_stats(dates, y, yhat, half, stress_windows: dict, label: str, roll_days: int = 20):
    """Return (summary dict, rolling daily coverage series)."""
    hit = np.abs(np.asarray(y) - np.asarray(yhat)) <= np.asarray(half)
    s = pd.Series(hit, index=pd.DatetimeIndex(dates))
    daily = s.groupby(level=0).mean()
    roll = daily.rolling(roll_days).mean()
    out = {"method": label, "coverage": hit.mean(), "width_pct": 2 * np.median(half),
           f"worst_{roll_days}d_cov": roll.min()}
    for name, (a, b) in stress_windows.items():
        out[f"cov_{name}"] = s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))].mean()
    return out, roll


def daily_rank_ic(df: pd.DataFrame, pred: str, target: str) -> float:
    """Mean over days of the cross-sectional Spearman correlation between pred and target."""
    return df.groupby("date").apply(lambda g: g[pred].corr(g[target], method="spearman")).mean()


def attention_spike_uplift(df: pd.DataFrame, col: str, q: float = 0.95, n_boot: int = 500) -> dict:
    """Next-day |move| / own 20d vol on top-(1-q) attention days vs the rest, with a bootstrap CI."""
    t = df[df.has_tw == 1].assign(ay_n=lambda x: np.abs(x.y) / x.rv20)
    top = t[col] > t[col].quantile(q)
    uplift = t.ay_n[top].mean() / t.ay_n[~top].mean() - 1
    boot = [t.ay_n[top].sample(frac=1, replace=True, random_state=b).mean()
            / t.ay_n[~top].sample(frac=1, replace=True, random_state=b).mean() - 1 for b in range(n_boot)]
    return {"uplift": float(uplift), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}

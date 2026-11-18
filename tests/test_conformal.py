"""Sanity checks for online_conformal on synthetic data where the right answer is known."""
import numpy as np

from trustband.conformal import online_conformal

ALPHA = 0.10


def _days(n_days, per_day):
    return np.repeat(np.arange(n_days), per_day)


def test_split_conformal_covers_90pct_on_exchangeable_data():
    rng = np.random.default_rng(0)
    cal = np.abs(rng.normal(size=5000))
    dates = _days(200, 50)
    y = rng.normal(size=len(dates))
    half = online_conformal(dates, y, np.zeros_like(y), np.ones_like(y), cal, adaptive=False, alpha=ALPHA)
    cov = np.mean(np.abs(y) <= half)
    assert abs(cov - 0.90) < 0.015


def test_aci_recovers_coverage_after_a_volatility_shock():
    rng = np.random.default_rng(1)
    cal = np.abs(rng.normal(size=4000))
    dates = _days(300, 50)
    scale = np.where(dates < 100, 1.0, 2.5)  # noise jumps 2.5x after day 100
    y = rng.normal(size=len(dates)) * scale
    args = (dates, y, np.zeros_like(y), np.ones_like(y), cal)
    split = online_conformal(*args, adaptive=False, alpha=ALPHA)
    aci = online_conformal(*args, adaptive=True, alpha=ALPHA, gamma=0.02, window=4000)
    after = dates >= 100
    split_cov = np.mean(np.abs(y[after]) <= split[after])
    aci_cov = np.mean(np.abs(y[after]) <= aci[after])
    assert split_cov < 0.70            # static band breaks
    assert abs(aci_cov - 0.90) < 0.03  # adaptive band recovers


def test_mondrian_buckets_cover_each_group():
    rng = np.random.default_rng(2)
    n_cal = 6000
    cal_groups = rng.integers(0, 2, n_cal)
    cal = np.abs(rng.normal(size=n_cal)) * np.where(cal_groups == 1, 3.0, 1.0)
    dates = _days(200, 50)
    groups = rng.integers(0, 2, len(dates))
    y = rng.normal(size=len(dates)) * np.where(groups == 1, 3.0, 1.0)
    args = (dates, y, np.zeros_like(y), np.ones_like(y), cal)
    pooled = online_conformal(*args, adaptive=False, alpha=ALPHA)
    mond = online_conformal(*args, cal_groups=cal_groups, groups=groups, adaptive=False, alpha=ALPHA)
    hit_pooled, hit_mond = np.abs(y) <= pooled, np.abs(y) <= mond
    # pooled bands are right on average but wrong inside each group
    assert hit_pooled[groups == 1].mean() < 0.85
    for g in (0, 1):
        assert abs(hit_mond[groups == g].mean() - 0.90) < 0.02

"""Forecast model y_hat(x) and error-size model sigma_hat(x), both LightGBM."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd


def fit_mean_and_scale(train: pd.DataFrame, feats: list[str], params: dict, n_folds: int = 4):
    """Mean model (Huber loss) + scale model sigma(x) ~ E|y - y_hat| trained on out-of-fold errors.

    Out-of-fold: training months are split into contiguous blocks and each block's residuals come
    from a model that never saw it. In-sample residuals are too small (the model has memorised
    them), which would make every band too narrow.
    """
    months = train["date"].dt.to_period("M")
    folds = np.array_split(np.sort(months.unique()), n_folds)
    oof = np.full(len(train), np.nan)
    for block in folds:
        held = months.isin(block).values
        m = lgb.LGBMRegressor(objective="huber", alpha=2.0, **params)
        m.fit(train.loc[~held, feats], train.loc[~held, "y"])
        oof[held] = m.predict(train.loc[held, feats])
    mean = lgb.LGBMRegressor(objective="huber", alpha=2.0, **params).fit(train[feats], train["y"])
    scale = lgb.LGBMRegressor(objective="l2", **params).fit(train[feats], np.abs(train["y"] - oof))
    return mean, scale


def predict(models, df: pd.DataFrame, feats: list[str], min_sigma: float = 0.05):
    mean, scale = models
    return mean.predict(df[feats]), np.clip(scale.predict(df[feats]), min_sigma, None)

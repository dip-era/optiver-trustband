"""Per stock-day features: price history plus naive and trust-weighted text aggregates."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trustband.trust import trust_weight

PRICE = ["r", "r_l1", "r5", "rv5", "rv20", "hl", "vol_z", "mkt_r", "mkt_rv5"]
NAIVE = PRICE + ["has_tw", "log_n", "attn", "sent_mean", "sent_std", "pos_frac", "neg_frac"]
TRUST = PRICE + ["has_tw", "eff_n", "tattn", "tsent_mean", "trust", "dup_share", "auto_share", "new_share"]
FEATURE_SETS = {"price": PRICE, "naive": NAIVE, "trust": TRUST}


def price_features(px: pd.DataFrame) -> pd.DataFrame:
    """Returns in %, target y = next-day close-to-close log return, volatility and volume features."""
    px = px.sort_values(["ticker", "date"]).copy()
    px["r"] = 100 * np.log(px["adj_close"]).groupby(px["ticker"]).diff()
    px["y"] = px.groupby("ticker")["r"].shift(-1)
    px["r_l1"] = px.groupby("ticker")["r"].shift(1)
    px["r5"] = px.groupby("ticker")["r"].transform(lambda s: s.rolling(5).sum())
    px["rv5"] = px.groupby("ticker")["r"].transform(lambda s: s.rolling(5).std())
    px["rv20"] = px.groupby("ticker")["r"].transform(lambda s: s.rolling(20).std())
    px["hl"] = 100 * np.log(px["High"] / px["Low"])
    lv = np.log1p(px["Volume"])
    mean20 = lv.groupby(px["ticker"]).transform(lambda s: s.rolling(20).mean().shift(1))
    std20 = lv.groupby(px["ticker"]).transform(lambda s: s.rolling(20).std().shift(1))
    px["vol_z"] = (lv - mean20) / std20
    mkt = px.groupby("date")["r"].mean().rename("mkt_r").to_frame()
    mkt["mkt_rv5"] = mkt["mkt_r"].rolling(5).std()
    return px.merge(mkt, on="date", how="left")


def text_features(tw: pd.DataFrame, panel_index: pd.MultiIndex, trust_cfg: dict) -> pd.DataFrame:
    """Aggregate tweets to (ticker, date): naive counts/sentiment and their trust-weighted versions."""
    tw = tw.copy()
    tw["w"] = trust_weight(tw, trust_cfg)
    tw["ws"] = tw["w"] * tw["sent"]
    tw["new"] = tw["acct_age_days"] < trust_cfg["new_account_days"]
    g = tw.groupby(["ticker", "date"])
    f = pd.DataFrame({
        "n": g.size(),
        "sent_mean": g["sent"].mean(),
        "sent_std": g["sent"].std(),
        "pos_frac": g["sent"].apply(lambda s: (s > 0.5).mean()),
        "neg_frac": g["sent"].apply(lambda s: (s < -0.5).mean()),
        "wsum": g["w"].sum(),
        "wssum": g["ws"].sum(),
        "dup_share": 1 - g["norm"].nunique() / g.size(),
        "auto_share": g["automated_client"].mean(),
        "new_share": g["new"].mean(),
    })
    f["tsent_mean"] = f["wssum"] / f["wsum"]
    f["trust"] = f["wsum"] / f["n"]
    f = f.reindex(panel_index)
    f["has_tw"] = f["n"].notna().astype(float)
    f["log_n"] = np.log1p(f["n"].fillna(0))
    f["eff_n"] = np.log1p(f["wsum"].fillna(0))
    # abnormal attention: today's (log) volume minus the stock's own trailing 20-day mean
    for src, dst in [("log_n", "attn"), ("eff_n", "tattn")]:
        base = f[src].groupby(level="ticker").transform(lambda s: s.rolling(20, min_periods=5).mean().shift(1))
        f[dst] = f[src] - base
    text_cols = sorted(set(NAIVE + TRUST) - set(PRICE))
    return f[text_cols]


def build_panel(px: pd.DataFrame, tw: pd.DataFrame, start: str, end: str, trust_cfg: dict) -> pd.DataFrame:
    """One row per (ticker, date) in [start, end] with target y, price and text features."""
    panel = price_features(px)
    panel = panel[(panel.date >= start) & (panel.date <= end)].dropna(subset=["y"] + PRICE)
    pidx = pd.MultiIndex.from_frame(panel[["ticker", "date"]])
    text = text_features(tw, pidx, trust_cfg)
    return pd.concat([panel.set_index(["ticker", "date"]), text], axis=1).reset_index()

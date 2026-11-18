"""Per-tweet credibility weights.

v0 is the hand-set heuristic from the pilot: each red flag multiplies the weight down, and
near-identical copies of the same text on the same stock-day share a single vote.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def trust_weight(tw: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Heuristic credibility in (0, 1] for each tweet row."""
    dup = tw.groupby(["ticker", "date", "norm"])["norm"].transform("size")
    w = np.ones(len(tw))
    w *= np.where(tw["automated_client"], cfg["automated_client"], 1.0)
    w *= np.where(tw["acct_age_days"] < cfg["new_account_days"], cfg["new_account"], 1.0)
    w *= np.where(tw["n_cashtags"] >= cfg["cashtag_spam_min"], cfg["cashtag_spam"], 1.0)
    w *= np.where(tw["followers"] < cfg["low_followers_max"], cfg["low_followers"], 1.0)
    return w / dup.values

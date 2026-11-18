"""Coordinated pump/dump injection attacks on the tweet stream (evaluation only)."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Pump-and-dump / short-and-distort style posts. {T} = ticker.
ATTACK_TEMPLATES = {
    "pump": [
        "${T} is about to explode!!! buyout rumors confirmed, get in now",
        "Huge news coming for ${T}, insiders are loading up, easy +40% this week",
        "${T} short squeeze incoming, do not miss this rocket",
        "BREAKING: ${T} in acquisition talks, massive upside, strong buy",
        "${T} going to the moon, analysts upgrading, easy double from here",
        "Just loaded more ${T}, record earnings leak says blowout quarter",
    ],
    "dump": [
        "${T} accounting fraud exposed, SEC investigation coming, sell now",
        "${T} is collapsing, insiders dumping shares, get out before it is too late",
        "BREAKING: ${T} CEO resigning amid scandal, stock will crash",
        "${T} massive guidance cut leaked, short this to zero",
        "Bankruptcy rumors for ${T} spreading, terrible news for holders",
        "${T} earnings disaster incoming, heavy downgrade expected",
    ],
}

# columns an injected post must carry for text_features() / trust_weight()
POST_COLUMNS = ["ticker", "date", "sent", "norm", "automated_client", "acct_age_days", "followers", "n_cashtags"]


def template_frame() -> pd.DataFrame:
    rows = [(kind, t) for kind, ts in ATTACK_TEMPLATES.items() for t in ts]
    return pd.DataFrame(rows, columns=["kind", "template"])


def inject_attack(tw: pd.DataFrame, target_days: pd.DataFrame, kind: str, templates: pd.DataFrame,
                  rng: np.random.Generator, fraction: float = 0.10, min_posts: int = 20,
                  target_seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add coordinated pump/dump posts to a random `fraction` of the (ticker, date) rows in target_days.

    naive  : brand-new low-follower accounts, automated client, near-identical text
    stealth: aged accounts with followers, human client, every post a unique paraphrase

    templates needs columns kind, template, sent (FinBERT score of the template).
    Returns (tweets + injected posts, attacked days with their direction).
    """
    tgt = target_days[["ticker", "date"]].sample(frac=fraction, random_state=target_seed).reset_index(drop=True)
    tgt["dir"] = rng.choice(["pump", "dump"], len(tgt))
    counts = tw.groupby(["ticker", "date"]).size()
    stealth = kind == "stealth"
    rows = []
    for t, d, direction in tgt.itertuples(index=False):
        k = int(max(min_posts, counts.get((t, d), 0)))  # at least double the day's real chatter
        picked = templates[templates.kind == direction].sample(k, replace=True, random_state=rng.integers(1e9))
        for i, (_, row) in enumerate(picked.iterrows()):
            rows.append(dict(
                ticker=t, date=d, sent=row["sent"],
                norm=f"{row['template']} #{i}" if stealth else row["template"],
                automated_client=not stealth,
                acct_age_days=rng.uniform(400, 2500) if stealth else rng.uniform(1, 30),
                followers=rng.integers(100, 3000) if stealth else rng.integers(0, 20),
                n_cashtags=1 if stealth else rng.integers(1, 6),
            ))
    fake = pd.DataFrame(rows)
    return pd.concat([tw[POST_COLUMNS], fake[POST_COLUMNS]], ignore_index=True), tgt

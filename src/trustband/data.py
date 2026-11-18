"""Load StockNet prices and raw tweets straight from the zip, and align tweets to trading days.

Reading from the zip avoids unpacking ~55k small files, which is very slow on Windows.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
import zipfile
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = "stocknet-dataset-master/"

# Clients a human types into; everything else (feeds, bots, schedulers) counts as automated.
HUMAN_CLIENTS = {
    "Twitter Web Client", "Twitter for iPhone", "Twitter for Android", "Twitter for iPad",
    "TweetDeck", "Twitter for Mac", "Mobile Web", "Mobile Web (M2)", "Mobile Web (M5)",
    "Tweetbot for iOS", "Tweetbot for Mac", "Twitter for Windows Phone", "Echofon",
    "Twitter for BlackBerry", "Twitter for Android Tablets", "Twitter Web App", "Fenix for Android",
    "Talon (Plus)", "Tweetlogix", "Janetter",
}
URL_RE = re.compile(r"https?://\S+")
CASHTAG_RE = re.compile(r"\$[A-Za-z][A-Za-z.\-]{0,5}\b")
SOURCE_RE = re.compile(r">([^<]*)<")
TWITTER_TIME = "%a %b %d %H:%M:%S %z %Y"


def download_stocknet(url: str, dest: Path) -> Path:
    """Fetch the StockNet repository zip (~100 MB) unless it is already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"downloading StockNet to {dest} ...")
        urllib.request.urlretrieve(url, dest)
    return dest


def norm_text(t: str) -> str:
    """Normalise for near-duplicate detection: drop URLs, cashtags, digits, punctuation."""
    t = URL_RE.sub("", t.lower())
    t = CASHTAG_RE.sub("", t)
    t = re.sub(r"[^a-z ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def load_prices(z: zipfile.ZipFile) -> pd.DataFrame:
    frames = []
    for name in z.namelist():
        if name.startswith(ROOT + "price/raw/") and name.endswith(".csv"):
            df = pd.read_csv(StringIO(z.read(name).decode()), parse_dates=["Date"])
            df["ticker"] = os.path.basename(name)[:-4]
            frames.append(df)
    px = pd.concat(frames, ignore_index=True).rename(columns={"Date": "date", "Adj Close": "adj_close"})
    px = px.dropna(subset=["adj_close"]).sort_values(["ticker", "date"])
    return px[["ticker", "date", "Open", "High", "Low", "Close", "adj_close", "Volume"]].reset_index(drop=True)


def load_tweets(z: zipfile.ZipFile) -> pd.DataFrame:
    rows = []
    for name in z.namelist():
        if not name.startswith(ROOT + "tweet/raw/") or name.endswith("/"):
            continue
        ticker = name.split("/")[-2]
        for line in z.read(name).decode("utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                tw = json.loads(line)
            except json.JSONDecodeError:
                continue
            u = tw.get("user") or {}
            src = SOURCE_RE.search(tw.get("source") or "")
            text = tw.get("text") or ""
            rows.append((
                ticker, tw["created_at"], tw.get("id"), text, u.get("id"),
                u.get("followers_count", 0), u.get("friends_count", 0), u.get("statuses_count", 0),
                u.get("created_at"), bool(u.get("verified")), src.group(1) if src else "",
                text.startswith("RT @"),
            ))
    tw = pd.DataFrame(rows, columns=[
        "ticker", "created_at", "tweet_id", "text", "user_id", "followers", "friends", "statuses",
        "user_created_at", "verified", "source", "is_rt",
    ])
    tw = tw.drop_duplicates(["ticker", "tweet_id"])
    tw["ts"] = pd.to_datetime(tw["created_at"], format=TWITTER_TIME, utc=True)
    user_ts = pd.to_datetime(tw["user_created_at"], format=TWITTER_TIME, utc=True, errors="coerce")
    tw["acct_age_days"] = (tw["ts"] - user_ts).dt.total_seconds() / 86400
    tw["post_rate"] = tw["statuses"] / tw["acct_age_days"].clip(lower=1)
    tw["automated_client"] = ~tw["source"].isin(HUMAN_CLIENTS)
    tw["n_cashtags"] = tw["text"].str.count(CASHTAG_RE.pattern)
    tw["has_url"] = tw["text"].str.contains("http")
    tw["norm"] = tw["text"].map(norm_text)
    return tw.drop(columns=["created_at", "user_created_at"]).reset_index(drop=True)


def align_to_trading_day(tw: pd.DataFrame, trading_days) -> pd.DataFrame:
    """Assign each tweet to the first trading day whose 16:00 New York close is >= tweet time.

    Features for day t therefore only use tweets visible before close(t), and the target is the
    close(t) -> close(t+1) return, so there is no look-ahead. Tweets after the last close are dropped.
    """
    days = pd.DatetimeIndex(np.sort(pd.unique(pd.DatetimeIndex(trading_days))))
    closes = (days + pd.Timedelta(hours=16)).tz_localize("America/New_York").tz_convert(None).to_numpy()
    idx = np.searchsorted(closes, tw["ts"].dt.tz_convert(None).to_numpy(), side="left")
    ok = idx < len(days)
    out = tw.loc[ok].copy()
    out["date"] = days[idx[ok]]
    return out


def build(zip_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(zip_path) as z:
        px = load_prices(z)
        tw = load_tweets(z)
    return px, align_to_trading_day(tw, px["date"])

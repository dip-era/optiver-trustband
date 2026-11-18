"""A tweet may only feed the prediction made at the first 16:00 New York close at or after it."""
import pandas as pd

from trustband.data import align_to_trading_day

# Fri 2015-03-06, Mon 03-09 (US daylight saving starts Sun 03-08), Tue 03-10
TRADING_DAYS = pd.to_datetime(["2015-03-06", "2015-03-09", "2015-03-10"])


def _aligned(utc_times):
    tw = pd.DataFrame({"ts": pd.to_datetime(utc_times, utc=True)})
    return align_to_trading_day(tw, TRADING_DAYS)["date"].dt.strftime("%Y-%m-%d").tolist()


def test_before_and_after_the_close_winter_time():
    # Fri close = 16:00 EST = 21:00 UTC
    assert _aligned(["2015-03-06 20:59", "2015-03-06 21:01"]) == ["2015-03-06", "2015-03-09"]


def test_weekend_tweets_roll_to_monday():
    assert _aligned(["2015-03-07 12:00", "2015-03-08 23:00"]) == ["2015-03-09", "2015-03-09"]


def test_close_moves_with_daylight_saving():
    # Mon close = 16:00 EDT = 20:00 UTC; a 20:30 UTC tweet is after the close
    assert _aligned(["2015-03-09 19:59", "2015-03-09 20:30"]) == ["2015-03-09", "2015-03-10"]


def test_tweets_after_the_last_close_are_dropped():
    assert _aligned(["2015-03-10 23:00"]) == []

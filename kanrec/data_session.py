from pathlib import Path

import numpy as np
import pandas as pd

DAY = 86400


def load_yoochoose_clicks(raw_dir, cache_dir):
    cache = Path(cache_dir) / "yoochoose_clicks.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = pd.read_csv(Path(raw_dir) / "yoochoose-clicks.dat", header=None, usecols=[0, 1, 2],
                     names=["SessionId", "TimeStr", "ItemId"], dtype={"SessionId": np.int32, "ItemId": np.int64})
    ts = pd.to_datetime(df["TimeStr"], format="%Y-%m-%dT%H:%M:%S.%fZ")
    df["Time"] = ts.astype("datetime64[ms]").astype("int64") / 1000.0
    df = df.drop(columns="TimeStr")
    assert 1.39e9 < df["Time"].min() < df["Time"].max() < 1.42e9, "timestamps not in seconds"
    df.to_parquet(cache)
    return df


def _filter(df, min_item_support=5, min_session_len=2):
    df = df[df.groupby("SessionId")["ItemId"].transform("size") >= min_session_len]
    df = df[df.groupby("ItemId")["SessionId"].transform("size") >= min_item_support]
    return df[df.groupby("SessionId")["ItemId"].transform("size") >= min_session_len]


def _split_last_day(df):
    session_end = df.groupby("SessionId")["Time"].max()
    cut = session_end.max() - DAY
    later = session_end.index[session_end >= cut]
    return df[~df["SessionId"].isin(later)], df[df["SessionId"].isin(later)], cut


def _restrict_to_train_items(test, train, min_session_len=2):
    test = test[test["ItemId"].isin(train["ItemId"].unique())]
    return test[test.groupby("SessionId")["ItemId"].transform("size") >= min_session_len]


def _most_recent_fraction(df, fraction):
    session_start = df.groupby("SessionId")["Time"].min().sort_values()
    keep = session_start.index[-(len(session_start) // fraction):]
    return df[df["SessionId"].isin(keep)]


def prepare_yoochoose(clicks, fraction=64):
    """GRU4Rec protocol: drop length-1 sessions and items with support < 5; test = sessions ending
    in the last day; train = most recent 1/fraction of the remaining sessions; validation = the
    last day of that train set. Test and validation keep only items seen in the final train set."""
    df = _filter(clicks)
    train_full, test, test_cut = _split_last_day(df)
    train_frac = _most_recent_fraction(train_full, fraction)
    train, valid, valid_cut = _split_last_day(train_frac)
    valid = _restrict_to_train_items(valid, train)
    test = _restrict_to_train_items(test, train)
    splits = {"train": train, "valid": valid, "test": test}
    for name in splits:
        splits[name] = splits[name].sort_values(["SessionId", "Time"], kind="stable").reset_index(drop=True)
    sessions = {k: set(v["SessionId"].unique()) for k, v in splits.items()}
    assert not (sessions["train"] & sessions["valid"]), "train/valid session overlap"
    assert not (sessions["train"] & sessions["test"]), "train/test session overlap"
    assert not (sessions["valid"] & sessions["test"]), "valid/test session overlap"
    assert splits["train"].groupby("SessionId")["Time"].max().max() < valid_cut
    assert splits["valid"].groupby("SessionId")["Time"].max().max() < test_cut
    return splits


def session_stats(splits):
    rows = {}
    for name in ("train", "valid", "test"):
        df = splits[name]
        lens = df.groupby("SessionId").size()
        rows[name] = {
            "Sessions": len(lens),
            "Events": len(df),
            "Items": df["ItemId"].nunique(),
            "Prediction targets": int((lens - 1).sum()),
            "Mean session length": round(float(lens.mean()), 2),
            "Median session length": float(lens.median()),
        }
    return pd.DataFrame(rows)

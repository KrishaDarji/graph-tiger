"""Data loading and chronological user-item splitting."""
from pathlib import Path
import pandas as pd


def load_interactions(path="data/processed/interactions.parquet"):
    df = pd.read_parquet(path)
    return df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)


def encode_ids(df):
    result = df.copy()
    result["user_idx"], user_values = pd.factorize(result["user_id"])
    result["item_idx"], item_values = pd.factorize(result["item_id"])
    return result, list(user_values), list(item_values)


def chronological_split(df):
    groups = df.groupby("user_idx", sort=False)
    train = groups.head(-2).copy()
    validation = groups.tail(2).groupby("user_idx", sort=False).nth(0).reset_index()
    test = groups.tail(1).reset_index(drop=True)
    return train, validation, test


def save_split(df, output_dir="data/processed"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    encoded, users, items = encode_ids(df)
    train, validation, test = chronological_split(encoded)
    train.to_parquet(output / "train.parquet", index=False)
    validation.to_parquet(output / "validation.parquet", index=False)
    test.to_parquet(output / "test.parquet", index=False)
    print(f"Train: {len(train):,}; validation: {len(validation):,}; test: {len(test):,}")
    print(f"Users: {len(users):,}; items: {len(items):,}")
    return train, validation, test, len(users), len(items)


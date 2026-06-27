"""
Download and prepare MovieLens datasets for both RecBole and Elliot.

Creates:
  data/recbole/ml-100k/   — RecBole atomic files (.inter)
  data/recbole/ml-1m/
  data/elliot/ml-100k/    — TSV splits (train / val / test)
  data/elliot/ml-1m/
"""

import os
import io
import zipfile
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

DATASETS = {
    "ml-100k": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "ratings_path": "ml-100k/u.data",
        "sep": "\t",
        "header": None,
        "names": ["user_id", "item_id", "rating", "timestamp"],
    },
    "ml-1m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "ratings_path": "ml-1m/ratings.dat",
        "sep": "::",
        "header": None,
        "names": ["user_id", "item_id", "rating", "timestamp"],
    },
}


def download_zip(url: str) -> bytes:
    print(f"Downloading {url} ...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    buf = io.BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            buf.write(chunk)
            bar.update(len(chunk))
    buf.seek(0)
    return buf.read()


def load_ratings(name: str, raw: bytes) -> pd.DataFrame:
    cfg = DATASETS[name]
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        with zf.open(cfg["ratings_path"]) as f:
            df = pd.read_csv(
                f,
                sep=cfg["sep"],
                header=cfg["header"],
                names=cfg["names"],
                engine="python",
            )
    return df


def split_by_user(df: pd.DataFrame, val_ratio=0.1, test_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    trains, vals, tests = [], [], []
    for _, group in df.groupby("user_id"):
        idx = rng.permutation(len(group))
        n_test = max(1, int(len(group) * test_ratio))
        n_val = max(1, int(len(group) * val_ratio))
        test_idx = idx[:n_test]
        val_idx = idx[n_test : n_test + n_val]
        train_idx = idx[n_test + n_val :]
        trains.append(group.iloc[train_idx])
        vals.append(group.iloc[val_idx])
        tests.append(group.iloc[test_idx])
    return (
        pd.concat(trains).reset_index(drop=True),
        pd.concat(vals).reset_index(drop=True),
        pd.concat(tests).reset_index(drop=True),
    )


def write_recbole(df_full: pd.DataFrame, out_dir: Path, name: str):
    """Write a single .inter file; RecBole does its own splitting."""
    out_dir.mkdir(parents=True, exist_ok=True)
    inter_path = out_dir / f"{name}.inter"
    with open(inter_path, "w") as f:
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for _, row in df_full.iterrows():
            f.write(f"{row.user_id}\t{row.item_id}\t{row.rating}\t{row.timestamp}\n")
    print(f"  RecBole → {inter_path}  ({len(df_full)} interactions)")


def write_elliot(train, val, test, out_dir: Path):
    """Write tab-separated split files for Elliot."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["user_id", "item_id", "rating", "timestamp"]
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{split_name}.tsv"
        split_df[cols].to_csv(path, sep="\t", index=False, header=False)
        print(f"  Elliot  → {path}  ({len(split_df)} interactions)")


def prepare(name: str, base: Path):
    print(f"\n=== {name} ===")
    raw = download_zip(DATASETS[name]["url"])
    df = load_ratings(name, raw)

    # binarise: treat all ratings as implicit
    df = df[df["rating"] >= 1].copy()

    train, val, test = split_by_user(df)

    write_recbole(df, base / "recbole" / name, name)
    write_elliot(train, val, test, base / "elliot" / name)


if __name__ == "__main__":
    base = Path(__file__).parent / "data"
    for name in DATASETS:
        prepare(name, base)
    print("\nAll datasets ready.")

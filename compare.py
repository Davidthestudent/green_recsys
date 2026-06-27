"""
Compare RecBole vs Elliot results and produce:
  - Console table (tabulate)
  - results/comparison_<dataset>.csv
  - results/plots/<dataset>_<metric>.png

Usage:
    python compare.py
    python compare.py --datasets ml-100k --metrics ndcg@10 recall@10
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

BASE = Path(__file__).parent

# Map Elliot model names → shared canonical names
ELLIOT_NAME_MAP = {
    "ItemKNN":  "ItemKNN",
    "BPRMF":    "BPR",
    "EASER":    "EASE",
    "NeuMF":    "NeuMF",
    "LightGCN": "LightGCN",
    "NGCF":     "NGCF",
    "MultiVAE": "MultiVAE",
    # "DMF":      "DMF",
}

DEFAULT_METRICS = ["ndcg@10", "recall@10", "hit@10", "precision@10"]


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def normalise_keys(d: dict) -> dict:
    """Lower-case all metric keys for consistent comparison."""
    return {k.lower(): v for k, v in d.items()}


ENERGY_KEYS = ["_energy_kwh", "_co2_kg", "_duration_s"]
ENERGY_LABELS = {
    "_energy_kwh": "Energy (kWh)",
    "_co2_kg":     "CO₂ (kg)",
    "_duration_s": "Time (s)",
}


def build_comparison(dataset: str, metrics: list[str]) -> pd.DataFrame:
    recbole_raw = load_json(BASE / "results/recbole" / f"{dataset}.json")
    elliot_raw  = load_json(BASE / "results/elliot"  / f"{dataset}.json")

    # Normalise Elliot model names; strip internal _ keys from performance metrics
    elliot  = {ELLIOT_NAME_MAP.get(k, k): normalise_keys(v) for k, v in elliot_raw.items()}
    recbole = {k: normalise_keys(v) for k, v in recbole_raw.items()}

    models = sorted(set(list(recbole.keys()) + list(elliot.keys())))
    rows = []
    for model in models:
        for metric in metrics:
            if metric.startswith("_"):
                continue
            rb_val = recbole.get(model, {}).get(metric)
            el_val = elliot.get(model, {}).get(metric)
            rows.append({
                "model":   model,
                "metric":  metric,
                "RecBole": rb_val,
                "Elliot":  el_val,
                "diff":    (rb_val - el_val) if (rb_val and el_val) else None,
            })
    return pd.DataFrame(rows)


def build_energy_comparison(dataset: str) -> pd.DataFrame:
    recbole_raw = load_json(BASE / "results/recbole" / f"{dataset}.json")
    elliot_raw  = load_json(BASE / "results/elliot"  / f"{dataset}.json")

    elliot  = {ELLIOT_NAME_MAP.get(k, k): normalise_keys(v) for k, v in elliot_raw.items()}
    recbole = {k: normalise_keys(v) for k, v in recbole_raw.items()}

    models = sorted(set(list(recbole.keys()) + list(elliot.keys())))
    rows = []
    for model in models:
        for key in ENERGY_KEYS:
            rb_val = recbole.get(model, {}).get(key)
            el_val = elliot.get(model, {}).get(key)
            if rb_val is None and el_val is None:
                continue
            rows.append({
                "model":   model,
                "metric":  ENERGY_LABELS[key],
                "RecBole": rb_val,
                "Elliot":  el_val,
            })
    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame, dataset: str, metric: str):
    sub = df[df["metric"] == metric][["model", "RecBole", "Elliot", "diff"]].copy()
    sub = sub.sort_values("model")
    for col in ["RecBole", "Elliot", "diff"]:
        sub[col] = sub[col].apply(lambda x: f"{x:.4f}" if x is not None else "—")
    print(f"\n{dataset}  |  {metric.upper()}")
    print(tabulate(sub, headers="keys", showindex=False, tablefmt="github"))


def _bar_chart(ax, models, rb_vals, el_vals, ylabel: str, title: str):
    x = range(len(models))
    width = 0.35
    ax.bar([i - width / 2 for i in x], rb_vals, width, label="RecBole", color="#4c72b0")
    ax.bar([i + width / 2 for i in x], el_vals, width, label="Elliot",  color="#dd8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(bottom=0)


def plot(df: pd.DataFrame, dataset: str, metric: str, out_dir: Path):
    sub = df[(df["metric"] == metric) & df["RecBole"].notna() & df["Elliot"].notna()]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(sub) * 1.4), 5))
    _bar_chart(ax, sub["model"].tolist(), sub["RecBole"], sub["Elliot"],
               metric.upper(), f"{dataset} — {metric.upper()}")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dataset}_{metric.replace('@', '')}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {path}")


def plot_energy(energy_df: pd.DataFrame, dataset: str, out_dir: Path):
    """One subplot per energy metric, all models side by side."""
    labels = energy_df["metric"].unique().tolist()
    if not labels:
        return

    fig, axes = plt.subplots(1, len(labels),
                             figsize=(5 * len(labels), 5),
                             squeeze=False)
    for ax, label in zip(axes[0], labels):
        sub = energy_df[energy_df["metric"] == label].dropna(subset=["RecBole", "Elliot"])
        if sub.empty:
            ax.set_visible(False)
            continue
        _bar_chart(ax, sub["model"].tolist(), sub["RecBole"], sub["Elliot"],
                   label, f"{dataset} — {label}")
        # Scientific notation for very small numbers
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.2e}" if v < 0.001 else f"{v:.3g}")
        )

    fig.suptitle(f"{dataset} — Energy & Carbon Footprint", fontsize=13, y=1.02)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dataset}_energy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ml-100k", "ml-1m"])
    parser.add_argument("--metrics",  nargs="+", default=DEFAULT_METRICS)
    args = parser.parse_args()

    plots_dir = BASE / "results/plots"

    for dataset in args.datasets:
        df = build_comparison(dataset, args.metrics)
        if df.empty:
            print(f"No results found for {dataset}. Run run_recbole.py and run_elliot.py first.")
            continue

        # Console tables — performance
        for metric in args.metrics:
            print_table(df, dataset, metric)

        # Console table — energy
        energy_df = build_energy_comparison(dataset)
        if not energy_df.empty:
            print(f"\n{dataset}  |  ENERGY & CARBON")
            sub = energy_df.copy()
            for col in ["RecBole", "Elliot"]:
                sub[col] = sub[col].apply(
                    lambda x: f"{x:.4g}" if x is not None else "—"
                )
            print(tabulate(sub, headers="keys", showindex=False, tablefmt="github"))

        # CSV
        csv_path = BASE / "results" / f"comparison_{dataset}.csv"
        df.to_csv(csv_path, index=False)
        energy_csv = BASE / "results" / f"energy_{dataset}.csv"
        if not energy_df.empty:
            energy_df.to_csv(energy_csv, index=False)
            print(f"\n  Energy CSV saved → {energy_csv}")
        print(f"\n  CSV saved → {csv_path}")

        # Plots — performance metrics
        for metric in args.metrics:
            plot(df, dataset, metric, plots_dir)

        # Plots — energy metrics
        if not energy_df.empty:
            plot_energy(energy_df, dataset, plots_dir)


if __name__ == "__main__":
    main()

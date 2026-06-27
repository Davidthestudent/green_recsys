"""
Run all models with Elliot and save results.

Usage:
    python run_elliot.py                    # all datasets, fixed params
    python run_elliot.py --tune             # grid search first, then final eval
    python run_elliot.py --datasets ml-100k
"""

import argparse
import copy
import itertools
import json
import re
import tempfile
from pathlib import Path

import yaml

BASE = Path(__file__).parent

DATASET_CONFIGS = {
    "ml-100k": BASE / "config/elliot/ml-100k.yml",
    "ml-1m":   BASE / "config/elliot/ml-1m.yml",
}

ELLIOT_TO_RECBOLE = {
    "ItemKNN":  "ItemKNN",
    "BPRMF":    "BPR",
    "EASER":    "EASE",
    "NeuMF":    "NeuMF",
    "LightGCN": "LightGCN",
    "NGCF":     "NGCF",
    "MultiVAE": "MultiVAE",
}

# Same search space as RecBole hyper files
PARAM_GRIDS = {
    "ItemKNN": {
        "neighbors": [50, 100, 200],
        "shrink":    [0, 10, 50],
    },
    "BPRMF": {
        "factors": [32, 64, 128],
    },
    "EASER": {
        "l2_norm": [100, 500, 1000, 5000],
    },
}

_METRIC_MAP = {
    "ndcg":      "ndcg@10",
    "recall":    "recall@10",
    "precision": "precision@10",
    "hr":        "hit@10",
    "mrr":       "mrr@10",
    "map":       "map@10",
}


def _perf_dir(dataset: str) -> Path:
    return BASE / "config" / "results" / dataset / "performance"


def _latest_tsv(dataset: str, cutoff: int = 10) -> Path | None:
    candidates = sorted(_perf_dir(dataset).glob(f"rec_cutoff_{cutoff}_*.tsv"),
                        key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _parse_tsv(tsv: Path) -> dict:
    """Return {model_name: {metric: value}} from a single TSV."""
    with open(tsv) as f:
        lines = f.readlines()
    if len(lines) < 2:
        return {}
    headers = lines[0].strip().split("\t")
    result = {}
    for line in lines[1:]:
        values = line.strip().split("\t")
        if not values[0]:
            continue
        m = re.match(r"([A-Za-z]+)", values[0])
        name = m.group(1) if m else values[0]
        result[name] = {h: float(v) for h, v in zip(headers, values) if h != "model"}
    return result


def parse_elliot_results(dataset: str, cutoff: int = 10) -> dict:
    tsv = _latest_tsv(dataset, cutoff)
    return _parse_tsv(tsv) if tsv else {}


def _run_with_fixed_params(run_experiment, base_cfg: dict,
                            dataset: str, model: str, params: dict) -> dict:
    """Write a temp YAML with fixed params, run Elliot, return parsed metrics."""
    cfg = copy.deepcopy(base_cfg)
    model_cfg = cfg["experiment"]["models"][model]
    for k, v in params.items():
        model_cfg[k] = v
    cfg["experiment"]["models"][model]["meta"]["hyper_max_evals"] = 1
    # Remove list-valued params that would trigger Elliot's built-in tuning
    for k in list(model_cfg.keys()):
        if isinstance(model_cfg[k], list) and k not in params:
            del model_cfg[k]
    # Run only the specified model (not all models in the config)
    cfg["experiment"]["models"] = {model: cfg["experiment"]["models"][model]}

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False,
        dir=BASE / "config/elliot"
    ) as f:
        yaml.dump(cfg, f)
        temp_path = Path(f.name)

    before = _latest_tsv(dataset)
    try:
        run_experiment(str(temp_path))
    finally:
        temp_path.unlink()

    after = _latest_tsv(dataset)
    if after and (before is None or after.stat().st_mtime > before.stat().st_mtime):
        return _parse_tsv(after).get(model, {})
    return {}


def tune_dataset(run_experiment, dataset: str, config_path: Path) -> dict:
    """Grid search over PARAM_GRIDS; return {model: best_params}."""
    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)

    best_params_per_model = {}

    for model, grid in PARAM_GRIDS.items():
        if model not in base_cfg.get("experiment", {}).get("models", {}):
            continue

        combos = list(itertools.product(*grid.values()))
        print(f"\n  Tuning {model}: {len(combos)} combinations "
              f"({', '.join(grid.keys())})")

        best_score = -float("inf")
        best_params = {}

        for combo in combos:
            params = dict(zip(grid.keys(), combo))
            metrics = _run_with_fixed_params(
                run_experiment, base_cfg, dataset, model, params
            )
            score = metrics.get("nDCG", -float("inf"))
            print(f"    {params} → nDCG@10={score:.4f}")
            if score > best_score:
                best_score = score
                best_params = params

        print(f"  Best: {best_params} (nDCG@10={best_score:.4f})")
        best_params_per_model[model] = best_params

    return best_params_per_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_CONFIGS),
                        choices=list(DATASET_CONFIGS))
    parser.add_argument("--models", nargs="+", default=None,
                        help="Run only these models (e.g. --models ItemKNN BPRMF)")
    parser.add_argument("--tune", action="store_true",
                        help="Grid search over PARAM_GRIDS before final eval")
    args = parser.parse_args()

    try:
        from elliot.run import run_experiment
    except ImportError:
        raise SystemExit(
            "Elliot not installed. Run:\n"
            "  pip install git+https://github.com/sisinflab/elliot.git\n"
            "or create a separate conda env — see requirements-elliot.txt"
        )

    out_dir = BASE / "results/elliot"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_best_params = {}

    for dataset in args.datasets:
        config_path = DATASET_CONFIGS[dataset]
        print(f"\n{'='*60}")
        print(f"  Elliot | {dataset}")
        print(f"{'='*60}")

        best_params = {}
        if args.tune:
            best_params = tune_dataset(run_experiment, dataset, config_path)
            all_best_params[dataset] = best_params
            print(f"\n  Running final eval with best params...")

        # Final eval: one model at a time so we can measure energy per model
        with open(config_path) as f:
            base_cfg = yaml.safe_load(f)
        models_in_cfg = [m for m in base_cfg["experiment"]["models"].keys()
                         if args.models is None or m in args.models]

        from codecarbon import EmissionsTracker
        normalised = {}
        for model in models_in_cfg:
            params = best_params.get(model, {})
            tracker = EmissionsTracker(save_to_file=False, log_level="error")
            tracker.start()
            metrics = _run_with_fixed_params(run_experiment, base_cfg, dataset, model, params)
            tracker.stop()
            if not metrics:
                continue
            ed = tracker.final_emissions_data
            row = {_METRIC_MAP.get(k.lower(), k.lower()): v for k, v in metrics.items()}
            row["_energy_kwh"] = ed.energy_consumed
            row["_co2_kg"]     = ed.emissions
            row["_duration_s"] = ed.duration
            normalised[ELLIOT_TO_RECBOLE.get(model, model)] = row

        out_path = out_dir / f"{dataset}.json"
        with open(out_path, "w") as f:
            json.dump(normalised, f, indent=2)
        print(f"\n  Saved → {out_path}")

    if args.tune and all_best_params:
        params_path = out_dir / "best_params.json"
        with open(params_path, "w") as f:
            json.dump(all_best_params, f, indent=2)
        print(f"\n  Best params saved → {params_path}")

    print("\nElliot runs complete.")


if __name__ == "__main__":
    main()

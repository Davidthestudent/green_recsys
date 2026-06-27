"""
Run all models with RecBole and save results.

Usage:
    python run_recbole.py                         # all datasets & models
    python run_recbole.py --datasets ml-100k      # single dataset
    python run_recbole.py --models ItemKNN        # subset of models
    python run_recbole.py --tune                  # hyperparameter search first
"""

import argparse
import itertools
import json
import re
from pathlib import Path

from recbole.quick_start import run_recbole

BASE = Path(__file__).parent

MODELS = {
    "ItemKNN":  BASE / "config/recbole/itemknn.yaml",
    "BPR":      BASE / "config/recbole/bpr.yaml",
    "EASE":     BASE / "config/recbole/ease.yaml",
    "NeuMF":    BASE / "config/recbole/neumf.yaml",
    "LightGCN": BASE / "config/recbole/lightgcn.yaml",
    "NGCF":     BASE / "config/recbole/ngcf.yaml",
    "MultiVAE": BASE / "config/recbole/multivae.yaml",
    # "DMF":      BASE / "config/recbole/dmf.yaml",
}

HYPER_FILES = {
    "ItemKNN": BASE / "config/recbole/itemknn_hyper.test",
    "BPR":     BASE / "config/recbole/bpr_hyper.test",
    "EASE":    BASE / "config/recbole/ease_hyper.test",
}

DATASETS = {
    "ml-100k": BASE / "config/datasets/ml-100k.yaml",
    "ml-1m":   BASE / "config/datasets/ml-1m.yaml",
}

BASE_CONFIG = {
    "checkpoint_dir": str(BASE / "results/recbole/checkpoints"),
}


def _recbole_config() -> dict:
    return {**BASE_CONFIG, "data_path": str(BASE / "data/recbole")}


def run(dataset: str, model: str, dataset_cfg: Path, model_cfg: Path,
        extra: dict | None = None) -> dict:
    from codecarbon import EmissionsTracker
    tracker = EmissionsTracker(save_to_file=False, log_level="error")
    tracker.start()
    result = run_recbole(
        model=model,
        dataset=dataset,
        config_file_list=[str(dataset_cfg), str(model_cfg)],
        config_dict={**_recbole_config(), **(extra or {})},
    )
    tracker.stop()
    ed = tracker.final_emissions_data
    metrics = {k: float(v) for k, v in result["test_result"].items()}
    metrics["_energy_kwh"]  = ed.energy_consumed
    metrics["_co2_kg"]      = ed.emissions
    metrics["_duration_s"]  = ed.duration
    return metrics


def _parse_hyper_file(path: Path) -> dict[str, list]:
    """Parse RecBole-style hyper file; only 'choice' lines are supported."""
    grid = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"(\w+)\s+choice\s+\[(.+)\]", line)
        if m:
            key = m.group(1)
            grid[key] = [_cast(v.strip()) for v in m.group(2).split(",")]
    return grid


def _cast(s: str):
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def tune(dataset: str, model: str, dataset_cfg: Path, model_cfg: Path,
         hyper_file: Path, n_trials: int = 25) -> dict:
    """External grid search — avoids hyperopt's RandomState.integers() bug."""
    grid = _parse_hyper_file(hyper_file)
    combos = list(itertools.product(*grid.values()))[:n_trials]
    print(f"  Grid: {grid}  →  {len(combos)} combinations")

    best_score = -float("inf")
    best_params: dict = {}

    for combo in combos:
        params = dict(zip(grid.keys(), combo))
        result = run_recbole(
            model=model,
            dataset=dataset,
            config_file_list=[str(dataset_cfg), str(model_cfg)],
            config_dict={**_recbole_config(), **params},
        )
        score = float(result.get("best_valid_score", -1))
        print(f"    {params}  →  valid NDCG={score:.4f}")
        if score > best_score:
            best_score = score
            best_params = params

    print(f"  Best: {best_params}  (valid NDCG={best_score:.4f})")
    return best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--models",   nargs="+", default=list(MODELS),   choices=list(MODELS))
    parser.add_argument("--tune", action="store_true",
                        help="Run hyperparameter search before final eval")
    parser.add_argument("--n-trials", type=int, default=25,
                        help="Number of hyperopt trials per dataset/model (default: 25)")
    args = parser.parse_args()

    out_dir = BASE / "results/recbole"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    best_params_all = {}

    for dataset in args.datasets:
        all_results[dataset] = {}
        best_params_all[dataset] = {}

        for model in args.models:
            print(f"\n{'='*60}")
            print(f"  RecBole | {dataset} | {model}")
            print(f"{'='*60}")
            try:
                extra = {}
                if args.tune and model in HYPER_FILES:
                    print(f"  Running hyperparameter search ({args.n_trials} trials)...")
                    best = tune(
                        dataset, model,
                        DATASETS[dataset], MODELS[model],
                        HYPER_FILES[model], args.n_trials,
                    )
                    best_params_all[dataset][model] = best
                    extra = best

                metrics = run(dataset, model, DATASETS[dataset], MODELS[model], extra)
                all_results[dataset][model] = metrics
                print(f"  NDCG@10: {metrics.get('ndcg@10', 'N/A'):.4f}")

            except Exception as exc:
                import traceback; traceback.print_exc()
                all_results[dataset][model] = {"error": str(exc)}

        out_path = out_dir / f"{dataset}.json"
        with open(out_path, "w") as f:
            json.dump(all_results[dataset], f, indent=2)
        print(f"\n  Saved → {out_path}")

    if args.tune:
        params_path = out_dir / "best_params.json"
        with open(params_path, "w") as f:
            json.dump(best_params_all, f, indent=2)
        print(f"\n  Best params saved → {params_path}")

    print("\nRecBole runs complete.")


if __name__ == "__main__":
    main()

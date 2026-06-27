# RecBole × Elliot Benchmark

Compare recommendation models that exist in **both** [RecBole](https://github.com/RUCAIBox/RecBole) and [Elliot](https://github.com/sisinflab/elliot) under identical hyperparameter defaults.

Inspired by [recsysdefaults](https://github.com/alansaid/recsysdefaults).

---

## Models

| Model | RecBole name | Elliot name |
|---|---|---|
| Item K-NN | `ItemKNN` | `ItemKNN` |
| BPR-MF | `BPR` | `BPRMF` |
| Neural MF | `NeuMF` | `NeuMF` |
| LightGCN | `LightGCN` | `LightGCN` |
| NGCF | `NGCF` | `NGCF` |
| EASE | `EASE` | `EASE_R` |
| DMF | `DMF` | `DMF` |

## Datasets

| Dataset | Users | Items | Interactions |
|---|---|---|---|
| MovieLens 100K | 943 | 1 682 | 100 000 |
| MovieLens 1M | 6 040 | 3 952 | 1 000 209 |

## Metrics

Evaluated at cutoff 10 and 20: **NDCG, Recall, Precision, HR, MRR, MAP**.  
Primary metric for ranking: **NDCG@10**.

---

## Setup

### 1 — Clone and enter

```bash
cd ~/Desktop/recsys_bench
```

### 2 — Install RecBole environment

```bash
conda create -n recbole python=3.10 -y
conda activate recbole
pip install -r requirements.txt
```

### 3 — Install Elliot environment

Elliot and RecBole can conflict on PyTorch/TF versions, so use a separate env:

```bash
conda create -n elliot python=3.10 -y
conda activate elliot
pip install git+https://github.com/sisinflab/elliot.git
pip install -r requirements-elliot.txt
```

---

## Running

### Step 1 — Prepare data (run once, any env)

```bash
python prepare_data.py
```

Creates `data/recbole/` and `data/elliot/` with train / val / test splits.

### Step 2 — Run RecBole

```bash
conda activate recbole
python run_recbole.py
# or a subset:
python run_recbole.py --datasets ml-100k --models BPR LightGCN
```

Results → `results/recbole/<dataset>.json`

### Step 3 — Run Elliot

```bash
conda activate elliot
python run_elliot.py
# or a subset:
python run_elliot.py --datasets ml-100k
```

Results → `results/elliot/<dataset>.json`

### Step 4 — Compare

```bash
python compare.py
```

Prints tables, writes `results/comparison_<dataset>.csv` and `results/plots/*.png`.

---

## Project layout

```
recsys_bench/
├── prepare_data.py        # download & format datasets
├── run_recbole.py         # RecBole experiments
├── run_elliot.py          # Elliot experiments
├── compare.py             # side-by-side comparison + plots
├── requirements.txt       # RecBole env deps
├── requirements-elliot.txt
├── config/
│   ├── datasets/          # RecBole dataset YAML configs
│   ├── recbole/           # per-model RecBole YAML configs
│   └── elliot/            # per-dataset Elliot YAML configs (all models inside)
├── data/                  # created by prepare_data.py
│   ├── recbole/
│   └── elliot/
└── results/
    ├── recbole/
    ├── elliot/
    ├── plots/
    └── comparison_*.csv
```

---

## Notes

- All models run with **default hyperparameters** as configured in `config/`.  
  Edit the YAML files to tune them.
- RecBole splits data internally (80/10/10 random per user).  
  Elliot uses the pre-split TSVs created by `prepare_data.py` (same ratios).
- Implicit feedback: all ratings ≥ 1 are treated as positive interactions.

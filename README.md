# aba-mining-tasks-graph-embedding-kmitl-internship-2026

Research project exploring Knowledge Graph Embedding (KGE) models for Assumption-Based Argumentation (ABA) relation prediction. Conducted jointly with an internship student from KMITL (April – June 2026).

## Task

Given a triple `(head, ?, tail)`, predict the ABA relation — **CONTRARY_TO**, **NOT_CONTRARY**, or **SUPPORT** — between nodes derived from hotel review arguments.

## Dataset

**`data/input_data/hotel_contrary_dataset_support.csv`**

| Relation | Count | % | Meaning |
|---|---|---|---|
| NOT_CONTRARY | ~86,873 | ~93.9% | Body pairs that are not contrary to each other |
| CONTRARY_TO | ~4,776 | ~5.2% | Body pairs where one is contrary to the other |
| SUPPORT | ~804 | ~0.9% | Body → claim inference edges |

Domains: **staff**, **price**, **check-in**, **check-out**

Split: stratified by `domain × relation` (80 / 10 / 10 train / val / test).  
Groups with fewer than 10 rows go entirely into train to avoid empty val/test splits.

## Models

### KGE Models (PyKEEN)

All five models share the same training configuration:
- Optimizer: Adam (lr = 0.0001)
- Loss: NSSALoss (adversarial temp = 0.5)
- Negative sampling: relation corruption only (3 negatives per positive)
- Training loop: sLCWA
- Early stopping: patience 10, frequency 10 epochs

| File | Model | Scoring function | Symmetry |
|---|---|---|---|
| `models/sup_rotate.py` | RotatE | `\|h ∘ r - t\|` (complex rotation) | symmetric + antisymmetric |
| `models/sup_complex.py` | ComplEx | `Re(hᵀ · diag(r) · conj(t))` | symmetric + antisymmetric |
| `models/sup_distmult.py` | DistMult | `sum(h * r * t)` | symmetric only |
| `models/sup_convkb.py` | ConvKB | convolutional over `[h, r, t]` | asymmetric |
| `models/sup_transe.py` | TransE | `\|h + r - t\|` (translation) | neither |

Each training script produces:
- `visualization_data.csv` — scored triples with 2D PCA entity coordinates
- `aba_tree_structure.csv` — ABA tree (claim → body → attacker with scores)
- `entity_embeddings.csv` — full embedding matrix (one row per entity)
- `trained_model_<Model>.pkl` — saved model weights (for ensemble)
- `triples_factory_<Model>.pkl` — entity/relation ID mapping
- `relation_eval_ranked.csv` — per-triple relation ranking results
- `negative_samples.csv` — all negative triples generated during training

### LR Baselines

| File | Description |
|---|---|
| `models/lr_baseline.py` | Logistic Regression on KGE entity embeddings; tries 4 feature strategies: concat, diff, hadamard, all |
| `models/lr_onehot.py` | Logistic Regression on one-hot entity encoding (no KGE embeddings needed) |

`lr_baseline.py`: set `SOURCE_MODEL` at the top to choose which KGE model's embeddings to use (`"rotate"`, `"complex"`, `"distmult"`, `"convkb"`, `"transe"`).

### Calibrated Ensemble

**`models/caliensemble.py`** combines all 5 KGE models via Platt scaling:

1. Score all `(h, t)` pairs with each model across all 3 relations
2. Fit one binary logistic calibrator per relation per model (5-fold CV on train set)
3. Average calibrated probabilities across models in the ensemble
4. Report metrics for every model subset (1-model through 5-model combinations)

### Comparison and Visualization

| File | Description |
|---|---|
| `models/compare_all_models.py` | Loads KGE CSVs + re-runs LR + loads Takashima JSON; prints Tables 1 & 2; saves PNG + JSON |
| `models/visualize_results.py` | Reads `sup_experiment_results.xlsx`; produces per-model bar charts for KGE + LR |

`compare_all_models.py` requires the Takashima thesis experiment JSON at:  
`../takashima-master-thesis-march-2026/data/training_results/exp_all_models_3class_fixed/experiment_results.json`

## Project Structure

```
aba-mining-tasks-graph-embedding-kmitl-internship-2026/
├── data/
│   ├── input_data/
│   │   └── hotel_contrary_dataset_support.csv   ← main dataset (3 relations, 4 domains)
│   └── experiment_results/
│       └── sup_experiment_results.xlsx          ← auto-recorded experiment log
├── models/
│   ├── sup_rotate.py          ← RotatE training + evaluation
│   ├── sup_complex.py         ← ComplEx training + evaluation
│   ├── sup_distmult.py        ← DistMult training + evaluation
│   ├── sup_convkb.py          ← ConvKB training + evaluation
│   ├── sup_transe.py          ← TransE training + evaluation
│   ├── lr_baseline.py         ← LR on KGE embeddings (4 strategies)
│   ├── lr_onehot.py           ← LR on one-hot entity encoding
│   ├── caliensemble.py        ← Calibrated ensemble (all 5 KGE models)
│   ├── compare_all_models.py  ← Full comparison (KGE + LR + Takashima)
│   ├── visualize_results.py   ← Visualization from experiment Excel
│   └── testcases_sup.py       ← Unit test suite (458 tests)
└── outputs/
    ├── rotate_sup_output/        ← RotatE outputs
    ├── complex_sup_output/       ← ComplEx outputs
    ├── distmult_sup_output/      ← DistMult outputs
    ├── convkb_sup_output/        ← ConvKB outputs
    ├── transe_sup_output/        ← TransE outputs
    ├── onehot_sup_output/        ← OneHot-LR outputs
    ├── ensemble_output/          ← Ensemble results
    └── compare_all_models.png    ← 3×3 comparison figure (15 models)
```

## How to Run

### 1. Train a KGE model

```bash
cd models
python sup_rotate.py   # or sup_complex.py, sup_distmult.py, etc.
```

Requires a GPU for practical training times. Outputs are saved to `outputs/<model>_sup_output/`.

### 2. LR baseline

```bash
# Edit SOURCE_MODEL in lr_baseline.py first (e.g. "rotate")
python models/lr_baseline.py

# One-hot version (no GPU, no prior KGE run needed)
python models/lr_onehot.py
```

### 3. Calibrated ensemble

All 5 KGE training scripts must have completed first (so all `.pkl` files exist):

```bash
python models/caliensemble.py
```

### 4. Full comparison plot

```bash
python models/compare_all_models.py
```

### 5. Run tests

```bash
cd models
python -m pytest testcases_sup.py -v
```

## Key Metrics

| Metric | Notes |
|---|---|
| **Macro-Hits@1** | Primary metric — per-relation accuracy averaged equally across 3 relations |
| **Micro-Hits@1** | Overall accuracy (dominated by NOT_CONTRARY at 93.9%) |
| **MRR** | Mean reciprocal rank across test triples |
| Hits@3, Hits@10 | Trivially 1.0 with only 3 relations — not informative |

The custom `RelationRankEvaluator` in each KGE script scores all 3 relations for every `(h, ?, t)` query and ranks the true relation — exactly mirroring the relation corruption training objective.

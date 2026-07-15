# aba-mining-tasks-graph-embedding-kmitl-internship-2026

This is a repository that contains the experiments for ABA mining tasks (using graph embedding), jointly conducted with an internship student from KMITL in April 2026.

## Project Structure

```
ABA-MINING-TASKS-GRAPH-EMBEDDING-KMITL-INTERNSHIP-2026/
├── rotate_output/                  # Output folder — RotatE full dataset
│   └── negative_samples.csv        # Negative triples captured during training
├── rotate_pnnp_output/             # Output folder — RotatE PNNP dataset
│   └── negative_samples.csv        # Negative triples captured during training
├── transe_output/                  # Output folder — TransE full dataset
│   └── negative_samples.csv        # Negative triples captured during training
├── transe_pnnp_output/             # Output folder — TransE PNNP dataset
│   └── negative_samples.csv        # Negative triples captured during training
├── experiment_results.xlsx         # Auto-recorded results for all runs
├── hotel_contrary_dataset_all.csv  # Full dataset (91,714 triples)
├── hotel_contrary_dataset_PNNP.csv # Filtered dataset (13,942 triples)
├── README.md                       # Project documentation
├── rotate_EXT.py                   # RotatE — full dataset (ExtendedBasicNegativeSampler)
├── rotate_pnnp_EXT.py              # RotatE — PNNP dataset (ExtendedBasicNegativeSampler)
├── rotate_pnnp.py                  # RotatE — PNNP dataset (standard basic sampler)
├── rotate.py                       # RotatE — full dataset (standard basic sampler)
├── transe_EXT.py                   # TransE — full dataset (ExtendedBasicNegativeSampler)
├── transe_pnnp_EXT.py              # TransE — PNNP dataset (ExtendedBasicNegativeSampler)
├── transe_pnnp.py                  # TransE — PNNP dataset (standard basic sampler)
└── transe.py                       # TransE — full dataset (standard basic sampler)
```

## Datasets

### Full Dataset (`hotel_contrary_dataset_all.csv`)

- **Total triples:** 91,714
- **Source:** All 4 sheet types from the original Excel files

**Relations:**

| Relation      | Count  | Percentage | Description                    |
|---------------|-------:|-----------:|---------------------------------|
| CONTRARY_TO   | 4,776  | 5.2%       | Human-verified contrary pairs   |
| NOT_CONTRARY  | 86,938 | 94.8%      | Verified non-contrary pairs     |

### Filtered Dataset — PNNP (`hotel_contrary_dataset_PNNP.csv`)

- **Total triples:** 13,942

## Models

The pipeline is built on PyKEEN with a custom negative sampler and evaluator. Five KGE models were trained and compared (RotatE and TransE are the models with scripts in this repo; ConvKB, ConvE, ComplEx, and DistMult were run under the same pipeline for the comparative study):

| Model    | Loss                | Loop  | Notes                                                              |
|----------|----------------------|-------|---------------------------------------------------------------------|
| RotatE   | NSSALoss             | sLCWA | Rotation in complex space, relation corruption (edge-only), baseline model |
| TransE   | NSSALoss             | sLCWA | Distance-based baseline, relation corruption                       |
| ConvKB   | NSSALoss             | sLCWA | CNN over triple, relation corruption, harder negatives              |
| ConvE    | BCEAfterSigmoidLoss  | LCWA  | 1-vs-All scoring, real-valued embeddings, compatible with PCA export |
| ComplEx  | NSSALoss             | sLCWA | Complex space, handles symmetric & asymmetric relations             |
| DistMult | NSSALoss             | sLCWA | Bilinear scoring, fast baseline, limited to symmetric relations     |

**Common hyperparameters:** Embedding Dim = 100, Epochs = 1000 (with early stopper), Batch Size = 256, LR = 0.0001, Random Seed = 42, Train/Val/Test = 80/10/10%.

**Custom negative sampling — relation (edge) corruption:** the `ExtendedBasicNegativeSampler` subclasses PyKEEN's `BasicNegativeSampler` with `corruption_scheme=("relation")`. For each positive triple `(h, r, t)`, it generates *k* negatives by sampling random relations while keeping head/tail entities fixed — semantically appropriate for contrary mining, since the goal is predicting which relation holds between a given `(head, tail)` pair rather than entity prediction.

**Evaluation:** a custom `RelationRankEvaluator` computes per-relation and per-domain ranking metrics (Mean Rank, MRR, Hits@1/3/10) using PyKEEN's `RankBasedEvaluator` internals. Macro-Hits@1 is treated as the primary metric, since standard KGE metrics (Hits@3, Hits@10, overall MRR) are not meaningful with only 3 relations.

## Requirements

- Python 3.11
- [PyKEEN](https://github.com/pykeen/pykeen) (1.11.1)
- pandas
- openpyxl (Excel result logging)
- scikit-learn (Logistic Regression baselines, calibration)

Install with:

```bash
pip install pykeen==1.11.1 pandas openpyxl scikit-learn
```

## How to Run

Each script trains and evaluates one model on one dataset variant, then logs results to `experiment_results.xlsx`.

```bash
# RotatE — full dataset, standard basic sampler
python rotate.py

# RotatE — full dataset, ExtendedBasicNegativeSampler (relation corruption)
python rotate_EXT.py

# RotatE — PNNP (filtered) dataset, standard basic sampler
python rotate_pnnp.py

# RotatE — PNNP (filtered) dataset, ExtendedBasicNegativeSampler
python rotate_pnnp_EXT.py
```

The `transe*.py` scripts follow the same four variants for TransE. Each run:

- Loads the corresponding dataset (`hotel_contrary_dataset_all.csv` or `hotel_contrary_dataset_PNNP.csv`)
- Trains with the hyperparameters above
- Writes captured negative triples to `<model>_output/negative_samples.csv` (or `<model>_pnnp_output/` for the PNNP variant)
- Appends a results row (MR, MRR, Hits@1/3/10, training time, timestamp) to `experiment_results.xlsx`

## Results

### Individual KGE Model Evaluation (5 models, 3-class, macro)

| Model    | MRR    | MR    | Macro H@1 | Accuracy | Precision | Recall | F1     | CT H@1 | NC H@1 | SUP H@1 |
|----------|--------|-------|-----------|----------|-----------|--------|--------|--------|--------|---------|
| TransE   | 0.9801 | 1.040 | 0.8066    | 0.9601   | 0.8378    | 0.8066 | 0.8079 | 0.431  | 0.989  | 1.000   |
| RotatE   | 0.9809 | 1.038 | 0.8138    | 0.9617   | 0.8472    | 0.8138 | 0.8169 | 0.452  | 0.989  | 1.000   |
| ComplEx  | 0.9882 | 1.024 | 0.8667    | 0.9769   | 0.9095    | 0.8667 | 0.8848 | 0.692  | 0.993  | 0.915   |
| ConvKB   | 0.9814 | 1.038 | 0.8134    | 0.9634   | 0.8481    | 0.8134 | 0.8213 | 0.500  | 0.989  | 0.951   |
| DistMult | 0.9777 | 1.045 | 0.7147    | 0.9555   | 0.9567    | 0.7147 | 0.7343 | 0.144  | 1.000  | 1.000   |

**Key finding:** ComplEx is the strongest individual model, particularly on CONTRARY_TO (0.692 Hits@1), because its Hermitian inner product handles both symmetric and antisymmetric relation patterns. DistMult collapses on CONTRARY_TO (0.144) because its symmetric scoring function is structurally mismatched to an antisymmetric relation. CONTRARY_TO is the hardest relation for every model, driven by data sparsity (~3,820 training triples vs. ~69,493 for NOT_CONTRARY).

### Baseline Comparison — KGE + Logistic Regression

| Model        | MRR    | MR    | Macro H@1 | Accuracy | Precision | Recall | F1     | AUC    |
|--------------|--------|-------|-----------|----------|-----------|--------|--------|--------|
| ComplEx+LR   | 0.9851 | 1.030 | 0.9704    | 0.9262   | 0.8893    | 0.9262 | 0.9057 | 0.9841 |
| DistMult+LR  | 0.9741 | 1.052 | 0.9482    | 0.9526   | 0.8318    | 0.9526 | 0.8712 | 0.9881 |
| OneHot+LR    | 0.9686 | 1.063 | 0.9372    | 0.9705   | 0.8163    | 0.9705 | 0.8607 | 0.9890 |

### Calibrated Ensemble

A calibrated ensemble pipeline averages one-vs-rest, softmax-normalized probabilities across model subsets (all 31 combinations of the 5 KGE models), with 5-fold cross-validation used to fit the calibrators without bias.

| Model / Ensemble    | Type        | Micro H@1 | Macro H@1 | CT H@1 | NC H@1 | SUP H@1 | MRR   | Mean Rank |
|----------------------|-------------|-----------|-----------|--------|--------|---------|-------|-----------|
| ComplEx               | Individual  | 0.977     | 0.869     | 0.699  | 0.993  | 0.915   | 0.988 | 1.024     |
| **ComplEx + ConvKB**  | Ensemble-2  | **0.978** | **0.887** | **0.690** | 0.994 | 0.976 | **0.989** | **1.022** |
| Full 5-model ensemble | Ensemble-5  | 0.968     | 0.835     | 0.510  | 0.993  | 1.000   | 0.984 | 1.032     |

**Key finding:** ComplEx + ConvKB is the best-performing combination overall (Macro H@1 = 0.887), marginally improving on standalone ComplEx while reducing variance. Larger ensembles (3–5 models) do not consistently outperform the best pair — the full 5-model ensemble actually scores worse than ComplEx alone, since including DistMult (CT H@1 = 0.144) dilutes ensemble quality. No combination exceeds ComplEx's individual CONTRARY_TO score of 0.699, confirming the bottleneck is data sparsity rather than model architecture.

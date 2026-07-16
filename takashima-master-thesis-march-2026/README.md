# ABA Link Prediction for Hotel Reviews

Implementation of attack link prediction in Argumentation-Based Analysis (ABA) using hotel review data, accompanying the master's thesis:

> **"Attack Link Prediction in Argumentation Frameworks Extracted from Hotel Reviews"**

The system predicts *attack* relationships between argumentative components (assumptions and propositions) extracted from hotel reviews, combining Relational Graph Convolutional Networks (R-GCN) with fine-tuned BERT (via LoRA).

---

## Pipeline Overview

```
Raw CSVs
  data/input/  ← original ABA dataset (node text)
  data/output/ ← silver-label contra-relation CSVs (attack edges)
        |
        |  python src/preprocess/save_graph.py
        v
  data/output/aba_graph_room_staff_combined.pkl
        |
        |  python src/experiments/run_named_models.py   (Experiments 1 & 2)
        |  python src/experiments/run_datasize_sweep.py (Experiment 3)
        v
  data/training_results/<experiment-id>/
    experiment_results.json
    box_plots.png
    bar_charts.png
    comprehensive_analysis.png
```

---

## Experiment Design

The thesis comprises three experiment sets:

| # | Experiment | Script | Purpose |
|---|---|---|---|
| 1 | All-model comparison | `run_named_models.py` | Compares 6 models under identical 5-fold CV conditions |
| 2 | Proposed model with optimal hyperparams | `run_named_models.py --only-model FinetunedBertRgcnMlp` | Main thesis result using LoRA-fine-tuned BERT + R-GCN |
| 3 | Data-size sensitivity | `run_datasize_sweep.py` | Learning curves at 20 / 40 / 60 / 80 / 100 % of training data |

### All-model comparison results (5-fold CV, seed=42)

Results from `data/training_results/exp_room_staff_bothcsv_baseline_named_models_named_models/`:

| Model | Accuracy | F1 | AUC | Notes |
|---|---|---|---|---|
| **FinetunedBertRgcnMlp** (proposed) | **0.897 ± 0.007** | **0.919 ± 0.005** | **0.946 ± 0.007** | R-GCN + LoRA fine-tuned BERT |
| FreezedBertRgcnMlp | 0.889 ± 0.018 | 0.908 ± 0.018 | 0.944 ± 0.007 | R-GCN + frozen BERT embeddings |
| FreezedBertMlp | 0.727 ± 0.074 | 0.806 ± 0.041 | 0.732 ± 0.086 | MLP on frozen BERT embeddings |
| FinetunedBertMlp | 0.597 ± 0.012 | 0.748 ± 0.009 | 0.500 ± 0.000 | MLP with BERT fine-tuning (no graph) |
| TfidfLr | 0.634 ± 0.012 | 0.760 ± 0.009 | 0.640 ± 0.007 | TF-IDF + Logistic Regression |
| Random | 0.499 ± 0.017 | 0.545 ± 0.015 | 0.495 ± 0.014 | Random baseline |

---

## Requirements

- Python 3.10+
- PyTorch 2.4.1 (CPU or CUDA 12.1)
- GPU with 12 GB+ VRAM recommended for `FinetunedBertRgcnMlp` (tested on RTX 4070 Ti)

---

## Installation

### Option A: pip (local)

```bash
# 1. Install PyTorch — choose one:
# CPU only
pip install --index-url https://download.pytorch.org/whl/cpu \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# CUDA 12.1
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# 2. Install PyTorch Geometric (replace +cpu with +cu121 for GPU)
pip install torch-scatter torch-sparse pyg-lib torch-geometric==2.5.3 \
  -f https://data.pyg.org/whl/torch-2.4.1+cpu.html

# 3. Install this package and remaining dependencies
pip install -e .
```

### Option B: Docker (recommended for reproducibility)

```bash
# CPU (default)
docker compose up -d app

# GPU (requires NVIDIA Container Toolkit)
docker compose --profile gpu up -d app-gpu

# Open a shell in the container
docker compose exec app bash
```

---

## Environment Setup (OpenAI API — data annotation only)

The scripts under `src/make_contrary_with_openai/` call the OpenAI API to generate contrary-relation annotations. This step was already performed to produce the silver-label CSV files used in all experiments. You only need this if you want to re-run the annotation pipeline from scratch.

```bash
cp .env.example .env
# Edit .env and replace the placeholder with your actual OpenAI API key
```

Example usage:

```bash
python src/make_contrary_with_openai/gpt_4omini_contMakeAnno.py
```

---

## Reproducing Thesis Results

All commands are run from the project root. Follow the steps in order.

### Step 0: Build the graph (required before any experiment)

Place the silver-label CSV files in `data/output/` and the original ABA dataset in `data/input/` (already included in the repository). Then run:

```bash
python src/preprocess/save_graph.py
```

Output: `data/output/aba_graph_room_staff_combined.pkl`
- Nodes: 4,186 (argumentative components)
- Edges: 7,298 (inference + attack relations)

---

### Step 1: All-model comparison (Experiment 1)

Runs all 6 models under 5-fold cross-validation and saves metrics, plots, and a JSON summary.

```bash
python src/experiments/run_named_models.py \
  --config config/robust_experiment.yaml \
  --experiment-id exp_all_models
```

Results are saved to `data/training_results/exp_all_models_named_models/`.

To run a single model for quick verification (e.g., TF-IDF baseline):

```bash
python src/experiments/run_named_models.py \
  --config config/robust_experiment.yaml \
  --only-model TfidfLr \
  --experiment-id exp_tfidf_verify
```

Available `--only-model` values: `FreezedBertRgcnMlp`, `FinetunedBertRgcnMlp`, `FreezedBertMlp`, `FinetunedBertMlp`, `TfidfLr`, `Random`

---

### Step 2: Proposed model with optimal hyperparameters (Experiment 2)

Uses the hyperparameters in `config/hypara_finetunedbert_rgcn.json`, which were selected by grid search (see *Hyperparameter Search* below).

```bash
python src/experiments/run_named_models.py \
  --config config/robust_experiment.yaml \
  --only-model FinetunedBertRgcnMlp \
  --sweep-config config/hypara_finetunedbert_rgcn.json \
  --experiment-id exp_proposed_model
```

Expected results (5-fold CV, seed=42):

| Metric   | Mean ± Std    |
|----------|---------------|
| Accuracy | 0.897 ± 0.007 |
| F1       | 0.919 ± 0.005 |
| AUC      | 0.946 ± 0.007 |

Results are saved to `data/training_results/exp_proposed_model_named_models/`.

> **Note on reproducibility:** Two sources of variation affect exact reproduction of these numbers:
> 1. **Library versions** — results were obtained with `transformers==4.40.0` and `peft==0.11.0` (pinned in `pyproject.toml`). Newer versions may produce ~1–2% different F1/AUC due to changes in LoRA internals.
> 2. **Partial seed coverage** — `config/robust_experiment.yaml` sets `seed: 42`, which fixes the 5-fold CV splits and negative sampling. However, model weight initialization and BERT fine-tuning are not explicitly seeded in `run_named_models.py`, so run-to-run variation of ~±1% is expected even with the same environment.

#### Hyperparameter Search

The optimal hyperparameters for `FinetunedBertRgcnMlp` were determined by grid search and stored in `config/hypara_finetunedbert_rgcn.json`:

| Hyperparameter | Value | Description |
|---|---|---|
| `hidden_dim` | 256 | R-GCN hidden dimension |
| `num_layers` | 4 | Number of R-GCN layers |
| `dropout_link` | 0.15 | Dropout rate in link prediction head |
| `learning_rate` | 0.001 | Adam learning rate |
| `num_epochs` | 200 | Training epochs |
| `max_length` | 256 | BERT input token length |
| `lora_r` | 32 | LoRA rank |
| `lora_alpha` | 64 | LoRA scaling factor |
| `lora_dropout` | 0.1 | LoRA dropout |

To re-run the grid search from scratch:

```bash
python src/experiments/run_named_models.py \
  --config config/robust_experiment.yaml \
  --only-model FinetunedBertRgcnMlp \
  --sweep-config config/gridsearch_datasize.yaml \
  --experiment-id exp_gridsearch
```

---

### Step 3: Data-size sensitivity analysis (Experiment 3)

Trains models at 20 / 40 / 60 / 80 / 100% of training data with 3 seeds (42, 43, 44).

```bash
python src/experiments/run_datasize_sweep.py \
  --config config/datasize_sweep.yaml
```

Results are saved to `data/training_results/<auto-generated-id>/`.

---

## Project Structure

```
.
├── config/
│   ├── robust_experiment.yaml           # Main experiment hyperparameters (5-fold CV settings)
│   ├── datasize_sweep.yaml              # Data-size sweep configuration
│   ├── hypara_finetunedbert_rgcn.json   # Optimal hyperparameters for proposed model (from grid search)
│   └── gridsearch_datasize.yaml         # Grid search parameter ranges
├── data/
│   ├── input/
│   │   └── Original ABA Dataset *.csv   # Node text (assumptions, propositions)
│   └── output/
│       ├── Silver_Room_ContP_BodyN_4omini.csv   # Attack edge labels (Room topic)
│       ├── Silver_Room_ContN_BodyP_4omini.csv
│       ├── Silver_Staff_ContP_BodyN_4omini.csv  # Attack edge labels (Staff topic)
│       ├── Silver_Staff_ContN_BodyP_4omini.csv
│       └── aba_graph_room_staff_combined.pkl    # Preprocessed graph (generated)
├── src/
│   ├── preprocess/
│   │   ├── save_graph.py                # Builds ABA graph from CSV files (entry point)
│   │   ├── extract_edge.py              # Edge extraction and categorization
│   │   ├── embed_node.py                # BERT node embedding generation (768-dim)
│   │   └── make_graph_dataset.py        # PyTorch Geometric Data construction
│   ├── augmentation/
│   │   └── generate_negative.py         # Hard / structural / random negative sampling
│   ├── model_defs/
│   │   └── models.py                    # All models: R-GCN, BERT+LoRA, baselines
│   ├── model_training/
│   │   ├── train.py                     # R-GCN training loop
│   │   ├── train_bert.py                # BERT fine-tuning training loop
│   │   └── evaluate.py                  # Evaluation metrics (Accuracy, F1, AUC)
│   ├── experiments/
│   │   ├── cross_validation.py          # K-fold CV splitting
│   │   ├── run_robust_experiment.py     # General experiment runner (shared utilities)
│   │   ├── run_named_models.py          # Multi-model comparison runner (main entry point)
│   │   ├── run_datasize_sweep.py        # Data-size sweep runner
│   │   └── run_gridsearch_datasize.py   # Grid search runner
│   ├── contrastive_learning/
│   │   └── pretrain_embeddings.py       # SimCLR-style contrastive pre-training (optional)
│   ├── make_contrary_with_openai/       # GPT-4o-mini annotation pipeline (OpenAI API)
│   ├── aba_link_prediction/             # Supporting package (data loaders, trainers, evaluators)
│   └── visualization/
│       └── plot_results.py              # Box plots, bar charts, statistical tests
├── tests/
│   ├── conftest.py
│   ├── unit/
│   └── integration/
├── docs/
│   ├── TF-IDF_LRについて.md
│   └── contrastive_learning_experiment_plan.md
├── .env.example                         # API key template (copy to .env)
├── Dockerfile
├── docker-compose.yaml
└── pyproject.toml
```

---

## Running Tests

Tests use mock data and do not require real data files.

```bash
# All tests
pytest tests/

# Unit tests only
pytest tests/unit/

# With coverage report
pytest --cov=src --cov-report=html tests/
```

---

## Citation

If you use this code, please cite:

```
[Citation will be added after thesis publication]
```

This work builds on Argumentation-Based Analysis (ABA) theory and uses the hotel review dataset from the ABA mining project. The contrastive learning component uses [PyGCL](https://github.com/PyGCL/PyGCL). LoRA fine-tuning uses [PEFT](https://github.com/huggingface/peft).

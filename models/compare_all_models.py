"""
Cross-model comparison for ABA relation prediction (3-class: CONTRARY_TO, NOT_CONTRARY, SUPPORT).

Loads results from three sources:
  • KGE models   — relation_eval_ranked.csv in each model's output folder
  • LR baselines — re-runs LR with predict_proba on the same train/test split
  • Takashima    — loads metrics from his experiment JSON (BERT-RGCN, BERT-MLP, TF-IDF+LR, etc.)

Outputs:
  • Tables 1 & 2 printed to stdout
  • outputs/compare_all_models.png  (3×3 figure with all metrics)
  • outputs/compare_all_models_results.json  (machine-readable)

Claude-Assisted
"""

import csv, json, warnings
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 10,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 8,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ── paths ─────────────────────────────────────────────────────────────────────
ABA_DIR       = Path(__file__).resolve().parent.parent
TAKASHIMA_EXP = (ABA_DIR.parent / "takashima-master-thesis-march-2026" /
                 "data/training_results/exp_all_models_3class_fixed/experiment_results.json")
CSV_PATH      = ABA_DIR / "data/input_data/hotel_contrary_dataset_support.csv"

RELS = ["CONTRARY_TO", "NOT_CONTRARY", "SUPPORT"]

# ── colour palette ────────────────────────────────────────────────────────────
C = {
    "TransE":              "#2196F3",
    "RotatE":              "#43A047",
    "ComplEx":             "#5E35B1",
    "DistMult":            "#FB8C00",
    "ConvKB":              "#E53935",
    "ComplEx+LR":          "#9575CD",
    "DistMult+LR":         "#FFA726",
    "OneHot+LR":           "#8D6E63",
    "BERT-RGCN\n(frozen)":    "#E91E63",
    "BERT-RGCN\n(finetuned)": "#F48FB1",
    "BERT-MLP\n(frozen)":     "#7B1FA2",
    "BERT-MLP\n(finetuned)":  "#CE93D8",
    "BERT-CosSim\n(ft)":      "#3949AB",
    "TF-IDF+LR":              "#00897B",
    "Random":                 "#9E9E9E",
    "CONTRARY_TO":  "#C62828",
    "NOT_CONTRARY": "#1565C0",
    "SUPPORT":      "#2E7D32",
}


# ══════════════════════════════════════════════════════════════════════════════
# Data loaders
# ══════════════════════════════════════════════════════════════════════════════
def load_kge(csv_path: Path):
    if not csv_path.exists():
        return None
    mrr_sum = rank_sum = h1 = n = 0
    per_rel: dict[str, list] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rk  = int(row["relation_rank"])
            rr  = float(row["reciprocal_rank"])
            rel = row["relation"]
            mrr_sum  += rr;  rank_sum += rk
            h1       += (rk == 1);  n += 1
            per_rel.setdefault(rel, []).append(1 if rk == 1 else 0)
    if n == 0:
        return None
    # ── derive class sizes from per_rel counts
    n_k = {k: len(v) for k, v in per_rel.items()}   # class sizes
    tp  = {k: sum(v) for k, v in per_rel.items()}   # TP per class
    fn  = {k: n_k[k] - tp[k] for k in n_k}          # FN per class
    # ── estimate FP proportionally to target class size among non-true classes
    #    FP_j ≈ Σ_{k≠j} FN_k * N_j / (N - N_k)
    fp  = {}
    for j in RELS:
        if j not in n_k:
            fp[j] = 0; continue
        fp[j] = sum(fn.get(k, 0) * n_k.get(j, 0) / max(n - n_k.get(k, 0), 1)
                    for k in RELS if k != j)
    prec = {k: tp[k] / max(tp[k] + fp[k], 1e-9) for k in RELS}
    rec  = {k: tp[k] / max(n_k.get(k, 0), 1e-9)  for k in RELS}
    f1   = {k: 2*prec[k]*rec[k] / max(prec[k]+rec[k], 1e-9) for k in RELS}
    n_rels = len([k for k in RELS if k in n_k])
    return {
        "MRR":        mrr_sum / n,
        "MR":         rank_sum / n,
        "Micro-H@1":  h1 / n,
        "Macro-H@1":  np.mean([tp[k]/n_k[k] for k in RELS if k in n_k]),
        "Accuracy":   h1 / n,
        "Precision":  np.mean([prec[k] for k in RELS if k in n_k]),
        "Recall":     np.mean([rec[k]  for k in RELS if k in n_k]),
        "F1":         np.mean([f1[k]   for k in RELS if k in n_k]),
        "AUC":        None,   # not computable without saved probability scores
        "per_rel":    {k: float(np.mean(per_rel.get(k, [0]))) for k in RELS},
        "prec_est":   True,   # precision/F1 are estimates
    }


def stratified_split(df, train_r=0.8, val_r=0.1, seed=42):
    """Same split as the KGE scripts — identical random seed ensures the test set is the same."""
    tr, vl, te = [], [], []
    for _, grp in df.groupby(["domain", "relation"]):
        if len(grp) < 10:
            tr.append(grp); continue
        tv, t = train_test_split(grp, test_size=1-train_r-val_r, random_state=seed)
        t2, v = train_test_split(tv,  test_size=val_r/(train_r+val_r), random_state=seed)
        tr.append(t2); vl.append(v); te.append(t)
    df_train = (pd.concat(tr + vl).sample(frac=1, random_state=seed)
                  .reset_index(drop=True))
    df_test  = (pd.concat(te).sample(frac=1, random_state=seed)
                  .reset_index(drop=True))
    return df_train, df_test


import pandas as pd

def run_lr(emb_path: Path | None, csv_path: Path, strategy="all"):
    """
    Re-runs LR with predict_proba on the same train/test split.
    Returns MRR, MR, Micro-H@1, Macro-H@1, Accuracy, Precision, Recall, F1, AUC.
    emb_path=None → OneHot encoding.
    """
    df = (pd.read_csv(csv_path)
            .drop_duplicates(subset=["head","relation","tail"])
            .reset_index(drop=True))
    df_train, df_test = stratified_split(df)

    classes = sorted(df["relation"].unique())

    # ── build features ──────────────────────────────────────────────────────
    if emb_path is None:
        # OneHot over entity names
        all_ents = sorted(set(df["head"].tolist() + df["tail"].tolist()))
        ent2idx  = {e: i for i, e in enumerate(all_ents)}
        n_ents   = len(all_ents)

        def one_hot_feat(df_part):
            X, y = [], []
            for _, row in df_part.iterrows():
                h = np.zeros(n_ents, dtype=np.float32)
                t = np.zeros(n_ents, dtype=np.float32)
                if str(row["head"]) in ent2idx: h[ent2idx[str(row["head"])]] = 1
                if str(row["tail"]) in ent2idx: t[ent2idx[str(row["tail"])]] = 1
                X.append(np.concatenate([h, t]))
                y.append(str(row["relation"]))
            return np.array(X, dtype=np.float32), np.array(y)

        X_tr, y_tr = one_hot_feat(df_train)
        X_te, y_te = one_hot_feat(df_test)
    else:
        df_emb   = pd.read_csv(emb_path)
        emb_cols = [c for c in df_emb.columns if c.startswith("emb_")]
        lkp      = {r["entity_name"]: r[emb_cols].values.astype(np.float32)
                    for _, r in df_emb.iterrows()}

        def emb_feat(df_part):
            X, y = [], []
            for _, row in df_part.iterrows():
                h, t = str(row["head"]), str(row["tail"])
                if h not in lkp or t not in lkp: continue
                he, te = lkp[h], lkp[t]
                if strategy == "all":
                    feat = np.concatenate([he, te, he-te, he*te])
                else:
                    feat = np.concatenate([he, te])
                X.append(feat); y.append(str(row["relation"]))
            return np.array(X, dtype=np.float32), np.array(y)

        X_tr, y_tr = emb_feat(df_train)
        X_te, y_te = emb_feat(df_test)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                              random_state=42, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    proba  = clf.predict_proba(X_te)   # (N, n_classes)
    cls_   = clf.classes_

    # ranking metrics from probabilities
    mrr_s = mr_s = h1 = 0
    for true_rel, probs in zip(y_te, proba):
        ranked = cls_[np.argsort(-probs)]
        rank   = int(np.where(ranked == true_rel)[0][0]) + 1
        mrr_s += 1/rank;  mr_s += rank;  h1 += (rank == 1)
    n = len(y_te)

    # classification metrics from hard predictions
    y_pred = clf.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    p, r, f, _ = precision_recall_fscore_support(y_te, y_pred,
                                                  average="macro", zero_division=0)
    # per-rel recall ≡ macro-H@1
    macro_h1 = r

    # AUC (one-vs-rest)
    lb   = LabelBinarizer().fit(cls_)
    y_bin = lb.transform(y_te)
    auc  = roc_auc_score(y_bin, proba, multi_class="ovr", average="macro")

    return {
        "MRR":       mrr_s / n,
        "MR":        mr_s  / n,
        "Micro-H@1": h1    / n,
        "Macro-H@1": macro_h1,
        "Accuracy":  acc,
        "Precision": p,
        "Recall":    r,
        "F1":        f,
        "AUC":       auc,
        "per_rel":   {},
        "prec_est":  False,
    }


def load_takashima(json_path: Path):
    labels = {
        "FreezedBertRgcnMlp":  "BERT-RGCN\n(frozen)",
        "FinetunedBertRgcnMlp":"BERT-RGCN\n(finetuned)",
        "FreezedBertMlp":      "BERT-MLP\n(frozen)",
        "FinetunedBertMlp":    "BERT-MLP\n(finetuned)",
        "FinetunedBertCosSim": "BERT-CosSim\n(ft)",
        "TfidfLr":             "TF-IDF+LR",
        "Random":              "Random",
    }
    with open(json_path) as f:
        d = json.load(f)
    out = {}
    for key, stats in d.get("statistics", {}).items():
        lbl = labels.get(key, key)
        g   = lambda m: stats.get(m, {}).get("mean", 0.0)
        auc = g("auc")
        out[lbl] = {
            "MRR":       g("mrr"),
            "MR":        g("mean_rank"),
            "Micro-H@1": g("hits@1_micro") or g("hits@1"),
            "Macro-H@1": g("recall"),
            "Accuracy":  g("accuracy"),
            "Precision": g("precision"),
            "Recall":    g("recall"),
            "F1":        g("f1"),
            "AUC":       auc if auc > 0.0 else None,
            "per_rel":   {},
            "prec_est":  False,
        }
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Collect all data
# ══════════════════════════════════════════════════════════════════════════════
KGE_MODELS = {
    "TransE":   "transe_sup_output",
    "RotatE":   "rotate_sup_output",
    "ComplEx":  "complex_sup_output",
    "DistMult": "distmult_sup_output",
    "ConvKB":   "convkb_sup_output",
}
LR_MODELS = {
    "ComplEx+LR":  "complex_sup_output",
    "DistMult+LR": "distmult_sup_output",
    "OneHot+LR":   None,
}

kge_data: dict[str, dict] = {}
for name, folder in KGE_MODELS.items():
    m = load_kge(ABA_DIR / "outputs" / folder / "relation_eval_ranked.csv")
    if m: kge_data[name] = m

print("Re-running LR classifiers with predict_proba (for MRR / MR / AUC)...")
lr_data: dict[str, dict] = {}
for name, folder in LR_MODELS.items():
    print(f"  {name} ...", end=" ", flush=True)
    emb = ABA_DIR / "outputs" / folder / "entity_embeddings.csv" if folder else None
    m   = run_lr(emb, CSV_PATH, strategy="all")
    lr_data[name] = m
    print(f"MRR={m['MRR']:.4f}  MR={m['MR']:.4f}  AUC={m['AUC']:.4f}")

tak_data: dict[str, dict] = load_takashima(TAKASHIMA_EXP)

all_models: dict[str, dict] = {**kge_data, **lr_data, **tak_data}
all_names  = list(all_models.keys())
all_colors = [C.get(n, "#999") for n in all_names]
all_x      = np.arange(len(all_names))

sep_kge = len(kge_data)  - 0.5   # between KGE and LR
sep_lr  = len(kge_data) + len(lr_data) - 0.5  # between LR and Takashima


# ══════════════════════════════════════════════════════════════════════════════
# Print tables
# ══════════════════════════════════════════════════════════════════════════════
def pval(v, est=False, na=False):
    if na   : return "    N/A"
    if v is None: return "    N/A"
    return f"{'~' if est else ' '}{v:>6.4f}"

print("\n" + "="*105)
print("TABLE 1 — Rank-Based Metrics  (all 15 models)")
print("  KGE: from CSV   LR: re-run predict_proba   TAK: exp_20260528_3R JSON  5-fold CV mean")
print("="*105)
print(f"{'Model':<22} {'Src':<4} {'MRR':>7} {'MR':>7} {'Micro-H@1':>10} {'Macro-H@1':>10}")
print("-"*65)
for name, src in [(n,"KGE") for n in kge_data] + \
                 [(n,"LR")  for n in lr_data]  + \
                 [(n,"TAK") for n in tak_data]:
    m = all_models[name]
    print(f"{name.replace(chr(10),' '):<22} {src:<4} "
          f"{m['MRR']:>7.4f} {m['MR']:>7.4f} "
          f"{m['Micro-H@1']:>10.4f} {m['Macro-H@1']:>10.4f}")

print("\n" + "="*105)
print("TABLE 2 — Classification Metrics  (3-class macro-averaged)")
print("  KGE Precision/F1: ~ estimated (proportional error distribution); AUC: N/A")
print("  LR: exact from predict_proba + eval CSV   TAK: exp_20260528_3R JSON")
print("="*105)
print(f"{'Model':<22} {'Src':<4} {'Acc':>7} {'Prec~':>7} {'Recall':>7} {'F1~':>7} {'AUC':>7}")
print("-"*65)
for name, src in [(n,"KGE") for n in kge_data] + \
                 [(n,"LR")  for n in lr_data]  + \
                 [(n,"TAK") for n in tak_data]:
    m   = all_models[name]
    est = m.get("prec_est", False)
    print(f"{name.replace(chr(10),' '):<22} {src:<4} "
          f"{pval(m['Accuracy'])} "
          f"{pval(m['Precision'], est=est)} "
          f"{pval(m['Recall'])} "
          f"{pval(m['F1'], est=est)} "
          f"{pval(m['AUC'], na=(m['AUC'] is None))}")


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════
def label_bars(ax, bars, values, fmt="{:.3f}", fs=6.2, pad_frac=0.012,
               ylim_top=1.0, na_idx=None, est_idx=None):
    na_idx  = set(na_idx  or [])
    est_idx = set(est_idx or [])
    for i, bar in enumerate(bars):
        v   = values[i]
        xc  = bar.get_x() + bar.get_width() / 2
        pad = pad_frac * ylim_top
        if i in na_idx or v is None:
            ax.text(xc, pad, "N/A", ha="center", va="bottom",
                    fontsize=fs, color="#aaa", style="italic")
        else:
            prefix = "~" if i in est_idx else ""
            ax.text(xc, v + pad, prefix + fmt.format(v),
                    ha="center", va="bottom", fontsize=fs, color="#333")


def simple_bar(ax, x, values, colors, ylim, ylabel, title,
               ticks, rot=40, fmt="{:.3f}", fs=6.2,
               hline=None, hline_lbl="", na_idx=None, est_idx=None):
    safe_v = [v if v is not None else 0.0 for v in values]
    bars = ax.bar(x, safe_v, color=colors, width=0.6,
                  edgecolor="white", linewidth=0.35, zorder=3)
    label_bars(ax, bars, values, fmt=fmt, fs=fs, ylim_top=ylim[1],
               na_idx=na_idx, est_idx=est_idx)
    if hline is not None:
        ax.axhline(hline, color="#E53935", linewidth=1.0, linestyle="--", zorder=4)
        ax.text(len(x)-0.5, hline + 0.012*ylim[1], hline_lbl,
                color="#E53935", fontsize=6.0, ha="right", va="bottom")
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_title(title, fontsize=10, pad=4)
    ax.yaxis.grid(True, color="#e5e5e5", zorder=0, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels(ticks, rotation=rot, ha="right", fontsize=7)
    return bars


def add_seps(ax, ylim):
    for xv, la, lb in [(sep_kge, "KGE", "LR"), (sep_lr, "LR", "Takashima")]:
        ax.axvline(xv, color="#bbb", linewidth=0.9, linestyle="--", zorder=2)
        ax.text(xv-0.15, ylim[1]*0.985, la, ha="right", va="top",
                fontsize=5.8, color="#888")
        ax.text(xv+0.15, ylim[1]*0.985, lb, ha="left",  va="top",
                fontsize=5.8, color="#888")


# index helpers
def idxs_of(names_list, condition):
    return [i for i, n in enumerate(names_list) if condition(n)]

kge_idx  = idxs_of(all_names, lambda n: n in kge_data)
lr_idx   = idxs_of(all_names, lambda n: n in lr_data)
tak_idx  = idxs_of(all_names, lambda n: n in tak_data)

auc_na   = idxs_of(all_names, lambda n: all_models[n]["AUC"] is None)
prec_est = idxs_of(all_names, lambda n: all_models[n].get("prec_est", False))

tick_all  = [n for n in all_names]   # newlines kept for multiline labels


# ══════════════════════════════════════════════════════════════════════════════
# Figure  (3 × 3)
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(28, 18))
fig.suptitle(
    "Model Comparison — KGE  |  LR on KGE embeddings  |  Takashima BERT/RGCN   "
    "[ABA mining + exp_20260528_3R, 3-class, 15 models]",
    fontsize=12.5, fontweight="bold", y=0.997,
)
gs = GridSpec(3, 3, figure=fig,
              hspace=0.78, wspace=0.34,
              left=0.052, right=0.978,
              top=0.965, bottom=0.08)
axes = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)]

for ytxt, txt in [
    (0.970, "Rank-Based Metrics — all 15 models  "
            "(KGE from CSV  ·  LR re-run predict_proba  ·  Takashima from JSON)"),
    (0.645, "Classification Metrics — all 15 models  "
            "(3-class macro  ·  ~ = estimated for KGE  ·  N/A = not computable)"),
]:
    fig.text(0.5, ytxt, txt, ha="center", va="top", fontsize=9, color="#444",
             bbox=dict(boxstyle="round,pad=0.22", fc="#f7f7f7", ec="#ccc", lw=0.7))

# ── helper to extract metric vector ──────────────────────────────────────────
def vec(metric):
    return [all_models[n].get(metric) for n in all_names]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MRR | Mean Rank | Micro-H@1
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
simple_bar(axes[0][0], all_x, vec("MRR"), all_colors,
           ylim=(0, 1.15), ylabel="MRR", title="MRR",
           ticks=tick_all, hline=1/3, hline_lbl="Random (0.333)")
add_seps(axes[0][0], (0, 1.15))

mr_top = max(v for v in vec("MR") if v) * 1.22 + 0.05
simple_bar(axes[0][1], all_x, vec("MR"), all_colors,
           ylim=(0, mr_top), ylabel="Mean Rank  (↓ better)",
           title="Mean Rank", ticks=tick_all, fmt="{:.3f}")
axes[0][1].axhline(1.0, color="#2E7D32", linewidth=0.9, linestyle=":", zorder=4)
axes[0][1].text(len(all_x)-0.5, 1.0+0.012*mr_top, "Perfect=1",
                color="#2E7D32", fontsize=6.0, ha="right", va="bottom")
add_seps(axes[0][1], (0, mr_top))

simple_bar(axes[0][2], all_x, vec("Micro-H@1"), all_colors,
           ylim=(0, 1.15), ylabel="Hits@1 (micro)",
           title="Hits@1 — Micro  (= Accuracy)", ticks=tick_all,
           hline=1/3, hline_lbl="Random (0.333)")
add_seps(axes[0][2], (0, 1.15))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Macro-H@1 | Per-rel H@1 (KGE) | Prec–Recall scatter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
simple_bar(axes[1][0], all_x, vec("Macro-H@1"), all_colors,
           ylim=(0, 1.15), ylabel="Hits@1 (macro)",
           title="Hits@1 — Macro  (= macro Recall)", ticks=tick_all,
           hline=1/3, hline_lbl="Random (0.333)")
add_seps(axes[1][0], (0, 1.15))

# per-relation H@1 — KGE only (grouped)
ax11 = axes[1][1]
kge_names = list(kge_data.keys())
kge_x5    = np.arange(len(kge_names))
w3 = 0.27
for i, rel in enumerate(RELS):
    xs   = kge_x5 + (i-1)*w3
    vals = [kge_data[n]["per_rel"].get(rel, 0.0) for n in kge_names]
    bars = ax11.bar(xs, vals, width=w3, color=C[rel],
                    edgecolor="white", linewidth=0.35, zorder=3, label=rel)
    for bar, v in zip(bars, vals):
        ax11.text(bar.get_x()+bar.get_width()/2, v+0.013,
                  f"{v:.3f}", ha="center", va="bottom", fontsize=5.8, color="#333")
ax11.set_ylim(0, 1.22)
ax11.set_ylabel("Hits@1");  ax11.set_title("Per-Relation Hits@1  (KGE only)")
ax11.yaxis.grid(True, color="#e5e5e5", zorder=0, linewidth=0.6)
ax11.set_axisbelow(True)
ax11.set_xticks(kge_x5)
ax11.set_xticklabels(kge_names, rotation=40, ha="right", fontsize=7)
ax11.legend(fontsize=6.5, loc="lower right",
            handles=[mpatches.Patch(color=C[r], label=r) for r in RELS])

# Prec–Recall scatter
ax12 = axes[1][2]
for i, name in enumerate(all_names):
    m   = all_models[name]
    prec = m.get("Precision"); rec = m.get("Recall")
    if prec is None or rec is None: continue
    col  = C.get(name, "#999")
    mkr  = "s" if name in kge_data else ("o" if name in lr_data else "^")
    ax12.scatter(rec, prec, color=col, s=100, marker=mkr, zorder=5,
                 edgecolors="white", linewidths=0.8)
    ax12.annotate(name.replace("\n"," "), (rec, prec),
                  textcoords="offset points", xytext=(5,3),
                  fontsize=6.5, color=col)
ax12.set_xlim(-0.05, 1.1); ax12.set_ylim(-0.05, 1.1)
ax12.set_xlabel("Recall  (macro, 3-class)", fontsize=8)
ax12.set_ylabel("Precision  (macro, 3-class  ~ = est.)", fontsize=8)
ax12.set_title("Precision vs. Recall\n(■ KGE  ● LR  ▲ Takashima  ~ = KGE estimated)")
ax12.grid(True, color="#e5e5e5", zorder=0, linewidth=0.6); ax12.set_axisbelow(True)
ax12.plot([0,1],[0,1], color="#ccc", linewidth=0.8, linestyle=":", zorder=1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# F1 | AUC | Precision
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
simple_bar(axes[2][0], all_x, vec("F1"), all_colors,
           ylim=(0, 1.12), ylabel="F1  (macro, 3-class)",
           title="F1  (~ = KGE estimated)", ticks=tick_all,
           est_idx=prec_est)
add_seps(axes[2][0], (0, 1.12))

# AUC
auc_vals = [all_models[n]["AUC"] if all_models[n]["AUC"] is not None else 0.0
            for n in all_names]
simple_bar(axes[2][1], all_x, auc_vals, all_colors,
           ylim=(0, 1.12), ylabel="AUC-ROC  (macro OvR)",
           title="AUC-ROC  (N/A for KGE)", ticks=tick_all,
           hline=0.5, hline_lbl="Random (0.5)",
           na_idx=auc_na)
add_seps(axes[2][1], (0, 1.12))
axes[2][1].set_title("AUC-ROC  (N/A for KGE  ·  macro one-vs-rest)", fontsize=9)

# Precision
simple_bar(axes[2][2], all_x, vec("Precision"), all_colors,
           ylim=(0, 1.12), ylabel="Precision  (macro, 3-class)",
           title="Precision  (~ = KGE estimated)", ticks=tick_all,
           est_idx=prec_est)
add_seps(axes[2][2], (0, 1.12))

# ── global legend ─────────────────────────────────────────────────────────────
items = ([(n, C[n], "s") for n in kge_data] +
         [(n.replace("\n","+"), C[n], "o") for n in lr_data] +
         [(n.replace("\n"," "), C.get(n,"#999"), "^") for n in tak_data])
handles = [plt.scatter([],[],color=col,marker=mk,s=55,label=lbl,
                        edgecolors="white",linewidths=0.5)
           for lbl,col,mk in items]
fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=7,
           frameon=True, framealpha=0.92, bbox_to_anchor=(0.5, 0.005))

# ── save figures ──────────────────────────────────────────────────────────────
out_png = ABA_DIR / "outputs" / "compare_all_models.png"
out_pdf = ABA_DIR / "outputs" / "compare_all_models.pdf"
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")

# ── save results JSON ─────────────────────────────────────────────────────────
import datetime

def _clean(v):
    """Convert numpy scalars and None to JSON-serialisable types."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return v

out_json = ABA_DIR / "outputs" / "compare_all_models_results.json"
json_payload = {
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "takashima_experiment": str(TAKASHIMA_EXP),
    "models": {},
}

for name in all_names:
    m = all_models[name]
    if name in kge_data:
        src, group = "KGE", "aba_mining_kge"
    elif name in lr_data:
        src, group = "LR", "aba_mining_lr"
    else:
        src, group = "Takashima", "takashima"

    json_payload["models"][name.replace("\n", " ")] = {
        "source":   src,
        "group":    group,
        "metrics": {
            "MRR":        _clean(m.get("MRR")),
            "MR":         _clean(m.get("MR")),
            "Micro-H@1":  _clean(m.get("Micro-H@1")),
            "Macro-H@1":  _clean(m.get("Macro-H@1")),
            "Accuracy":   _clean(m.get("Accuracy")),
            "Precision":  _clean(m.get("Precision")),
            "Recall":     _clean(m.get("Recall")),
            "F1":         _clean(m.get("F1")),
            "AUC":        _clean(m.get("AUC")),
        },
        "notes": {
            "precision_estimated": bool(m.get("prec_est", False)),
            "auc_available":       m.get("AUC") is not None,
        },
        "per_rel_hits1": {
            k: _clean(v)
            for k, v in m.get("per_rel", {}).items()
        },
    }

with open(out_json, "w") as f:
    json.dump(json_payload, f, indent=2)
print(f"Saved: {out_json}")

plt.show()

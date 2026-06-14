"""
Auto-reading model comparison visualization between KGE models and LR
============================================
Reads results directly from:
  1. relation_eval_ranked.csv  in each model's output folder  (KGE models)
  2. sup_experiment_results.xlsx                               (LR baseline)

  Claude-Assisted
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        9,
    'axes.titlesize':   10,
    'axes.labelsize':   9,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'figure.dpi':       150,
})

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "data" / "experiment_results" / "sup_experiment_results.xlsx"

# Map: display name → output folder name
KGE_MODELS = {
    'TransE':   'transe_sup_output',
    'RotatE':   'rotate_sup_output',
    'DistMult': 'distmult_sup_output',
    'ComplEx':  'complex_sup_output',
    'ConvKB':   'convkb_sup_output',
}

RELATIONS = ['CONTRARY_TO', 'NOT_CONTRARY', 'SUPPORT']

MODEL_COLORS = {
    'TransE':    '#378ADD',
    'RotatE':    '#1D9E75',
    'DistMult':  '#BA7517',
    'ComplEx':   '#534AB7',
    'ConvKB':    '#D85A30',
    'LR-KGE':    '#888780',   
    'LR-OneHot': '#C0A060',   
}

REL_COLORS = {
    'CONTRARY_TO':  '#D85A30',
    'NOT_CONTRARY': '#534AB7',
    'SUPPORT':      '#1D9E75',
}


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════
def load_kge_metrics(model_name, folder_name):
    """
    Load metrics from relation_eval_ranked.csv in the model's output folder.
    Computes MRR, Macro-Hits@1, Micro-Hits@1, Mean Rank, and per-relation Hits@1.
    Returns a dict or None if file not found.
    """
    csv_path = ROOT / "outputs" / folder_name / "relation_eval_ranked.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {model_name}: {csv_path} not found")
        return None

    df = pd.read_csv(csv_path)

    # Filter out inverse relations defensively
    df = df[~df['relation'].str.endswith('_inverse', na=False)].copy()

    if len(df) == 0:
        print(f"  [SKIP] {model_name}: no valid triples after filtering")
        return None

    # Aggregate metrics
    mrr       = df['reciprocal_rank'].mean()
    micro_h1  = df['hits_at_1'].mean()
    mean_rank = df['relation_rank'].mean()

    # Per-relation Hits@1
    per_rel = df.groupby('relation')['hits_at_1'].mean()
    macro_h1 = per_rel.mean()

    rel_h1 = {rel: per_rel.get(rel, 0.0) for rel in RELATIONS}

    print(f"  {model_name:<10}  MRR={mrr:.4f}  MacroH1={macro_h1:.4f}  "
          f"MeanRank={mean_rank:.4f}  "
          f"CT={rel_h1['CONTRARY_TO']:.3f}  "
          f"NC={rel_h1['NOT_CONTRARY']:.3f}  "
          f"S={rel_h1['SUPPORT']:.3f}")

    return {
        'mrr':        mrr,
        'micro_h1':   micro_h1,
        'macro_h1':   macro_h1,
        'mean_rank':  mean_rank,
        'contrary':   rel_h1['CONTRARY_TO'],
        'not_contrary': rel_h1['NOT_CONTRARY'],
        'support':    rel_h1['SUPPORT'],
    }


def load_lr_kge_metrics():
    """
    Load best LR-on-KGE-embeddings result from Excel.
    Looks for rows where Evaluator starts with 'LR-' (written by lr_baseline.py).
    Picks the single best Macro-H@1 across all source models and strategies.
    Per-relation not available — None.
    """
    if not RESULTS_FILE.exists():
        print(f"  [SKIP] LR-KGE: {RESULTS_FILE} not found")
        return None

    df = pd.read_excel(RESULTS_FILE)
    df.columns = df.columns.str.strip()

    # LR-on-KGE rows — Evaluator starts with "LR-" but NOT "OneHot"
    lr_rows = df[
        df['Evaluator'].str.startswith('LR-', na=False) &
        ~df['Evaluator'].str.contains('OneHot', na=False)
    ].copy()

    if len(lr_rows) == 0:
        print("  [SKIP] LR-KGE: no LR- rows found in Excel")
        return None

    # Best row = highest MRR column (stores macro-H@1 for LR rows)
    best_row  = lr_rows.loc[lr_rows['MRR'].idxmax()]
    macro_h1  = best_row['MRR']
    micro_h1  = best_row['Hits@1']
    mean_rank = 0.0   # not applicable for LR

    print(f"  {'LR-KGE':<10}  MacroH1={macro_h1:.4f}  MicroH1={micro_h1:.4f}  "
          f"(model={best_row['Model']}  evaluator={best_row['Evaluator']})")

    return {
        'mrr':          macro_h1,
        'micro_h1':     micro_h1,
        'macro_h1':     macro_h1,
        'mean_rank':    mean_rank,
        'contrary':     None,
        'not_contrary': None,
        'support':      None,
    }


def load_lr_onehot_metrics():
    """
    Load One-Hot LR result from onehot_sup_output/onehot_lr_eval.csv.
    Computes macro/micro Hits@1 directly from per-triple predictions.
    Falls back to Excel if CSV not found.
    """
    csv_path = ROOT / "outputs" / "onehot_sup_output" / "onehot_lr_eval.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        # per-relation Hits@1
        per_rel  = df.groupby('true_label')['correct'].mean()
        macro_h1 = per_rel.mean()
        micro_h1 = df['correct'].mean()
        print(f"  {'LR-OneHot':<10}  MacroH1={macro_h1:.4f}  MicroH1={micro_h1:.4f}  "
              f"(from onehot_lr_eval.csv)")
        return {
            'mrr':          macro_h1,
            'micro_h1':     micro_h1,
            'macro_h1':     macro_h1,
            'mean_rank':    0.0,
            'contrary':     None,
            'not_contrary': None,
            'support':      None,
        }

    # Fallback: read from Excel
    if not RESULTS_FILE.exists():
        print(f"  [SKIP] LR-OneHot: neither CSV nor Excel found")
        return None

    df = pd.read_excel(RESULTS_FILE)
    df.columns = df.columns.str.strip()
    rows = df[df['Evaluator'] == 'OneHot-LR'].copy()

    if len(rows) == 0:
        print("  [SKIP] LR-OneHot: no OneHot-LR rows in Excel")
        return None

    best_row  = rows.loc[rows['MRR'].idxmax()]
    macro_h1  = best_row['MRR']
    micro_h1  = best_row['Hits@1']
    print(f"  {'LR-OneHot':<10}  MacroH1={macro_h1:.4f}  MicroH1={micro_h1:.4f}  "
          f"(from Excel fallback)")
    return {
        'mrr':          macro_h1,
        'micro_h1':     micro_h1,
        'macro_h1':     macro_h1,
        'mean_rank':    0.0,
        'contrary':     None,
        'not_contrary': None,
        'support':      None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Load all models
# ══════════════════════════════════════════════════════════════════════════════
print("Loading results...")
print("-" * 70)

results = {}
for name, folder in KGE_MODELS.items():
    m = load_kge_metrics(name, folder)
    if m:
        results[name] = m

lr_kge = load_lr_kge_metrics()
if lr_kge:
    results['LR-KGE'] = lr_kge

lr_onehot = load_lr_onehot_metrics()
if lr_onehot:
    results['LR-OneHot'] = lr_onehot

if not results:
    raise RuntimeError(
        "No results found. Make sure you have run at least one model "
        "and that this script is in the project root folder."
    )

MODELS_AVAIL = list(results.keys())
N = len(MODELS_AVAIL)
COLORS = [MODEL_COLORS.get(m, '#888') for m in MODELS_AVAIL]

print(f"\nLoaded {N} models: {MODELS_AVAIL}")
print("-" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# Build arrays for plotting
# ══════════════════════════════════════════════════════════════════════════════
def arr(key):
    return [results[m].get(key) or 0.0 for m in MODELS_AVAIL]

macro_h1    = arr('macro_h1')
micro_h1    = arr('micro_h1')
mrr         = arr('mrr')
mean_rank   = arr('mean_rank')
contrary    = arr('contrary')
not_contrary= arr('not_contrary')
support     = arr('support')

X     = np.arange(N)
BAR_W = 0.25


# ══════════════════════════════════════════════════════════════════════════════
# Figure layout
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(16, 9))
fig.suptitle('Model evaluation — ABA contrary mining (relation prediction)',
             fontsize=12, fontweight='500', y=0.98)

gs = GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35,
              left=0.07, right=0.97, top=0.93, bottom=0.09)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])

MAJORITY = 1.0 / 3.0


def label_bars(ax, bars, fmt='{:.3f}', offset=0.008, fontsize=7):
    for bar in bars:
        h = bar.get_height()
        if h > 0.001:
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                    fmt.format(h), ha='center', va='bottom',
                    fontsize=fontsize, color='#444')


def rotated_xticks(ax, labels, fontsize=8):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=fontsize)


# ── Subplot 1: Macro-Hits@1 ───────────────────────────────────────────────────
bars1 = ax1.bar(X, macro_h1, color=COLORS, width=0.6,
                edgecolor='white', linewidth=0.5, zorder=3)
ax1.axhline(MAJORITY, color='#E24B4A', linewidth=1.2, linestyle='--',
            label=f'Majority baseline ({MAJORITY:.3f})', zorder=4)
label_bars(ax1, bars1)
ax1.set_ylim(0, 1.1)
ax1.set_ylabel('Macro-Hits@1')
ax1.set_title('Macro-Hits@1 (primary metric)')
ax1.yaxis.grid(True, color='#eee', zorder=0)
ax1.set_axisbelow(True)
rotated_xticks(ax1, MODELS_AVAIL)
ax1.legend(fontsize=7, loc='upper left')

# ── Subplot 2: Per-relation Hits@1 ───────────────────────────────────────────
# Only plot models that have per-relation data (KGE models, not LR)
kge_mask  = [i for i, m in enumerate(MODELS_AVAIL) if results[m]['contrary'] is not None]
kge_names = [MODELS_AVAIL[i] for i in kge_mask]
kge_x     = np.arange(len(kge_names))

b_ct  = ax2.bar(kge_x - BAR_W, [contrary[i]     for i in kge_mask],
                width=BAR_W, color=REL_COLORS['CONTRARY_TO'],
                edgecolor='white', linewidth=0.5, zorder=3)
b_nc  = ax2.bar(kge_x,          [not_contrary[i] for i in kge_mask],
                width=BAR_W, color=REL_COLORS['NOT_CONTRARY'],
                edgecolor='white', linewidth=0.5, zorder=3)
b_sup = ax2.bar(kge_x + BAR_W,  [support[i]      for i in kge_mask],
                width=BAR_W, color=REL_COLORS['SUPPORT'],
                edgecolor='white', linewidth=0.5, zorder=3)

ax2.set_ylim(0, 1.15)
ax2.set_ylabel('Hits@1')
ax2.set_title('Per-relation Hits@1 (KGE models)')
ax2.yaxis.grid(True, color='#eee', zorder=0)
ax2.set_axisbelow(True)
rotated_xticks(ax2, kge_names)
ax2.legend(fontsize=7, loc='upper left',
           handles=[mpatches.Patch(color=REL_COLORS[r], label=r) for r in RELATIONS])

# ── Subplot 3: MRR ────────────────────────────────────────────────────────────
bars3 = ax3.bar(X, mrr, color=COLORS, width=0.6,
                edgecolor='white', linewidth=0.5, zorder=3)
label_bars(ax3, bars3)
ax3.set_ylim(0, 1.1)
ax3.set_ylabel('MRR')
ax3.set_title('MRR  (LR: Macro-H1 used as proxy)')
ax3.yaxis.grid(True, color='#eee', zorder=0)
ax3.set_axisbelow(True)
rotated_xticks(ax3, MODELS_AVAIL)

# ── Subplot 4: Micro vs Macro Hits@1 ─────────────────────────────────────────
b_micro = ax4.bar(X - BAR_W / 2, micro_h1, width=BAR_W,
                  color=[c + 'aa' for c in COLORS],
                  edgecolor=COLORS, linewidth=1, zorder=3)
b_macro2= ax4.bar(X + BAR_W / 2, macro_h1, width=BAR_W,
                  color=COLORS, edgecolor='white', linewidth=0.5, zorder=3)
ax4.axhline(MAJORITY, color='#E24B4A', linewidth=1.2, linestyle='--', zorder=4)
ax4.set_ylim(0, 1.15)
ax4.set_ylabel('Hits@1')
ax4.set_title('Micro vs Macro Hits@1')
ax4.yaxis.grid(True, color='#eee', zorder=0)
ax4.set_axisbelow(True)
rotated_xticks(ax4, MODELS_AVAIL)
ax4.legend(fontsize=7, loc='upper left',
           handles=[mpatches.Patch(facecolor='#ccc', edgecolor='#555', label='Micro'),
                    mpatches.Patch(color='#555', label='Macro')])

# ── Subplot 5: Mean Rank (horizontal) ────────────────────────────────────────
y_pos = np.arange(N)
bars5 = ax5.barh(y_pos, mean_rank, color=COLORS,
                 edgecolor='white', linewidth=0.5, zorder=3, height=0.6)
for bar, val in zip(bars5, mean_rank):
    if val > 0.001:
        ax5.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{val:.3f}', va='center', fontsize=7, color='#444')
ax5.set_xlim(0, 3.5)
ax5.set_xlabel('Mean Rank (lower = better)')
ax5.set_title('Mean Rank — max possible = 3')
ax5.set_yticks(y_pos)
ax5.set_yticklabels(MODELS_AVAIL, fontsize=8)
ax5.xaxis.grid(True, color='#eee', zorder=0)
ax5.set_axisbelow(True)
ax5.axvline(1.0, color='#1D9E75', linewidth=1, linestyle=':', alpha=0.8,
            label='Perfect = 1.0')
ax5.legend(fontsize=7)

# ── Subplot 6: Scatter Macro-H1 vs MRR ───────────────────────────────────────
for i, model in enumerate(MODELS_AVAIL):
    xv, yv = macro_h1[i], mrr[i]
    if xv > 0 or yv > 0:
        ax6.scatter(xv, yv, color=COLORS[i], s=90, zorder=5,
                    edgecolors='white', linewidths=0.8)
        ax6.annotate(model, (xv, yv),
                     textcoords='offset points', xytext=(6, 4),
                     fontsize=8, color=COLORS[i])
ax6.axvline(MAJORITY, color='#E24B4A', linewidth=1, linestyle='--',
            alpha=0.6, label='Majority baseline')
ax6.plot([0, 1], [0, 1], color='#ddd', linewidth=0.8, linestyle=':', zorder=1)
ax6.set_xlim(-0.05, 1.1)
ax6.set_ylim(-0.05, 1.1)
ax6.set_xlabel('Macro-Hits@1')
ax6.set_ylabel('MRR')
ax6.set_title('Macro-Hits@1 vs MRR')
ax6.grid(True, color='#eee', zorder=0)
ax6.set_axisbelow(True)
ax6.legend(fontsize=7)

# ── Bottom legend ─────────────────────────────────────────────────────────────
legend_patches = [mpatches.Patch(color=MODEL_COLORS.get(m, '#888'), label=m)
                  for m in MODELS_AVAIL]
fig.legend(handles=legend_patches, loc='lower center', ncol=N,
           fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.01))

# ── Save ──────────────────────────────────────────────────────────────────────
OUT_PNG = ROOT / 'outputs' / 'results_comparison.png'
OUT_PDF = ROOT / 'outputs' / 'results_comparison.pdf'
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
plt.savefig(OUT_PDF, bbox_inches='tight')
print(f'\nSaved: {OUT_PNG}')
print(f'Saved: {OUT_PDF}')
plt.show()
"""Regenerate the manuscript figures from the corrected evaluation data.

Figure set for arXiv v2:
  fig1  component parameter pies              (reused unchanged from v1: structural)
  fig2  extraction artifact, reported vs corrected GSM8K   (new)
  fig3  corrected delta heatmaps, GSM8K-trained
  fig4  corrected delta heatmaps, UltraChat-trained
  fig5  effect sizes with 95% bootstrap CIs   (new; replaces the v1 radar and Pareto)

Palette: validated diverging pair blue <-> red with a neutral gray midpoint, and
categorical slots 1 (blue) and 2 (orange), which pass all-pairs CVD and
normal-vision separation. Every heatmap cell is annotated, so the figures remain
readable in grayscale and colour never carries meaning alone.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "reanalysis")
OUT = os.path.join(ROOT, "arxiv_v2", "figures")
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------- palette
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
GRAY_MID = "#f0efec"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

DIVERGING = LinearSegmentedColormap.from_list("bl_gy_rd", [RED, GRAY_MID, BLUE])

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK2,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "legend.frameon": False,
})

MODELS = {
    "qwen3_5_0_8b_base": "Qwen3.5-0.8B (sequential)",
    "falcon_h1_0_5b_base": "Falcon-H1-0.5B (parallel)",
}
ORDER = {
    "qwen3_5_0_8b_base": ["all_layers", "softmax_only", "gdn_only", "mlp_only",
                          "softmax_plus_mlp", "gdn_plus_mlp"],
    "falcon_h1_0_5b_base": ["all_eligible", "attention_only", "ssm_only", "mlp_only",
                            "attention_plus_mlp", "ssm_plus_mlp"],
}
BENCHES = ["mmlu", "gsm8k", "arc_challenge", "hellaswag"]
BLABEL = {"mmlu": "MMLU", "gsm8k": "GSM8K", "arc_challenge": "ARC-C",
          "hellaswag": "HellaSwag"}


def tidy(name):
    return name.replace("_", " + ") if "_plus_" in name else name.replace("_", " ")


def cond_label(c):
    return c.replace("_plus_", " + ").replace("_", "-")


df = pd.read_csv(os.path.join(DATA, "all_results_corrected.csv"))


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, stem + "." + ext), dpi=200,
                    bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", stem)


# ============================================================ fig2: artifact
def fig_artifact():
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    for ax, (mk, title) in zip(axes, MODELS.items()):
        sub = df[(df.model_key == mk) & (df.benchmark == "gsm8k")
                 & (df.train_dataset.isin(["base", "gsm8k_train"]))]
        rows = [("base", sub[sub.condition == "__base__"].iloc[0])]
        for c in ORDER[mk]:
            r = sub[(sub.condition == c) & (sub.train_dataset == "gsm8k_train")]
            if len(r):
                rows.append((cond_label(c), r.iloc[0]))
        labels = [r[0] for r in rows]
        rep = [r[1].acc_reported * 100 for r in rows]
        cor = [r[1].acc_corrected * 100 for r in rows]
        y = np.arange(len(rows))
        h = 0.36
        ax.barh(y + h / 2 + 0.01, rep, height=h, color=ORANGE,
                label="as reported (v1)")
        ax.barh(y - h / 2 - 0.01, cor, height=h, color=BLUE,
                label="corrected (v2)")
        for yi, (a, b) in enumerate(zip(rep, cor)):
            ax.text(a + 1.0, yi + h / 2 + 0.01, f"{a:.1f}", va="center",
                    ha="left", fontsize=7.5, color=INK2)
            ax.text(b + 1.0, yi - h / 2 - 0.01, f"{b:.1f}", va="center",
                    ha="left", fontsize=7.5, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 75)
        ax.set_xlabel("GSM8K accuracy (%)")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", alpha=0.9)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=2, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, "fig_extraction_artifact")


# ========================================================= fig3/4: heatmaps
def fig_heatmap(domain, stem):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.6),
                             gridspec_kw={"wspace": 0.42})
    vmax = 0
    mats = {}
    for mk in MODELS:
        conds = [c for c in ORDER[mk]
                 if len(df[(df.model_key == mk) & (df.condition == c)
                           & (df.train_dataset == domain)])]
        m = np.full((len(conds), len(BENCHES)), np.nan)
        for i, c in enumerate(conds):
            for j, b in enumerate(BENCHES):
                r = df[(df.model_key == mk) & (df.condition == c)
                       & (df.train_dataset == domain) & (df.benchmark == b)]
                if len(r):
                    m[i, j] = r.iloc[0].delta_corrected * 100
        mats[mk] = (conds, m)
        vmax = max(vmax, np.nanmax(np.abs(m)))
    vmax = float(np.ceil(vmax / 2) * 2)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    for ax, (mk, title) in zip(axes, MODELS.items()):
        conds, m = mats[mk]
        im = ax.imshow(m, cmap=DIVERGING, norm=norm, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if np.isnan(m[i, j]):
                    continue
                shade = abs(m[i, j]) / vmax
                ax.text(j, i, f"{m[i, j]:+.1f}", ha="center", va="center",
                        fontsize=8, color="white" if shade > 0.62 else INK)
        ax.set_xticks(range(len(BENCHES)))
        ax.set_xticklabels([BLABEL[b] for b in BENCHES], fontsize=8.5)
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels([cond_label(c) for c in conds], fontsize=8.5)
        ax.set_title(title, loc="left")
        ax.set_xticks(np.arange(-.5, len(BENCHES), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(conds), 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("accuracy change vs base (pp)", fontsize=8.5, color=INK2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, color=MUTED)
    save(fig, stem)


# ====================================================== fig5: effect sizes
def fig_effects():
    bs = pd.read_csv(os.path.join(DATA, "key_bootstrap_corrected.csv"))
    bs = bs.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for i, r in bs.iterrows():
        is_qwen = r.model_key.startswith("qwen")
        col = BLUE if is_qwen else ORANGE
        mk = "o" if is_qwen else "s"
        ax.plot([r.ci_low_pp, r.ci_high_pp], [i, i], color=col, lw=1.6,
                solid_capstyle="butt", zorder=2)
        ax.plot([r.mean_diff_pp], [i], mk, color=col, ms=7,
                markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)
    ax.axvline(0, color=BASELINE, lw=1.0, zorder=1)
    def pretty(lab):
        # model identity is carried by colour + marker, so drop it from the text
        # but keep the training domain and benchmark, which disambiguate rows
        body = lab.split(" ", 1)[1] if lab.split(" ", 1)[0] in ("Qwen", "Falcon") else lab
        ctx, comp = body.split(": ", 1)
        comp = comp.replace("_plus_", " + ").replace("_", "-")
        return comp + "   (" + ctx + ")"

    ax.set_yticks(range(len(bs)))
    ax.set_yticklabels([pretty(l) for l in bs.label], fontsize=8.5)
    ax.set_xlabel("difference in accuracy (percentage points), corrected labels")
    ax.grid(axis="x", alpha=0.9)
    ax.set_axisbelow(True)
    ax.margins(y=0.04)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], color=BLUE, marker="o", ls="-", ms=7,
               label="Qwen3.5-0.8B (sequential)"),
        Line2D([], [], color=ORANGE, marker="s", ls="-", ms=7,
               label="Falcon-H1-0.5B (parallel)")],
        loc="upper center", ncol=2, fontsize=9, labelcolor=INK2,
        bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    save(fig, "fig_effect_sizes")


fig_artifact()
fig_heatmap("gsm8k_train", "fig_heatmap_gsm8k_corrected")
fig_heatmap("ultrachat", "fig_heatmap_ultrachat_corrected")
fig_effects()
print("figures ->", OUT)

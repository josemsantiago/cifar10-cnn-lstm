"""Publication-quality figures for the report. Pure matplotlib, clean APA-friendly
styling matched to the Week 2 and Week 3 projects (150 dpi, no top/right spines,
subtle grid).
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import MaxNLocator

import config as C

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

INK = "#1b2a41"
FOCAL = "#c0392b"       # the CNN-LSTM, the model under study
CTRL = "#2c6fbb"        # controls
ACCENT = "#4d9078"
BAR = ["#c0392b", "#2c6fbb", "#4d9078", "#e0a458", "#7d5ba6"]


def _colors(names):
    """The focal model is always red; the controls take the cool palette."""
    cool = ["#2c6fbb", "#4d9078", "#e0a458", "#7d5ba6"]
    out, i = [], 0
    for n in names:
        if n == "CNN-LSTM":
            out.append(FOCAL)
        else:
            out.append(cool[i % len(cool)]); i += 1
    return out


# ===================================================== fig 1: the data
def fig_data(samples, aug_examples, path):
    """Panel A stacks n_per sample rows under each class column; Panel B is one
    image beside independent draws of the augmentation policy."""
    fig = plt.figure(figsize=(9.2, 4.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[len(next(iter(samples.values()))), 1.0],
                          hspace=0.68)

    def bare(ax):
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)

    ax = fig.add_subplot(gs[0])                          # classes as columns
    n_per = len(next(iter(samples.values())))
    grid = np.hstack([np.vstack(list(samples[c])) for c in C.CLASSES])
    ax.imshow(grid)
    ax.set_xticks([C.IMG_SIZE * (k + 0.5) for k in range(len(C.CLASSES))])
    ax.set_xticklabels(C.CLASSES, fontsize=8, rotation=35, ha="right")
    ax.set_yticks([]); bare(ax)
    ax.set_title(f"A. CIFAR-10 classes ({n_per} training images each)",
                 loc="left", fontweight="bold", fontsize=10)

    ax = fig.add_subplot(gs[1])                          # augmentation strip
    ax.imshow(np.hstack(aug_examples))
    ax.set_xticks([C.IMG_SIZE * (k + 0.5) for k in range(len(aug_examples))])
    ax.set_xticklabels(["original"] + [str(k) for k in range(1, len(aug_examples))],
                       fontsize=8)
    ax.set_yticks([]); bare(ax)
    ax.set_title("B. One training image and nine draws of the crop-and-flip policy",
                 loc="left", fontweight="bold", fontsize=10)

    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# ===================================================== fig 2: architecture diagram
def fig_architecture(summary_rows, path):
    """Data-flow diagram of the CNN-LSTM: image, convolutional trunk, the row
    sequence, the recurrent layer, and the classifier."""
    fig, ax = plt.subplots(figsize=(11.5, 3.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")

    stages = [
        ("Input\nimage", "3 x 32 x 32", "#e8eef5"),
        ("Conv block 1\n2x (3x3, BN, ReLU)\nmaxpool, drop 0.2", "64 x 16 x 16", "#cfe0f0"),
        ("Conv block 2\n2x (3x3, BN, ReLU)\nmaxpool, drop 0.3", "128 x 8 x 8", "#cfe0f0"),
        ("Conv block 3\n2x (3x3, BN, ReLU)\ndrop 0.4", "256 x 8 x 8", "#cfe0f0"),
        ("1x1 conv\nchannel\ncompression", "128 x 8 x 8", "#cfe0f0"),
        ("Row\nsequencing", "8 steps x 1024", "#f6e6c8"),
        ("Bidirectional\nLSTM\n256 units/dir", "8 x 512", "#f3c9c2"),
        ("Final state\n+ dropout 0.5\n+ linear", "10 logits", "#dceee5"),
    ]
    w, gap = 10.0, 2.3
    x0 = 1.2
    for k, (label, shape, colour) in enumerate(stages):
        x = x0 + k * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 9), w, 12, boxstyle="round,pad=0.35",
                                    linewidth=0.8, edgecolor=INK, facecolor=colour))
        ax.text(x + w / 2, 15, label, ha="center", va="center", fontsize=6.8, color=INK)
        ax.text(x + w / 2, 6.6, shape, ha="center", va="center", fontsize=7.4,
                color="#555", style="italic")
        if k:
            ax.add_patch(FancyArrowPatch((x - gap + 0.35, 15), (x - 0.35, 15),
                                         arrowstyle="-|>", mutation_scale=9,
                                         linewidth=0.9, color="#666"))

    ax.annotate("", xy=(60.9, 23.4), xytext=(13.0, 23.4),
                arrowprops=dict(arrowstyle="-", lw=0.8, color="#999"))
    ax.text(37.0, 24.4, "convolutional trunk — shared by all CNN models",
            ha="center", fontsize=8, color="#666")
    ax.annotate("", xy=(97.8, 23.4), xytext=(62.2, 23.4),
                arrowprops=dict(arrowstyle="-", lw=0.8, color="#999"))
    ax.text(80.0, 24.4, "recurrent head — the model under study",
            ha="center", fontsize=8, color="#666")

    total = sum(r["parameters"] for r in summary_rows)
    ax.text(50, 1.8, f"{total:,} trainable parameters", ha="center",
            fontsize=8.5, color=INK)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# ===================================================== fig 3: training curves
def fig_training(histories, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = list(histories.keys())
    cols = _colors(names)

    ax = axes[0]                                          # focal model, loss
    h = histories["CNN-LSTM"]
    ep = h["epoch"]
    ax.plot(ep, h["train_loss"], color=INK, lw=1.8, label="Training loss")
    ax.plot(ep, h["val_loss"], color=FOCAL, lw=1.6, ls="--", label="Validation loss")
    ax.axvline(h["best_epoch"], color="#888", ls=":", lw=1)
    ax.annotate(f"best epoch {h['best_epoch']}", (h["best_epoch"], max(h["train_loss"])),
                textcoords="offset points", xytext=(-72, -6), fontsize=8, color="#555")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("A. CNN-LSTM training and validation loss", loc="left",
                 fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]                                          # all models, val accuracy
    for name, col in zip(names, cols):
        h = histories[name]
        ax.plot(h["epoch"], h["val_acc"], color=col, lw=2.2 if name == "CNN-LSTM" else 1.4,
                label=name)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation accuracy")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("B. Validation accuracy by architecture", loc="left",
                 fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# ===================================================== fig 4: model comparison
def fig_model_comparison(cmp_df, path):
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    names = cmp_df["model"].tolist()
    x = np.arange(len(names)); w = 0.35
    ax.bar(x - w / 2, cmp_df["accuracy"], w, color=_colors(names), edgecolor="white",
           label="Test accuracy")
    ax.bar(x + w / 2, cmp_df["f1"], w, color=_colors(names), alpha=0.55,
           edgecolor="white", hatch="//", label="Macro F1")
    for xi, (a, f, p) in enumerate(zip(cmp_df["accuracy"], cmp_df["f1"],
                                       cmp_df["parameters"])):
        ax.text(xi - w / 2, a + 0.012, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(xi + w / 2, f + 0.012, f"{f:.3f}", ha="center", fontsize=8)
        ax.text(xi, 0.035, f"{p/1e6:.2f}M params", ha="center", fontsize=7.5, color="white")
    ax.axhline(0.10, color="#999", ls="--", lw=1)
    ax.text(len(names) - 0.5, 0.115, "random baseline (10%)", ha="right",
            fontsize=7.5, color="#777")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.02)
    ax.set_title("Held-out test performance (n = 10,000)", loc="left", fontsize=9.5)
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper right")
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# ===================================================== fig 5: confusion + per class
def fig_confusion_perclass(cm, per_class, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]                                          # row-normalized confusion
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(C.CLASSES))); ax.set_yticks(range(len(C.CLASSES)))
    ax.set_xticklabels(C.CLASSES, rotation=55, ha="right", fontsize=7.5)
    ax.set_yticklabels(C.CLASSES, fontsize=7.5)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(len(C.CLASSES)):
        for j in range(len(C.CLASSES)):
            if norm[i, j] >= 0.04:
                ax.text(j, i, f"{norm[i,j]*100:.0f}", ha="center", va="center",
                        fontsize=6.6, color="white" if norm[i, j] > 0.5 else INK)
    ax.grid(False)
    ax.set_title("A. CNN-LSTM confusion matrix (row %, test set)", loc="left",
                 fontweight="bold", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)

    ax = axes[1]                                          # per-class F1
    order = np.argsort([r["f1"] for r in per_class])
    names = [per_class[i]["class_name"] for i in order]
    vals = [per_class[i]["f1"] for i in order]
    ax.barh(names, vals, color=[FOCAL if v < 0.85 else CTRL for v in vals],
            edgecolor="white")
    macro = float(np.mean([r["f1"] for r in per_class]))
    ax.axvline(macro, color=INK, ls="--", lw=1)
    ax.text(macro + 0.005, -0.4, f"macro {macro:.3f}", fontsize=8, color=INK)
    for i, v in enumerate(vals):
        ax.text(v - 0.02, i, f"{v:.3f}", ha="right", va="center", fontsize=7.5,
                color="white")
    ax.set_xlim(0, 1.0); ax.set_xlabel("F1 score")
    ax.set_title("B. Per-class F1 (CNN-LSTM)", loc="left", fontweight="bold", fontsize=10)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


# ===================================================== fig 6: optimization grids
def fig_optimization(hidden_df, seq_df, aug_df, path):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))

    ax = axes[0]                                          # hidden-unit sweep
    ax.plot(hidden_df["hidden_units"], hidden_df["test_accuracy"], "o-",
            color=FOCAL, lw=1.8, label="Test accuracy")
    ax.plot(hidden_df["hidden_units"], hidden_df["val_accuracy"], "s--",
            color=CTRL, lw=1.4, label="Validation accuracy")
    for x, y, p in zip(hidden_df["hidden_units"], hidden_df["test_accuracy"],
                       hidden_df["parameters"]):
        ax.annotate(f"{p/1e6:.1f}M", (x, y), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=7, color="#666")
    ax.set_xscale("log", base=2)
    ax.set_xticks(hidden_df["hidden_units"])
    ax.set_xticklabels(hidden_df["hidden_units"])
    ax.set_xlabel("LSTM units per direction"); ax.set_ylabel("Accuracy")
    lo = float(min(hidden_df["test_accuracy"].min(), hidden_df["val_accuracy"].min()))
    hi = float(max(hidden_df["test_accuracy"].max(), hidden_df["val_accuracy"].max()))
    ax.set_ylim(lo - 0.10 * (hi - lo), hi + 0.05 * (hi - lo))
    ax.set_title("A. Recurrent capacity", loc="left", fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]                                          # sequencing ablation
    labels = [f"{r.axis}\n{'bi' if r.bidirectional else 'uni'}directional"
              for r in seq_df.itertuples()]
    ax.bar(labels, seq_df["test_accuracy"], color=[FOCAL] + [CTRL] * (len(labels) - 1),
           edgecolor="white", width=0.55)
    for i, v in enumerate(seq_df["test_accuracy"]):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    lo = float(seq_df["test_accuracy"].min())
    ax.set_ylim(lo - 0.03, float(seq_df["test_accuracy"].max()) + 0.025)
    ax.set_ylabel("Test accuracy")
    ax.set_title("B. Sequence formation", loc="left", fontweight="bold", fontsize=10)

    ax = axes[2]                                          # augmentation ablation
    x = np.arange(len(aug_df)); w = 0.35
    ax.bar(x - w / 2, aug_df["train_accuracy"], w, color="#b8c4d4",
           edgecolor="white", label="Final training accuracy")
    ax.bar(x + w / 2, aug_df["test_accuracy"], w, color=FOCAL,
           edgecolor="white", label="Test accuracy")
    for i, (tr, te) in enumerate(zip(aug_df["train_accuracy"], aug_df["test_accuracy"])):
        ax.text(i - w / 2, tr + 0.008, f"{tr:.3f}", ha="center", fontsize=7.5)
        ax.text(i + w / 2, te + 0.008, f"{te:.3f}", ha="center", fontsize=7.5)
        ax.annotate(f"gap {tr-te:+.3f}", (i, max(tr, te) + 0.055), ha="center",
                    fontsize=8, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["augmented" if a else "no augmentation"
                        for a in aug_df["augmented"]], fontsize=9)
    ax.set_ylim(0, 1.16); ax.set_ylabel("Accuracy")
    ax.set_title("C. Augmentation", loc="left", fontweight="bold", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)

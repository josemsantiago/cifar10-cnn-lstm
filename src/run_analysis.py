"""Orchestrator: run the whole CNN-LSTM image-classification study end to end.

Loads and verifies CIFAR-10, trains the CNN-LSTM and its three controls under one
shared recipe, scores them on the untouched official test split, then runs the
three optimization grids (recurrent capacity, sequence formation, augmentation)
and writes every figure, CSV table, and a single results.json.

    cd src && python3 run_analysis.py
"""
import argparse
import json
import platform
import time
import warnings

import numpy as np
import pandas as pd
import torch

import config as C
import data as D
import models as M
import plots as P
import train as T

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _round(o, nd=4):
    """Recursively round floats so results.json is clean and diff-friendly."""
    if isinstance(o, (float, np.floating)):
        return round(float(o), nd)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, list):
        return [_round(v, nd) for v in o]
    return o


def _fit(name_or_model, loaders, device, epochs, tag, **kwargs):
    """Build (if given a name), train, and score one model on the test split."""
    model = M.build(name_or_model, **kwargs) if isinstance(name_or_model, str) \
        else name_or_model
    tr, va, te = loaders
    model, hist = T.train_model(model, tr, va, device, epochs=epochs, tag=tag)
    metrics = T.evaluate(model, te, device)
    return model, hist, metrics


def main(epochs=C.EPOCHS):
    t0 = time.time()
    device = T.get_device()
    results = {"seed": C.SEED, "device": str(device), "torch": torch.__version__,
               "platform": platform.platform(), "epochs": epochs}
    print(f"Device: {device} | torch {torch.__version__} | seed {C.SEED} | {epochs} epochs\n")

    # ---------------------------------------------------------------- data
    desc = D.describe_data()
    desc.to_csv(C.OUT_DIR / "table_data_description.csv", index=False)
    mean, std = D.channel_statistics()
    results["data"] = dict(
        n_train=int(desc["train"].sum()), n_val=int(desc["validation"].sum()),
        n_test=int(desc["test"].sum()), n_classes=len(C.CLASSES),
        image_shape=[3, C.IMG_SIZE, C.IMG_SIZE],
        channel_mean=[float(v) for v in mean], channel_std=[float(v) for v in std])

    loaders = D.make_loaders(augment=True)
    P.fig_data(D.sample_images(6), _augmentation_strip(), C.FIG_DIR / "fig01_data.png")

    # ---------------------------------------------------------------- architecture
    summary = M.layer_summary(M.build("CNN-LSTM"))
    pd.DataFrame(summary).to_csv(C.OUT_DIR / "table_architecture.csv", index=False)
    results["architecture"] = summary
    P.fig_architecture(summary, C.FIG_DIR / "fig02_architecture.png")

    # ---------------------------------------------------------------- models
    print("=" * 74)
    print("Stage 1/2 — four architectures, one recipe")
    print("=" * 74)
    rows, histories, focal = [], {}, {}
    for name in M.MODEL_NAMES:
        print(f"\n{name}")
        model, hist, met = _fit(name, loaders, device, epochs, tag=name)
        counts = M.count_parameters(model)
        rows.append(dict(model=name, parameters=counts["total"],
                         trunk_parameters=counts["trunk"], head_parameters=counts["head"],
                         **{k: met[k] for k in ("accuracy", "precision", "recall",
                                                "f1", "top2_accuracy")},
                         best_epoch=hist["best_epoch"], val_accuracy=hist["best_val_acc"],
                         epoch_seconds=float(np.mean(hist["seconds"]))))
        histories[name] = hist
        if name == "CNN-LSTM":
            focal = met

    cmp_df = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    cmp_df.to_csv(C.OUT_DIR / "table_model_comparison.csv", index=False)
    results["model_comparison"] = cmp_df.drop(columns=[]).to_dict(orient="records")
    results["training_history"] = {k: {m: v[m] for m in
                                       ("epoch", "train_loss", "train_acc",
                                        "val_loss", "val_acc", "best_epoch",
                                        "best_val_acc")}
                                   for k, v in histories.items()}

    per_class = T.per_class_table(focal["y_true"], focal["y_pred"])
    pd.DataFrame(per_class).to_csv(C.OUT_DIR / "table_per_class.csv", index=False)
    cm = T.confusion(focal["y_true"], focal["y_pred"])
    results["per_class"] = per_class
    results["confusion_matrix"] = cm.tolist()
    results["top_confusions"] = _top_confusions(cm)

    P.fig_training(histories, C.FIG_DIR / "fig03_training.png")
    P.fig_model_comparison(cmp_df, C.FIG_DIR / "fig04_comparison.png")
    P.fig_confusion_perclass(cm, per_class, C.FIG_DIR / "fig05_confusion.png")

    # ---------------------------------------------------------------- optimization
    print("\n" + "=" * 74)
    print("Stage 2/2 — optimization grids")
    print("=" * 74)
    base = cmp_df[cmp_df["model"] == "CNN-LSTM"].iloc[0]
    base_hist = histories["CNN-LSTM"]

    hidden = _hidden_sweep(loaders, device, epochs, base, base_hist)
    hidden.to_csv(C.OUT_DIR / "table_hidden_sweep.csv", index=False)
    results["hidden_sweep"] = hidden.to_dict(orient="records")

    seq = _sequence_sweep(loaders, device, epochs, base)
    seq.to_csv(C.OUT_DIR / "table_sequence_ablation.csv", index=False)
    results["sequence_ablation"] = seq.to_dict(orient="records")

    aug = _augmentation_sweep(loaders, device, epochs, base, base_hist)
    aug.to_csv(C.OUT_DIR / "table_augmentation.csv", index=False)
    results["augmentation_ablation"] = aug.to_dict(orient="records")

    P.fig_optimization(hidden, seq, aug, C.FIG_DIR / "fig06_optimization.png")

    # ---------------------------------------------------------------- persist
    results["random_baseline_accuracy"] = 1.0 / len(C.CLASSES)
    results["runtime_sec"] = round(time.time() - t0, 1)
    with open(C.OUT_DIR / "results.json", "w") as f:
        json.dump(_round(results), f, indent=2)

    _print_summary(results, cmp_df, per_class, hidden, seq, aug, cm)


# ------------------------------------------------------------------ grids
def _hidden_sweep(loaders, device, epochs, base, base_hist):
    """Recurrent capacity: LSTM units per direction against held-out accuracy.
    The 256-unit row is the headline run, reused rather than retrained."""
    rows = []
    for h in C.HIDDEN_SWEEP:
        if h == C.LSTM_HIDDEN:
            rows.append(dict(hidden_units=h, parameters=int(base["parameters"]),
                             val_accuracy=float(base["val_accuracy"]),
                             test_accuracy=float(base["accuracy"]),
                             f1=float(base["f1"]), best_epoch=int(base["best_epoch"])))
            continue
        print(f"\nLSTM hidden = {h}")
        model, hist, met = _fit("CNN-LSTM", loaders, device, epochs,
                                tag=f"hidden{h}", hidden=h)
        rows.append(dict(hidden_units=h,
                         parameters=M.count_parameters(model)["total"],
                         val_accuracy=hist["best_val_acc"], test_accuracy=met["accuracy"],
                         f1=met["f1"], best_epoch=hist["best_epoch"]))
    return pd.DataFrame(rows)


def _sequence_sweep(loaders, device, epochs, base):
    """How the feature map becomes a sequence: rows or columns, one direction or
    two. Isolates whether the recurrence is reading real spatial order."""
    rows = []
    for axis, bidi in C.SEQUENCE_SWEEP:
        if (axis, bidi) == (C.LSTM_AXIS, C.LSTM_BIDIRECTIONAL):
            rows.append(dict(axis=axis, bidirectional=bidi,
                             parameters=int(base["parameters"]),
                             val_accuracy=float(base["val_accuracy"]),
                             test_accuracy=float(base["accuracy"]),
                             f1=float(base["f1"])))
            continue
        print(f"\nSequence: axis={axis}, bidirectional={bidi}")
        model, hist, met = _fit("CNN-LSTM", loaders, device, epochs,
                                tag=f"{axis}-{'bi' if bidi else 'uni'}",
                                axis=axis, bidirectional=bidi)
        rows.append(dict(axis=axis, bidirectional=bidi,
                         parameters=M.count_parameters(model)["total"],
                         val_accuracy=hist["best_val_acc"], test_accuracy=met["accuracy"],
                         f1=met["f1"]))
    return pd.DataFrame(rows)


def _augmentation_sweep(loaders, device, epochs, base, base_hist):
    """Crop-and-flip augmentation on and off, everything else held fixed."""
    rows = [dict(augmented=True, train_accuracy=base_hist["train_acc"][-1],
                 val_accuracy=float(base["val_accuracy"]),
                 test_accuracy=float(base["accuracy"]), f1=float(base["f1"]))]
    print("\nAugmentation disabled")
    plain = D.make_loaders(augment=False)
    _, hist, met = _fit("CNN-LSTM", plain, device, epochs, tag="no-aug")
    rows.append(dict(augmented=False, train_accuracy=hist["train_acc"][-1],
                     val_accuracy=hist["best_val_acc"], test_accuracy=met["accuracy"],
                     f1=met["f1"]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ helpers
def _augmentation_strip(n=9):
    """One training image plus n independent draws of the augmentation policy,
    for the data figure."""
    from torchvision import datasets, transforms
    D.ensure_extracted()
    raw = datasets.CIFAR10(C.DATA_DIR, train=True)
    img = raw.data[7]
    tf = transforms.Compose([transforms.RandomCrop(C.IMG_SIZE, padding=C.AUG_PAD),
                             transforms.RandomHorizontalFlip(p=C.AUG_HFLIP_P)])
    torch.manual_seed(C.SEED)
    pil = transforms.functional.to_pil_image(img)
    return [img] + [np.asarray(tf(pil)) for _ in range(n)]


def _top_confusions(cm, k=5):
    """The k most frequent off-diagonal (true, predicted) pairs."""
    pairs = [(int(cm[i, j]), C.CLASSES[i], C.CLASSES[j])
             for i in range(len(C.CLASSES)) for j in range(len(C.CLASSES)) if i != j]
    pairs.sort(reverse=True)
    return [dict(true=t, predicted=p, count=c) for c, t, p in pairs[:k]]


def _print_summary(r, cmp_df, per_class, hidden, seq, aug, cm):
    print("\n" + "=" * 74)
    print(f"CIFAR-10 CNN-LSTM study — seed {r['seed']} — {r['runtime_sec']/60:.1f} min "
          f"on {r['device']}")
    print("=" * 74)
    print(f"Data: {r['data']['n_train']:,} train / {r['data']['n_val']:,} val / "
          f"{r['data']['n_test']:,} test across {r['data']['n_classes']} classes")
    print("\nModel comparison (official test split):")
    print(cmp_df.to_string(index=False, columns=[
        "model", "parameters", "accuracy", "precision", "recall", "f1",
        "top2_accuracy", "best_epoch", "epoch_seconds"], float_format="%.4f"))
    pc = pd.DataFrame(per_class)
    print("\nCNN-LSTM per-class F1 (worst three, best three):")
    pc = pc.sort_values("f1")
    print(pd.concat([pc.head(3), pc.tail(3)]).to_string(index=False, float_format="%.4f"))
    print("\nMost frequent confusions:", ", ".join(
        f"{d['true']}->{d['predicted']} ({d['count']})" for d in r["top_confusions"]))
    print("\nRecurrent-capacity sweep:")
    print(hidden.to_string(index=False, float_format="%.4f"))
    print("\nSequence-formation ablation:")
    print(seq.to_string(index=False, float_format="%.4f"))
    print("\nAugmentation ablation:")
    print(aug.to_string(index=False, float_format="%.4f"))
    print("\nWrote figures/ (6 PNGs), outputs/ (7 CSVs + results.json).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.EPOCHS,
                    help="override the shared epoch budget (smoke tests only)")
    main(**vars(ap.parse_args()))

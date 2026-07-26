"""Training and evaluation: one recipe applied identically to every model, so
the comparison measures architecture rather than tuning effort.

AdamW with a two-epoch linear warmup into a cosine decay, gradient clipping for
the recurrent paths, mild label smoothing, and epoch-level checkpointing on
validation accuracy. The selected weights are always the best validation epoch,
and the official test split is scored exactly once per model, at the end.
"""
import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

import config as C


# ------------------------------------------------------------------ environment
def get_device():
    """Apple GPU when present, otherwise CPU. The recipe is identical either way."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=C.SEED):
    """Reseed every generator that touches initialization, shuffling, or dropout."""
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _scheduler(optimizer, epochs, steps_per_epoch):
    """Linear warmup then cosine decay to one hundredth of the peak rate."""
    warm = C.WARMUP_EPOCHS * steps_per_epoch
    total = epochs * steps_per_epoch

    def factor(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        p = (step - warm) / max(1, total - warm)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# ------------------------------------------------------------------ training
def train_model(model, train_loader, val_loader, device, epochs=C.EPOCHS,
                seed=C.SEED, verbose=True, tag=""):
    """Train one model and return (best_weights, history). History records the
    per-epoch training loss and accuracy, validation loss and accuracy, and
    learning rate, which is what the training-curve figure plots."""
    set_seed(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    sched = _scheduler(opt, epochs, len(train_loader))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=C.LABEL_SMOOTHING)

    history = {k: [] for k in ("epoch", "train_loss", "train_acc",
                               "val_loss", "val_acc", "lr", "seconds")}
    best = dict(acc=-1.0, epoch=-1, state=None)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        loss_sum = correct = seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            opt.step()
            sched.step()
            loss_sum += loss.item() * yb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            seen += yb.size(0)

        val = _pass(model, val_loader, device, loss_fn)
        history["epoch"].append(ep)
        history["train_loss"].append(loss_sum / seen)
        history["train_acc"].append(correct / seen)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["accuracy"])
        history["lr"].append(opt.param_groups[0]["lr"])
        history["seconds"].append(round(time.time() - t0, 1))

        if val["accuracy"] > best["acc"]:
            best.update(acc=val["accuracy"], epoch=ep,
                        state=copy.deepcopy(model.state_dict()))
        if verbose:
            print(f"  [{tag}] epoch {ep:2d}/{epochs}  "
                  f"train {loss_sum/seen:.3f}/{correct/seen:.3f}  "
                  f"val {val['loss']:.3f}/{val['accuracy']:.4f}  "
                  f"({history['seconds'][-1]:.0f}s)")

    model.load_state_dict(best["state"])
    history["best_epoch"] = best["epoch"]
    history["best_val_acc"] = best["acc"]
    return model, history


@torch.no_grad()
def _pass(model, loader, device, loss_fn=None):
    """Single evaluation pass returning loss, probabilities, and labels."""
    model.eval()
    logits, labels, loss_sum, seen = [], [], 0.0, 0
    for xb, yb in loader:
        xb = xb.to(device)
        out = model(xb).float().cpu()
        if loss_fn is not None:
            loss_sum += loss_fn(out, yb).item() * yb.size(0)
        seen += yb.size(0)
        logits.append(out)
        labels.append(yb)
    logits = torch.cat(logits)
    labels = torch.cat(labels).numpy()
    pred = logits.argmax(1).numpy()
    return dict(loss=loss_sum / max(1, seen), accuracy=accuracy_score(labels, pred),
                proba=torch.softmax(logits, 1).numpy(), y_true=labels, y_pred=pred)


# ------------------------------------------------------------------ metrics
def evaluate(model, loader, device):
    """Full metric bundle on a held-out partition. Precision, recall, and F1 are
    macro-averaged because CIFAR-10 is exactly balanced, so each class should
    weigh the same."""
    r = _pass(model, loader, device)
    y, p, proba = r["y_true"], r["y_pred"], r["proba"]
    top2 = np.mean([yi in row for yi, row in zip(y, np.argsort(-proba, 1)[:, :2])])
    return dict(
        accuracy=float(accuracy_score(y, p)),
        precision=float(precision_score(y, p, average="macro", zero_division=0)),
        recall=float(recall_score(y, p, average="macro", zero_division=0)),
        f1=float(f1_score(y, p, average="macro", zero_division=0)),
        top2_accuracy=float(top2),
        y_true=y, y_pred=p, proba=proba)


def per_class_table(y_true, y_pred):
    """Precision, recall, F1, and support for each of the ten classes."""
    rep = classification_report(y_true, y_pred, target_names=C.CLASSES,
                                output_dict=True, zero_division=0)
    return [dict(class_name=n, precision=rep[n]["precision"], recall=rep[n]["recall"],
                 f1=rep[n]["f1-score"], support=int(rep[n]["support"]))
            for n in C.CLASSES]


def confusion(y_true, y_pred):
    """Ten-by-ten confusion matrix, rows true and columns predicted."""
    return confusion_matrix(y_true, y_pred)

"""CIFAR-10: load and verify the real data, normalize it, augment the training
partition, and carve a stratified validation set out of the official training
split so the official test split is never touched during model selection.
"""
import hashlib
import tarfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import config as C

ARCHIVE = C.DATA_DIR / "cifar-10-python.tar.gz"
ARCHIVE_MD5 = "c58f30108f718f92721af3b95e74349a"      # published by Krizhevsky (2009)


# ------------------------------------------------------------------ raw data
def ensure_extracted():
    """Verify the cached archive against its published MD5 and extract it once."""
    if C.CIFAR_DIR.exists():
        return
    if not ARCHIVE.exists():
        raise FileNotFoundError(
            f"{ARCHIVE} is missing. See data/SOURCE.txt for the retrieval command.")
    h = hashlib.md5()
    with open(ARCHIVE, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != ARCHIVE_MD5:
        raise ValueError(f"CIFAR-10 archive checksum mismatch: {h.hexdigest()}")
    with tarfile.open(ARCHIVE) as t:
        t.extractall(C.DATA_DIR)


def _transforms(augment):
    """Evaluation transform is normalization only; training adds the crop/flip
    policy. Both share the same channel statistics."""
    norm = [transforms.ToTensor(), transforms.Normalize(C.CHANNEL_MEAN, C.CHANNEL_STD)]
    if not augment:
        return transforms.Compose(norm)
    return transforms.Compose([
        transforms.RandomCrop(C.IMG_SIZE, padding=C.AUG_PAD),
        transforms.RandomHorizontalFlip(p=C.AUG_HFLIP_P),
        *norm,
    ])


def load_datasets(augment=True):
    """Return (train, val, test) datasets. The 50,000 official training images
    are split 45,000/5,000 by a stratified draw under the master seed; the
    validation and test partitions carry the evaluation transform only, so no
    augmentation ever contaminates a reported metric."""
    ensure_extracted()
    train_aug = datasets.CIFAR10(C.DATA_DIR, train=True, transform=_transforms(augment))
    train_ev = datasets.CIFAR10(C.DATA_DIR, train=True, transform=_transforms(False))
    test = datasets.CIFAR10(C.DATA_DIR, train=False, transform=_transforms(False))

    y = np.asarray(train_aug.targets)
    _verify(y, np.asarray(test.targets), train_aug.data)

    tr_idx, va_idx = _stratified_split(y, C.N_VAL, C.SEED)
    return Subset(train_aug, tr_idx), Subset(train_ev, va_idx), test


def _stratified_split(y, n_val, seed):
    """Hold out n_val/10 images per class, chosen by a seeded permutation."""
    rng = np.random.default_rng(seed)
    per_class = n_val // len(C.CLASSES)
    val = []
    for k in range(len(C.CLASSES)):
        idx = np.flatnonzero(y == k)
        val.append(rng.permutation(idx)[:per_class])
    val = np.sort(np.concatenate(val))
    train = np.setdiff1d(np.arange(len(y)), val)
    return train.tolist(), val.tolist()


def _verify(y_train, y_test, images):
    """Integrity checks that fail loudly rather than train on the wrong file."""
    assert images.shape == (C.N_TRAIN_FULL, C.IMG_SIZE, C.IMG_SIZE, 3), images.shape
    assert len(y_train) == C.N_TRAIN_FULL and len(y_test) == C.N_TEST
    assert images.dtype == np.uint8 and images.min() == 0 and images.max() == 255
    counts = np.bincount(y_train, minlength=len(C.CLASSES))
    assert (counts == C.N_TRAIN_FULL // len(C.CLASSES)).all(), counts
    assert (np.bincount(y_test, minlength=len(C.CLASSES))
            == C.N_TEST // len(C.CLASSES)).all()


# ------------------------------------------------------------------ loaders
def make_loaders(augment=True, batch_size=None, seed=None):
    """DataLoaders whose shuffling and augmentation draws are fixed by the seed."""
    bs = batch_size or C.BATCH_SIZE
    g = torch.Generator().manual_seed(seed if seed is not None else C.SEED)
    train, val, test = load_datasets(augment=augment)
    common = dict(num_workers=C.NUM_WORKERS, pin_memory=False,
                  persistent_workers=C.NUM_WORKERS > 0)
    return (
        DataLoader(train, batch_size=bs, shuffle=True, generator=g,
                   drop_last=True, worker_init_fn=_worker_seed, **common),
        DataLoader(val, batch_size=512, shuffle=False, **common),
        DataLoader(test, batch_size=512, shuffle=False, **common),
    )


def _worker_seed(worker_id):
    """Give each dataloader worker a deterministic, distinct RNG stream."""
    s = C.SEED + worker_id
    np.random.seed(s % (2 ** 32))
    torch.manual_seed(s)


# ------------------------------------------------------------------ description
def describe_data():
    """Per-class image counts across the three partitions, for the data table."""
    ensure_extracted()
    train = datasets.CIFAR10(C.DATA_DIR, train=True)
    test = datasets.CIFAR10(C.DATA_DIR, train=False)
    y = np.asarray(train.targets)
    _, va_idx = _stratified_split(y, C.N_VAL, C.SEED)
    val_mask = np.zeros(len(y), bool)
    val_mask[va_idx] = True
    rows = []
    for k, name in enumerate(C.CLASSES):
        in_k = y == k
        rows.append(dict(class_id=k, class_name=name,
                         train=int((in_k & ~val_mask).sum()),
                         validation=int((in_k & val_mask).sum()),
                         test=int((np.asarray(test.targets) == k).sum())))
    return pd.DataFrame(rows)


def channel_statistics():
    """Per-channel mean and standard deviation of the raw training images,
    recomputed so the constants in config.py are verified rather than trusted."""
    ensure_extracted()
    x = datasets.CIFAR10(C.DATA_DIR, train=True).data.astype(np.float64) / 255.0
    mean, std = x.mean(axis=(0, 1, 2)), x.std(axis=(0, 1, 2))
    assert np.allclose(mean, C.CHANNEL_MEAN, atol=5e-4), mean
    assert np.allclose(std, C.CHANNEL_STD, atol=5e-4), std
    return mean, std


def sample_images(n_per_class=6):
    """One strip of raw images per class for the dataset figure."""
    ensure_extracted()
    train = datasets.CIFAR10(C.DATA_DIR, train=True)
    y = np.asarray(train.targets)
    rng = np.random.default_rng(C.SEED)
    return {name: train.data[rng.choice(np.flatnonzero(y == k), n_per_class, replace=False)]
            for k, name in enumerate(C.CLASSES)}

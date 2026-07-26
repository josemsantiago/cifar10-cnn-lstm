"""Central configuration for the Week 4 CNN-LSTM image-classification study.

Every tunable constant lives here so the whole pipeline is auditable from one
place: the master seed, the data location, the normalization and augmentation
policy, the four model definitions, the shared training recipe, and the three
optimization grids.
"""
from pathlib import Path

# ----------------------------------------------------------------- reproducibility
SEED = 20260726                      # master seed (date of the run)

# ----------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "outputs"
CIFAR_DIR = DATA_DIR / "cifar-10-batches-py"     # extracted by data.load_datasets

for _d in (FIG_DIR, OUT_DIR):
    _d.mkdir(exist_ok=True)

# ----------------------------------------------------------------- data schema
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
IMG_SIZE = 32                        # CIFAR-10 images are 32 x 32 x 3
N_TRAIN_FULL = 50_000                # official training split
N_TEST = 10_000                      # official test split, held out untouched
N_VAL = 5_000                        # carved from the training split, stratified

# Channel statistics of the CIFAR-10 training split (computed in data.py and
# asserted against these values, so normalization is never silently wrong).
CHANNEL_MEAN = (0.4914, 0.4822, 0.4465)
CHANNEL_STD = (0.2470, 0.2435, 0.2616)

# Augmentation policy applied to the training partition only.
AUG_PAD = 4                          # random 32x32 crop from a 40x40 zero-padded image
AUG_HFLIP_P = 0.5                    # horizontal flip probability

# ----------------------------------------------------------------- architecture
# Convolutional trunk shared by every CNN-based model, so any difference between
# them is attributable to the classification head and not to the features.
CONV_CHANNELS = (64, 128, 256)       # three double-conv blocks
CONV_POOL = (True, True, False)      # pool after blocks 1 and 2 -> 8 x 8 feature map
CONV_DROPOUT = (0.2, 0.3, 0.4)       # spatial dropout, increasing with depth
SEQ_PROJ_CHANNELS = 128              # 1x1 convolution that compresses channels before sequencing

LSTM_HIDDEN = 256                    # units per direction
LSTM_BIDIRECTIONAL = True
LSTM_AXIS = "row"                    # sequence the feature map along rows ("row") or columns ("col")
HEAD_DROPOUT = 0.5

PIXEL_LSTM_HIDDEN = 256              # CNN-free baseline: raw pixel rows into a stacked BiLSTM
PIXEL_LSTM_LAYERS = 2
PIXEL_LSTM_DROPOUT = 0.3

DENSE_HEAD_UNITS = 320               # sized so the dense control head matches the LSTM head's
                                     # parameter count to within 0.2% (2.626M vs 2.631M)

# ----------------------------------------------------------------- training recipe
# One recipe for every model, so the comparison isolates architecture.
EPOCHS = 20
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.05
GRAD_CLIP = 5.0                      # LSTM gradients need a ceiling
WARMUP_EPOCHS = 2                    # linear warmup, then cosine decay to LR/100
NUM_WORKERS = 4

# ----------------------------------------------------------------- optimization grids
# Each grid re-trains the focal model under the same recipe and epoch budget, so
# every row in the resulting tables is directly comparable to the headline run.
HIDDEN_SWEEP = [64, 128, 256, 512]                    # LSTM units per direction
SEQUENCE_SWEEP = [                                    # how the feature map becomes a sequence
    ("row", True),                                    # (axis, bidirectional)
    ("row", False),
    ("col", True),
]
AUGMENTATION_SWEEP = [True, False]                    # crop + flip on or off

"""The models: a CNN-LSTM hybrid and three controls that isolate what the
recurrent head actually contributes.

Every CNN-based model shares one convolutional trunk, so any difference between
them is attributable to the head and not to the features. The fourth model drops
the trunk entirely and reads raw pixel rows into a stacked LSTM, which fixes the
other end of the scale: what recurrence alone can do on images.
"""
import torch
import torch.nn as nn

import config as C


# ------------------------------------------------------------------ trunk
def _conv_block(c_in, c_out, pool, p_drop):
    """Two 3x3 convolutions with batch normalization, then optional pooling and
    spatial dropout. Batch norm stabilizes the trunk that feeds the LSTM."""
    layers = [
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    layers.append(nn.Dropout2d(p_drop))
    return nn.Sequential(*layers)


class ConvTrunk(nn.Module):
    """Three double-convolution blocks (64, 128, 256 channels) that pool twice,
    followed by a 1x1 convolution compressing the channel axis. Turns a
    3 x 32 x 32 image into a 128 x 8 x 8 map of local feature detectors."""

    def __init__(self, proj=C.SEQ_PROJ_CHANNELS):
        super().__init__()
        blocks, c_in = [], 3
        for c_out, pool, p in zip(C.CONV_CHANNELS, C.CONV_POOL, C.CONV_DROPOUT):
            blocks.append(_conv_block(c_in, c_out, pool, p))
            c_in = c_out
        self.blocks = nn.Sequential(*blocks)
        self.project = nn.Sequential(
            nn.Conv2d(c_in, proj, 1, bias=False),
            nn.BatchNorm2d(proj), nn.ReLU(inplace=True))
        self.out_channels = proj
        self.out_size = C.IMG_SIZE // (2 ** sum(C.CONV_POOL))

    def forward(self, x):
        return self.project(self.blocks(x))


def _to_sequence(z, axis):
    """Read a (B, C, H, W) feature map as a sequence of H rows (or W columns),
    each step carrying every channel across the orthogonal axis."""
    b, c, h, w = z.shape
    if axis == "row":
        return z.permute(0, 2, 3, 1).reshape(b, h, w * c)
    if axis == "col":
        return z.permute(0, 3, 2, 1).reshape(b, w, h * c)
    raise ValueError(f"unknown sequence axis: {axis}")


def _last_state(h_n, bidirectional):
    """Final hidden state of the top layer; both directions concatenated when
    the layer is bidirectional."""
    return torch.cat([h_n[-2], h_n[-1]], dim=1) if bidirectional else h_n[-1]


# ------------------------------------------------------------------ models
class CnnLstm(nn.Module):
    """The focal model. Convolutions extract local spatial features, the map is
    read as a top-to-bottom sequence of rows, and a bidirectional LSTM
    integrates them into one vector that the classifier scores."""

    def __init__(self, hidden=C.LSTM_HIDDEN, bidirectional=C.LSTM_BIDIRECTIONAL,
                 axis=C.LSTM_AXIS):
        super().__init__()
        self.trunk = ConvTrunk()
        self.axis = axis
        self.bidirectional = bidirectional
        step_dim = self.trunk.out_channels * self.trunk.out_size
        self.lstm = nn.LSTM(step_dim, hidden, batch_first=True,
                            bidirectional=bidirectional)
        self.head = nn.Sequential(
            nn.Dropout(C.HEAD_DROPOUT),
            nn.Linear(hidden * (2 if bidirectional else 1), len(C.CLASSES)))

    def forward(self, x):
        seq = _to_sequence(self.trunk(x), self.axis)
        _, (h_n, _) = self.lstm(seq)
        return self.head(_last_state(h_n, self.bidirectional))


class CnnGap(nn.Module):
    """Control 1: the same trunk with the conventional convolutional head,
    global average pooling straight into a linear classifier."""

    def __init__(self):
        super().__init__()
        self.trunk = ConvTrunk()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(C.HEAD_DROPOUT),
            nn.Linear(self.trunk.out_channels, len(C.CLASSES)))

    def forward(self, x):
        return self.head(self.trunk(x))


class CnnDense(nn.Module):
    """Control 2: the same trunk with a fully connected head sized to match the
    LSTM head's parameter count, so the comparison tests recurrence rather than
    capacity."""

    def __init__(self, units=C.DENSE_HEAD_UNITS):
        super().__init__()
        self.trunk = ConvTrunk()
        flat = self.trunk.out_channels * self.trunk.out_size ** 2
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, units), nn.BatchNorm1d(units), nn.ReLU(inplace=True),
            nn.Dropout(C.HEAD_DROPOUT),
            nn.Linear(units, len(C.CLASSES)))

    def forward(self, x):
        return self.head(self.trunk(x))


class PixelLstm(nn.Module):
    """Control 3: no convolutions at all. The image is a 32-step sequence of
    rows, each step the 96 raw channel-interleaved pixel values of that row,
    fed to a stacked bidirectional LSTM."""

    def __init__(self, hidden=C.PIXEL_LSTM_HIDDEN, layers=C.PIXEL_LSTM_LAYERS):
        super().__init__()
        self.lstm = nn.LSTM(C.IMG_SIZE * 3, hidden, num_layers=layers,
                            batch_first=True, bidirectional=True,
                            dropout=C.PIXEL_LSTM_DROPOUT)
        self.head = nn.Sequential(
            nn.Dropout(C.PIXEL_LSTM_DROPOUT),
            nn.Linear(hidden * 2, len(C.CLASSES)))

    def forward(self, x):
        _, (h_n, _) = self.lstm(_to_sequence(x, "row"))
        return self.head(_last_state(h_n, True))


# ------------------------------------------------------------------ registry
def build(name, **kwargs):
    """Construct a model by the name used throughout the report and outputs."""
    return {
        "CNN-LSTM": CnnLstm,
        "CNN + GAP head": CnnGap,
        "CNN + dense head": CnnDense,
        "Pixel-LSTM (no CNN)": PixelLstm,
    }[name](**kwargs)


MODEL_NAMES = ["CNN-LSTM", "CNN + GAP head", "CNN + dense head", "Pixel-LSTM (no CNN)"]


# ------------------------------------------------------------------ inspection
def count_parameters(model):
    """Trainable parameters, total and split between trunk and head."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trunk = sum(p.numel() for n, p in model.named_parameters()
                if p.requires_grad and n.startswith("trunk"))
    return dict(total=total, trunk=trunk, head=total - trunk)


def layer_summary(model, input_shape=(1, 3, C.IMG_SIZE, C.IMG_SIZE)):
    """Per-stage output shape and parameter count, captured by running one
    dummy batch through the top-level children. Serves as the model summary."""
    rows, x = [], torch.zeros(input_shape)
    model = model.eval()
    with torch.no_grad():
        for name, child in model.named_children():
            if isinstance(child, nn.LSTM):
                x = _to_sequence(x, getattr(model, "axis", "row"))
                out, (h_n, _) = child(x)
                x = _last_state(h_n, child.bidirectional)
                shape = tuple(out.shape[1:])
            else:
                x = child(x)
                shape = tuple(x.shape[1:])
            rows.append(dict(stage=name,
                             output_shape=" x ".join(str(d) for d in shape),
                             parameters=sum(p.numel() for p in child.parameters())))
    return rows

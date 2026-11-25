"""Shared deterministic CIFAR-10 input planes and resumable prediction caches."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor


MEAN = (.4914, .4822, .4465)
STD = (.2023, .1994, .2010)
PALETTE = ["#ffca00", "#ff8000", "#2ca02c", "#00edff", "#a65628",
           "#ff00ff", "#ff0030", "#6c00ff", "#004cff", "#69ff00"]
CLASSES = ("Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog",
           "Horse", "Ship", "Truck")
PUBLICATION_ANCHORS = (30, 72, 42)
SHUFFLE_SEEDS = (8, 2, 1)


def dataset(root, train):
    return CIFAR10(root, train=train, download=False,
                   transform=Compose([ToTensor(), Normalize(MEAN, STD)]))


@contextmanager
def atomic(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        try:
            yield f
            f.flush()
            os.fsync(f.fileno())
            os.replace(f.name, path)
        finally:
            Path(f.name).unlink(missing_ok=True)


def save_json(path, content):
    with atomic(path) as f:
        f.write((json.dumps(content, indent=2, allow_nan=False) + "\n").encode())


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def load_weights(model, path, device):
    weights = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    return model.to(device).eval()


def make_plane(images, grid, margin, device):
    # Same Gram-Schmidt basis and margins as dbViz/data.py:get_plane, plane_dataset.
    if grid < 2:
        raise ValueError("A plane needs at least two grid points per side.")
    first, second, third = images
    a, b = second-first, third-first
    anorm = torch.dot(a.flatten(), a.flatten()).sqrt()
    if float(anorm) == 0:
        raise ValueError("Duplicate anchors.")
    u = a / anorm
    x3 = torch.dot(u.flatten(), b.flatten())
    perpendicular = b-x3*u
    bnorm = torch.dot(perpendicular.flatten(), perpendicular.flatten()).sqrt()
    if float(bnorm) == 0:
        raise ValueError("Collinear anchors.")
    v = perpendicular/bnorm
    y3 = torch.dot(v.flatten(), b.flatten())
    coords = np.array([[0, 0], [anorm.item(), 0], [x3.item(), y3.item()]], dtype=np.float32)
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    dx, dy = (xmax-xmin)*margin, (ymax-ymin)*margin
    extent = [float(xmin-dx), float(xmax+dx), float(ymin-dy), float(ymax+dy)]
    xs = torch.linspace(*extent[:2], grid).to(device)
    ys = torch.linspace(*extent[2:], grid).to(device)
    return (first.to(device), u.to(device), v.to(device), xs, ys), coords, extent


@torch.inference_mode()
def predict_plane(model, plane, batch_size, input_size=32, return_confidence=False):
    origin, u, v, xs, ys = plane
    if batch_size < 1:
        raise ValueError("Use a positive inference batch size.")
    size = len(xs)
    labels = np.empty(size * size, np.uint8)
    confidence = np.empty(size * size, np.float32) if return_confidence else None
    for start in range(0, len(labels), batch_size):
        stop = min(len(labels), start + batch_size)
        i = torch.arange(start, stop, device=xs.device)
        inputs = origin + xs[i//size, None, None, None]*u + ys[i%size, None, None, None]*v
        if input_size != origin.shape[-1]:
            inputs = F.interpolate(inputs, (input_size, input_size), mode="bilinear",
                                   align_corners=False, antialias=True)
        result = model(inputs.contiguous())
        if result.shape != (stop-start, 10) or not torch.isfinite(result).all():
            raise ValueError("The model must return ten finite CIFAR-10 logits per image.")
        labels[start:stop] = result.argmax(1).to(torch.uint8).cpu().numpy()
        if return_confidence:
            confidence[start:stop] = torch.softmax(result.float(), dim=1).amax(dim=1).cpu().numpy()
    return (labels, confidence) if return_confidence else labels


def plot_anchors(train, column_seed=None):
    if column_seed is None:
        ids = list(PUBLICATION_ANCHORS)
    else:
        # dbViz/main.py random_shuffle replaces --imgs with RandomState.choice.
        ids = np.random.RandomState(column_seed).choice(len(train), size=3).tolist()
    images = [train[i][0] for i in ids]
    if column_seed is not None:
        generator = torch.Generator().manual_seed(column_seed)
        for j, image in enumerate(images):
            order = torch.randperm(image.shape[-2] * image.shape[-1], generator=generator)
            images[j] = image.flatten(1)[:, order].reshape_as(image)
    return images, ids, [train.targets[i] for i in ids]

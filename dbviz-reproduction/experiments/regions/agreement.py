"""Compare two CIFAR-10 ResNet18 checkpoints on the dbViz input-space planes."""
import argparse
import fcntl
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from ..arch.source import resnet
from . import evaluation
from .evaluation import CLASSES as CLASS_NAMES, PUBLICATION_ANCHORS as FIGURE_IDS
from .evaluation import atomic as atomic_file, dataset, digest, load_weights, make_plane, save_json


def triplet_ids(labels, seed):
    rng = random.Random(seed)
    indices, classes = [], set()
    while len(indices) < 3:
        index = rng.randint(0, len(labels) - 1)
        label = labels[index]
        if label not in classes:
            indices.append(index)
            classes.add(label)
    return indices


def build_plane(split, indices, grid, margin, device):
    return make_plane([split[i][0] for i in indices], grid, margin, device)


def predict_pair(models, inputs):
    labels = [model(inputs).argmax(dim=1) for model in models]
    return torch.stack(labels).to(dtype=torch.uint8).cpu().numpy()


def plane_batches(plane, start, batch_size, models, device):
    origin, axis_x, axis_y, xs, ys = plane
    size = len(xs)
    for offset in range(start, size * size, batch_size):
        end = min(offset + batch_size, size * size)
        flat = torch.arange(offset, end, device=device)
        x = xs[flat // size].view(-1, 1, 1, 1)
        y = ys[flat % size].view(-1, 1, 1, 1)
        inputs = origin + x * axis_x + y * axis_y
        yield end, predict_pair(models, inputs)


def score_plane(plane, batch_size, models, device):
    matches = 0
    for _, (first, second) in plane_batches(plane, 0, batch_size, models, device):
        matches += int(np.count_nonzero(first == second))
    return matches


def figure_labels(plane, batch_size, models, device, cache):
    size = len(plane[-1]) ** 2
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            predictions = saved["labels"]
            start = int(saved["offset"].item())
        if predictions.shape != (2, size) or not 0 <= start <= size:
            raise ValueError("Inconsistent figure cache")
    else:
        predictions = np.zeros((2, size), dtype=np.uint8)
        start = 0
    last_save = time.monotonic()
    for end, batch in plane_batches(plane, start, batch_size, models, device):
        predictions[:, start:end] = batch
        start = end
        if end == size or time.monotonic() - last_save > 60:
            with atomic_file(cache) as handle:
                np.savez_compressed(handle, labels=predictions, offset=end)
            print(f"Figure plane: {end}/{size} positions", flush=True)
            last_save = time.monotonic()
    return predictions


def draw(labels, coords, extent, anchor_labels, seeds, out, title="ResNet-18 decision regions"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.colors import ListedColormap

    palette = plt.get_cmap("gist_rainbow")
    colors = [palette(i) for i in (45, 30, 170, 150, 65, 245, 0, 220, 180, 90)]
    colors[2], colors[4] = "#2ca02c", "#a65628"
    cmap = ListedColormap(colors)
    figure, axes = plt.subplots(2, 1, figsize=(6.2, 8.7))
    figure.subplots_adjust(left=0.12, right=0.88, bottom=0.15, top=0.86, hspace=0.22)
    figure.suptitle(title, y=0.98, fontsize=18, fontweight="semibold")
    figure.legend([Patch(facecolor=c) for c in colors], CLASS_NAMES, loc="upper center",
                  bbox_to_anchor=(0.5, 0.947), frameon=False, ncol=5, fontsize=9,
                  columnspacing=1.1, handlelength=1.25, handletextpad=0.4)
    markers = ("o", "^", "X")
    size = int(round(np.sqrt(labels.shape[1])))
    # imshow expects bounds at pixel edges, while the coordinate axes refer to centres.
    dx = (extent[1] - extent[0]) / (2 * (size - 1))
    dy = (extent[3] - extent[2]) / (2 * (size - 1))
    image_edges = extent + np.array([-dx, dx, -dy, dy])
    for ax, seed, result in zip(axes, seeds, labels):
        ax.imshow(result.reshape(size, size).T, origin="lower", interpolation="nearest",
                  extent=image_edges, aspect="auto", cmap=cmap, vmin=-0.5, vmax=9.5)
        ax.set_box_aspect(0.75)  # Figure 3 uses 4:3 panels.
        for (x, y), marker in zip(coords, markers):
            ax.plot(x, y, marker=marker, color="#131518", markersize=7, linestyle="none")
        ax.set_title(f"Training seed {seed}", loc="left", fontsize=11, pad=5)
        ax.set_axis_off()
    handles = [Line2D([], [], marker=m, color="#131518", linestyle="none", markersize=6)
               for m in markers]
    anchor_names = [f"{CLASS_NAMES[label]}  #{idx}" for label, idx in zip(anchor_labels, FIGURE_IDS)]
    figure.legend(handles, anchor_names, loc="lower center", bbox_to_anchor=(0.5, 0.052),
                  frameon=False, ncol=3, fontsize=9, title="CIFAR-10 training images")
    with atomic_file(out) as handle:
        figure.savefig(handle, format="png", dpi=180, facecolor="white")
    plt.close(figure)


def real_image_scores(dataset, models, batch_size, device):
    correct = np.zeros(2, dtype=np.int64)
    matches = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for inputs, target in loader:
        prediction = predict_pair(models, inputs.to(device))
        correct += (prediction == target.numpy()).sum(axis=1)
        matches += int(np.count_nonzero(prediction[0] == prediction[1]))
    return {"points": len(dataset), "correct_each": correct.tolist(),
            "accuracy_each": (correct / len(dataset)).tolist(),
            "matches": matches, "agreement": matches / len(dataset)}


def run(args):
    out, cache, device = args.out, args.cache, args.device
    test = dataset(args.data, train=False)
    train = dataset(args.data, train=True)
    model_file, geometry_file = Path(resnet.__file__), Path(evaluation.__file__)
    settings = {"schema": 1, "model_sha256": [digest(p) for p in args.checkpoints],
                "reference_model_sha256": digest(model_file), "geometry_sha256": digest(geometry_file),
                "figure_grid": args.figure_grid, "score_grid": args.score_grid,
                "figure_margin": 0.5, "score_margin": 0.1, "data_seed": args.data_seed,
                "model_seeds": args.seeds, "device": device, "torch_version": str(torch.__version__),
                "training_anchors": list(FIGURE_IDS), "anchor_augmentation": False}
    if settings["model_sha256"][0] == settings["model_sha256"][1]:
        raise ValueError("Checkpoints contain identical bytes; provide the two independently trained runs")
    state_path = cache / "progress.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if settings != state["settings"]:
            raise ValueError("Cache was created with different checkpoints or evaluation settings; use another --cache.")
    else:
        state = {"settings": settings, "planes": []}
        save_json(state_path, state)
    models = [load_weights(resnet.ResNet18(), path, device) for path in args.checkpoints]

    if "test" not in state:
        state["test"] = real_image_scores(test, models, args.batch_size, device)
        save_json(state_path, state)
        print("Real test images:", state["test"], flush=True)

    plane, coords, extent = build_plane(train, FIGURE_IDS, args.figure_grid, 0.5, device)
    labels = figure_labels(plane, args.batch_size, models, device, cache / "figure.npz")
    anchor_labels = [train.targets[i] for i in FIGURE_IDS]
    draw(labels, coords, extent, anchor_labels, args.seeds, out / "figure3_resnet18.png")
    fixed_matches = int(np.count_nonzero(labels[0] == labels[1]))
    fixed_total = labels.shape[1]
    state["figure_plane"] = {"indices": list(FIGURE_IDS), "true_labels": anchor_labels, "matches": fixed_matches,
                             "points": fixed_total, "agreement": fixed_matches / fixed_total}
    save_json(state_path, state)

    start = len(state["planes"])
    stop = args.planes if args.max_new_planes is None else min(args.planes, start + args.max_new_planes)
    for i in range(start, stop):
        ids = triplet_ids(test.targets, i + args.data_seed)
        plane, _, _ = build_plane(test, ids, args.score_grid, 0.1, device)
        matches = score_plane(plane, args.batch_size, models, device)
        state["planes"].append({"plane": i + 1, "indices": ids, "matches": matches,
                                "points": args.score_grid ** 2, "agreement": matches / args.score_grid ** 2})
        save_json(state_path, state)
        if (i + 1) % 10 == 0 or i + 1 == stop:
            print(f"Scored {i + 1}/{args.planes} planes", flush=True)
    planes = state["planes"][:args.planes]
    matches = sum(row["matches"] for row in planes)
    points = len(planes) * args.score_grid ** 2
    report = {**state, "planes": planes,
              "score": {"requested_planes": args.planes, "evaluated_planes": len(planes),
                        "complete": len(planes) == args.planes, "matches": matches, "points": points,
                        "agreement": matches / points if points else None}}
    save_json(out / "agreement.json", report)
    print(json.dumps(report["score"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Torchvision CIFAR10 root")
    parser.add_argument("--checkpoints", required=True, nargs=2, type=Path, metavar=("FIRST", "SECOND"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--figure-grid", type=int, default=500)
    parser.add_argument("--score-grid", type=int, default=50)
    parser.add_argument("--planes", type=int, default=500)
    parser.add_argument("--data-seed", type=int, default=1)
    parser.add_argument("--seeds", nargs=2, type=int, default=[0, 1])
    parser.add_argument("--max-new-planes", type=int, help="Maximum additional full planes this invocation")
    args = parser.parse_args()
    if args.batch_size < 1 or min(args.figure_grid, args.score_grid) < 2 or args.planes < 1:
        parser.error("Positive batch size, at least one plane, and grid sides of at least two are required.")
    if args.max_new_planes is not None and args.max_new_planes < 0:
        parser.error("--max-new-planes must be nonnegative")
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.cache / "lock").open("a") as lock, torch.inference_mode():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args)


if __name__ == "__main__":
    main()

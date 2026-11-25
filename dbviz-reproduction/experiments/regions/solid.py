"""Redraw dbViz figures 1–3 from the saved CIFAR-10 class grids on CPU."""
import argparse
from contextlib import nullcontext
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import numpy as np


NAMES = {"fc": "Fully connected", "vit": "ViT", "wrn10": "WideResNet-28×10",
         "dense": "DenseNet-121", "resnet": "ResNet-18", "mixer": "MLP-Mixer", "vgg": "VGG-19"}
TITLES = {1: "Decision regions through natural images",
          2: "Decision regions through pixel-shuffled images",
          3: "Decision regions across model architectures"}
SYMBOLS = ("●", "▲", "×")


def read_result(archive, measurements, number, extension):
    name = f"figure{number}_g500.{extension}"
    if archive:
        return archive.read(f"figures/{name}")
    return (measurements / name).read_bytes()


def draw(number, meta, grids, train, out, ev):
    info = meta["input"]
    models, seeds = meta["architectures"], meta["seeds"]
    shuffled = number == 2
    anchors = [ev.plot_anchors(train, s) for s in (info["column_sampling_seeds"] or [None])]
    assert [a[1] for a in anchors] == info["image_indices"]
    assert [a[2] for a in anchors] == info["anchor_classes"]
    assert list(ev.CLASSES) == meta["class_order"] and meta["grid"] == 500 and meta["figure"] == number
    markers = tuple(zip(("o", "^", "X"), SYMBOLS))
    plane_geometry = []
    for images, _, _ in anchors:
        _, coords, bounds = ev.make_plane(images, meta["grid"], info["margin_each_side"], "cpu")
        xmin, xmax, ymin, ymax = bounds
        dx, dy = (xmax - xmin) / 998, (ymax - ymin) / 998
        plane_geometry.append((coords, (xmin - dx, xmax + dx, ymin - dy, ymax + dy)))

    if shuffled:
        row_keys = [(m, s) for m in models for s in seeds]
        rows = [f"{NAMES[m]}\nseed {s}" for m, s in row_keys]
        titles = ["   ".join(f"{sym} {ev.CLASSES[i]}" for sym, i in zip(SYMBOLS, classes))
                  for _, _, classes in anchors]
    else:
        rows = [f"seed {s}" for s in seeds]
        titles = [NAMES[m] for m in models]
    nr, nc = len(rows), len(titles)
    fig_w = .95 + 2.47 * nc
    fig_h = 1.34 + 1.92 * nr + (.84 if shuffled else 0)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    grid = fig.add_gridspec(nr + int(shuffled), nc,
                            height_ratios=[1.] * nr + ([.37] if shuffled else []),
                            left=(.86 if shuffled else .64) / fig_w, right=1 - .16 / fig_w,
                            top=1 - .89 / fig_h, bottom=(.60 if nc < 5 else .48) / fig_h,
                            hspace=.095, wspace=.06)
    palette = ListedColormap(ev.PALETTE)
    norm = BoundaryNorm(np.arange(-.5, 10.5), palette.N)
    map_keys = {key for key in grids.files if not key.endswith("_confidence")}
    assert map_keys == {f"row{r+1}_col{c+1}" for r in range(nr) for c in range(nc)}
    for r, row in enumerate(rows):
        for c, title in enumerate(titles):
            m, s = row_keys[r] if shuffled else (models[c], seeds[r])
            plane = c if shuffled else 0
            panel = meta["panel_counts"][f"{m}/seed{s}/plane{plane + 1}"]
            labels = grids[f"row{r+1}_col{c+1}"]
            assert labels.shape == (500, 500) and labels.dtype == np.uint8
            assert np.bincount(labels.ravel(), minlength=10).tolist() == panel["class_counts"]
            assert panel["image_indices"] == anchors[plane][1]
            coords, edges = plane_geometry[plane]
            ax = fig.add_subplot(grid[r, c])
            ax.imshow(labels, cmap=palette, norm=norm, origin="lower", extent=edges,
                      interpolation="nearest", aspect="auto")
            for (x, y), (marker, _) in zip(coords, markers):
                ax.scatter(x, y, marker=marker, s=18, color="#202124", linewidths=.25)
            if r == 0:
                ax.set_title(title, fontsize=8.5 if shuffled else 10, pad=4.5)
            if c == 0:
                ax.set_ylabel(row, fontsize=9, labelpad=5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_visible(False)
    if shuffled:
        for c, (images, _, _) in enumerate(anchors):
            tiles = [np.clip(im.permute(1, 2, 0).numpy() * ev.STD + ev.MEAN, 0, 1)
                     for im in images]
            gap = np.ones((32, 2, 3))
            strip = np.concatenate([tiles[0], gap, tiles[1], gap, tiles[2]], axis=1)
            ax = fig.add_subplot(grid[nr, c])
            ax.imshow(strip, interpolation="nearest")
            ax.set_axis_off()
    fig.suptitle(TITLES[number], y=1 - .10 / fig_h, fontsize=16, weight="semibold")
    caption = "Color = predicted class"
    if not shuffled:
        anchor_key = "   ".join(f"{sym} {ev.CLASSES[i]}" for sym, i in zip(SYMBOLS, anchors[0][2]))
        caption += "      |      Anchors:  " + anchor_key
    else:
        caption += "      |      Symbols identify the three shuffled images in each column"
    fig.text(.5, 1 - .44 / fig_h, caption, ha="center", va="top", fontsize=8.5, color="#364553")
    fig.legend([Patch(facecolor=color) for color in ev.PALETTE], ev.CLASSES, frameon=False,
               ncol=5 if nc < 5 else 10, fontsize=8, loc="lower center", bbox_to_anchor=(.52, .006),
               handlelength=1.1, columnspacing=1.25)
    fig.savefig(out / f"figure{number}_solid.png", dpi=320, facecolor="white")
    plt.close(fig)
    print(f"Figure {number}: saved; {nr * nc} grids checked.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets"))
    parser.add_argument("--work", type=Path, default=Path("runs/architectures"))
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--archive", type=Path, help="read figures/*.json and *.npz from an exported archive")
    sources.add_argument("--measurements", type=Path, help="directory containing the figure JSON and NPZ files")
    parser.add_argument("--figures", type=int, choices=(1, 2, 3), nargs="+", default=[1, 2, 3])
    parser.add_argument("--output", type=Path, help="default: WORK/figures-solid")
    args = parser.parse_args()
    from . import evaluation
    train = evaluation.dataset(args.data, train=True)
    measurements = args.measurements or args.work / "figures"
    out = args.output or args.work / "figures-solid"
    out.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.archive) if args.archive else nullcontext() as archive:
        for number in args.figures:
            meta = json.loads(read_result(archive, measurements, number, "json"))
            with np.load(BytesIO(read_result(archive, measurements, number, "npz")), allow_pickle=False) as grids:
                draw(number, meta, grids, train, out, evaluation)


if __name__ == "__main__":
    main()

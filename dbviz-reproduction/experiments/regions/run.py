"""Evaluate trained CIFAR-10 checkpoints and draw figures 1–3."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import torch

from ..arch.models import ARCHITECTURES, checkpoint, create
from .evaluation import (CLASSES, MEAN, PALETTE, SHUFFLE_SEEDS, STD, atomic,
                         dataset, digest, load_weights, make_plane, plot_anchors, predict_plane, save_json)


GROUPS = {
    1: ("fc", "vit", "wrn10"),
    2: ("vit", "dense", "resnet"),
    3: ("fc", "mixer", "vit", "vgg", "dense", "resnet", "wrn10"),
}
SEEDS = {1: (0,1), 2: (0,), 3: (0,1,2)}
NAMES = {"fc":"Fully connected","vit":"ViT","dense":"DenseNet-121","resnet":"ResNet-18",
         "wrn10":"WideResNet-28×10",
         "mixer":"MLP-Mixer","vgg":"VGG-19"}
SUBTITLES = {1:"Decision regions through natural images", 2:"Decision regions through pixel-shuffled images",
             3:"Decision regions across model architectures"}


def cached_visual(path, model, model_digest, images, grid, batch_size, device, input_size, cache):
    plane, coords, bounds = make_plane(images, grid, .5, device)
    identity = hashlib.sha256(torch.stack(images).numpy().tobytes()).hexdigest()
    signature = json.dumps([model_digest, identity, grid, .5, input_size, str(device),
                            str(torch.__version__), "v2-probability"])
    key = hashlib.sha256(signature.encode()).hexdigest()
    saved = cache / (key+".npz")
    if saved.exists():
        with np.load(saved, allow_pickle=False) as record:
            labels, confidence = record["labels"], record["confidence"]
    else:
        # A CUDA process holds at most one classifier for each display panel.
        if model[0] is None:
            model[0] = load_weights(create(model[1]), path, device)
        labels, confidence = predict_plane(model[0], plane, batch_size, input_size,
                                           return_confidence=True)
        with atomic(saved) as f:
            np.savez_compressed(f, labels=labels, confidence=confidence)
    if labels.shape != (grid*grid,) or labels.dtype != np.uint8 or int(labels.max()) > 9:
        raise ValueError(f"Corrupt visual prediction cache {saved}")
    if confidence.shape != labels.shape or not np.isfinite(confidence).all() or not (.099 <= confidence.min() <= confidence.max() <= 1.000001):
        raise ValueError(f"Invalid model confidence in {saved}")
    return labels, coords, bounds, confidence


def figure_stem(args):
    default = args.models == list(GROUPS[args.figure]) and args.seeds == list(SEEDS[args.figure])
    subset = "" if default else "_" + "-".join(args.models) + "_s" + "-".join(map(str, args.seeds))
    return f"figure{args.figure}{subset}_g{args.grid}"


def render_panels(figure_number, rows, titles, anchors, panels, output):
    nrows, ncols = len(rows), len(titles)
    include_thumbnails = figure_number == 2
    fig = plt.figure(figsize=(max(6.5, ncols*2.38+.9), nrows*1.8+1.55+(0.83 if include_thumbnails else 0)))
    gs = fig.add_gridspec(nrows + int(include_thumbnails), ncols,
                          height_ratios=[1.0]*nrows + ([.36] if include_thumbnails else []),
                          left=.105, right=.985, top=1-.82/fig.get_figheight(),
                          bottom=.62/fig.get_figheight(), hspace=.1, wspace=.06)
    cmap = ListedColormap(PALETTE)
    for (r, c), (labels, coords, extent, confidence) in panels.items():
        ax = fig.add_subplot(gs[r,c])
        side = round(len(labels)**.5)
        # Extent is in grid-centre coordinates, whereas imshow accepts pixel edges.
        dx = (extent[1]-extent[0])/(2*(side-1))
        dy = (extent[3]-extent[2])/(2*(side-1))
        edges = np.array(extent)+np.array([-dx,dx,-dy,dy])
        ax.imshow(labels.reshape(side,side).T, alpha=confidence.reshape(side,side).T,
                  origin="lower", aspect="auto", extent=edges,
                  cmap=cmap, vmin=-.5, vmax=9.5, interpolation="nearest")
        for (x,y),marker in zip(coords,("o","^","X")):
            ax.scatter(x,y,s=18,marker=marker,c="#202124",linewidths=.2)
        if r == 0:
            ax.set_title(titles[c],fontsize=9.5,pad=5)
        if c == 0:
            ax.set_ylabel(rows[r],fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    if include_thumbnails:
        for c, (images, _, _) in enumerate(anchors):
            tiles = [np.clip(im.permute(1,2,0).numpy()*np.array(STD)+np.array(MEAN),0,1) for im in images]
            separator = np.ones((32,2,3))
            strip = np.concatenate([tiles[0],separator,tiles[1],separator,tiles[2]],axis=1)
            ax = fig.add_subplot(gs[-1,c])
            ax.imshow(strip, interpolation="nearest")
            ax.axis("off")
    fig.suptitle(SUBTITLES[figure_number],fontsize=15,weight="semibold",y=.985)
    fig.text(.535, 1-.39/fig.get_figheight(), "Color = predicted class   ·   opacity = top-class probability",
             ha="center",va="top",size=8,color="#41505e")
    fig.legend([Patch(facecolor=color) for color in PALETTE], CLASSES, loc="lower center",
               frameon=False,ncol=min(10,5 if ncols<5 else 10),fontsize=7.5,
               handlelength=1.0,columnspacing=1.0,bbox_to_anchor=(.53,.012))
    save_plot(fig, output)


def save_plot(fig, path):
    with atomic(path) as f:
        fig.savefig(f, format="png",dpi=170,facecolor="white",bbox_inches="tight",pad_inches=.1)
    plt.close(fig)


def visual(args, files, output, info):
    train = dataset(args.data, train=True)
    shuffle = args.figure == 2
    anchors = [plot_anchors(train, s) for s in SHUFFLE_SEEDS] if shuffle else [plot_anchors(train)]
    info["input"] = {"split":"CIFAR-10 train", "random_augmentation":False,
                     "margin_each_side":.5, "column_sampling_seeds":list(SHUFFLE_SEEDS) if shuffle else None,
                     "permutation":"torch.randperm seeded per column, drawn independently per image, RGB channels preserved" if shuffle else None,
                     "image_indices":[a[1] for a in anchors], "anchor_classes":[a[2] for a in anchors]}
    coverage = {}
    panels = {}
    info["rendering"] = {"colors":"CIFAR-10 class palette from dbViz produce_plot_sepleg",
                         "opacity":"winning softmax probability with T=1 (produce_plot_alt convention)",
                         "raw_maps":"NPZ entries rowN_colM and rowN_colM_confidence, stored y by x from lower origin"}
    if shuffle:
        row_keys = [(m,s) for m in args.models for s in args.seeds]
        rows = [f"{NAMES[m]}\nseed {s}" for m,s in row_keys]
        titles = [" · ".join(CLASSES[i] for i in codes) for _,_,codes in anchors]
    else:
        rows = [f"seed {s}" for s in args.seeds]
        titles = [NAMES[m] for m in args.models]
    for m, s, path in files:
        state = [None, m]
        sha = info["models"][f"{m}/seed{s}"]
        for idx, (images, ids, _) in enumerate(anchors):
            print(f"Figure {args.figure}: {m}, seed {s}, images {ids}", flush=True)
            row, col = ((row_keys.index((m,s)), idx) if shuffle
                        else (args.seeds.index(s), args.models.index(m)))
            panel = cached_visual(path,state,sha,images,args.grid,args.batch_size,args.device,32,
                                  args.work/"cache"/"visual_planes")
            panels[row,col] = panel
            coverage[f"{m}/seed{s}/plane{idx+1}"] = {
                "image_indices": ids, "class_counts":np.bincount(panel[0], minlength=10).tolist()}
        del state
    info["class_order"] = CLASSES
    info["panel_counts"] = coverage
    render_panels(args.figure, rows, titles, anchors, panels, output.with_suffix(".png"))
    with atomic(output.with_suffix(".npz")) as f:
        maps = {}
        for (r,c), result in sorted(panels.items()):
            name = f"row{r+1}_col{c+1}"
            maps[name] = result[0].reshape(args.grid,args.grid).T
            maps[name+"_confidence"] = result[3].reshape(args.grid,args.grid).T
        np.savez_compressed(f, **maps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True,type=Path)
    parser.add_argument("--work",required=True,type=Path)
    parser.add_argument("--figure",required=True,type=int,choices=GROUPS)
    parser.add_argument("--models",nargs="+",choices=ARCHITECTURES)
    parser.add_argument("--seeds",nargs="+",type=int)
    parser.add_argument("--device",default="cuda",choices=["cuda","cpu","mps"])
    parser.add_argument("--grid",type=int,default=500)
    parser.add_argument("--batch-size",type=int,default=64)
    parser.add_argument("--threads",type=int,default=2)
    args=parser.parse_args()
    args.models = args.models or list(GROUPS[args.figure])
    args.seeds = args.seeds or list(SEEDS[args.figure])
    if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Provide unique architecture names and training seeds.")
    if args.grid<2 or min(args.batch_size,args.threads)<1:
        parser.error("Invalid grid, batch size or thread count.")
    torch.set_num_threads(args.threads)
    files = [(m,s,checkpoint(args.work,m,s)) for m in args.models for s in args.seeds]
    missing = [str(p) for _,_,p in files if not p.is_file()]
    if missing:
        parser.error("Missing trained checkpoints:\n" + "\n".join(missing))
    out = args.work/"figures"/figure_stem(args)
    info = {"figure":args.figure,"grid":args.grid,"device":args.device,"torch":str(torch.__version__),
            "architectures":args.models,"seeds":args.seeds,"models":{f"{m}/seed{s}":digest(p) for m,s,p in files},
            "wide_model_note":"The original visual captions specify WideResNet without a factor. Default display uses WideResNet-28×10."}
    visual(args,files,out,info)
    save_json(out.with_suffix(".json"),info)
    print(f"Saved {out.with_suffix('.png')}",flush=True)


if __name__ == "__main__":
    main()

"""Draw the four Chinese 500-epoch figures from the published measurements (CPU)."""
import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap, to_hex, to_rgb
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker
from matplotlib.patches import Patch
from fontTools import subset
from fontTools.ttLib import TTCollection, TTFont
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WIDTHS = [1, 2, 4, 6, 7, 8, 10, 12, 20, 40, 64]
PANEL_WIDTHS = [1, 4, 7, 10, 20, 64]
NOISE = "标签噪声"
WIDTH = "网络宽度参数 $k$"
FRAG = "平均连通区域数"
MARGIN = "类别区域边距中位数"
TITLES = ["从全部训练样本中随机选取三个样本", "从同一类别的正确标注样本中随机选取三个样本"]
GROUPS = ["全部样本", "正确标注", "错误标注"]
NAMES = "飞机 小汽车 鸟 猫 鹿 狗 青蛙 马 船 卡车".split()
CAPTIONS = [
    "（甲）三个样本分属不同类别。在两种噪声设置下，这三个样本的训练标签均正确。",
    "（乙）三个样本均为小汽车。在两种噪声设置下，这三个样本的训练标签均正确。",
]
FIGURE_TITLES = {6: "测试错误率", 7: "不同网络宽度下的决策区域",
                 10: "不同网络宽度下的决策区域碎片化程度", 12: "不同网络宽度下的类别区域边距"}
BASE_COLORS = ["#ffcb00", "#ff8000", "#2ca02c", "#00e5ec", "#a65628",
               "#ec00ec", "#ff003d", "#7900fa", "#0045ff", "#3bff00"]
PANEL_COLORS = [to_hex(0.82 * np.array(to_rgb(color)) + 0.18) for color in BASE_COLORS]
MARKERS = ["o", "^", "X"]


def choose_font(path, tmp):
    """Pick an installed Chinese face with full glyph coverage, including in TTCs."""
    visible = "".join([NOISE, WIDTH, FRAG, MARGIN, *TITLES, *GROUPS, *NAMES, *CAPTIONS,
                       *FIGURE_TITLES.values(), "真实类别训练标签噪声样本分组"])
    required = {ord(c) for c in visible if ord(c) > 127}
    bundled = ROOT / "assets" / "fonts" / "PlotLabelsSC-Regular.ttf"
    if path is None and bundled.is_file():
        with TTFont(bundled, lazy=True) as font:
            if not required.issubset(font.getBestCmap() or {}):
                raise ValueError("内附字体缺少本组图需要的字符。请检查安装或使用 --font。")
        font_manager.fontManager.addfont(bundled)
        family = font_manager.FontProperties(fname=bundled).get_name()
        return family, family
    if path:
        candidates = [path]
    else:
        favorites = ("notoserifcjk", "sourcehanserif", "notoserifsc", "notosanscjk",
                     "sourcehansans", "notosanssc", "songti", "simsun", "msyh",
                     "pingfang", "hiraginosansgb", "fandolsong", "arialunicode")
        ranked = []
        for item in font_manager.findSystemFonts():
            stem = Path(item).stem.lower().replace(" ", "").replace("-", "")
            for rank, family in enumerate(favorites):
                if family in stem:
                    ranked.append((rank, "bold" in stem, item))
                    break
        candidates = [Path(item) for _, _, item in sorted(ranked)]
    for candidate in candidates:
        if candidate.suffix.lower() in (".ttc", ".otc"):
            collection = TTCollection(candidate, lazy=True)
            faces = collection.fonts
        else:
            collection = None
            faces = [TTFont(candidate, lazy=True)]
        try:
            def rank_face(face):
                names = face["name"]
                family = names.getBestFamilyName() or ""
                style = names.getBestSubFamilyName() or ""
                return ("SC" not in family, style.lower() not in ("regular", "normal"))
            for face in sorted(faces, key=rank_face):
                if not required.issubset(face.getBestCmap() or {}):
                    continue
                family = face["name"].getBestFamilyName()
                keep = subset.Subsetter()
                keep.populate(unicodes=required | set(range(32, 127)))
                keep.subset(face)
                ext = ".ttf" if "glyf" in face else ".otf"
                font_path = tmp / ("chart-font" + ext)
                face.save(font_path)
                font_manager.fontManager.addfont(font_path)
                return font_manager.FontProperties(fname=font_path).get_name(), family
        finally:
            if collection:
                collection.close()
            else:
                faces[0].close()
    raise ValueError("未找到包含全部图中文字的字体。请安装 Noto CJK，或用 --font 指定简体中文字体文件。")


def load_curves(path):
    fields = {6: ["test_error"], 10: ["random_train_mean", "same_class_correct_mean"],
              12: ["all_median", "correct_median", "mislabeled_median"]}
    expected = {(f, field, n, k) for f, columns in fields.items() for field in columns
                for n in (0, 20) for k in WIDTHS if f != 12 or field == "all_median" or n == 20}
    values = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for line in csv.DictReader(handle):
            key = int(line["figure"]), line["metric"], int(line["label_noise_pct"]), int(line["width_k"])
            value = float(line["value"])
            if (key in values or int(line["epochs"]) != 500 or not np.isfinite(value)
                    or key not in expected or line["source"] != "reproduction_measured"):
                raise ValueError(f"实测曲线记录不匹配：{key}")
            values[key] = value
    if set(values) != expected:
        raise ValueError(f"曲线数据缺少 {sorted(expected - set(values))}")
    return {(f, field, n): [values[f, field, n, k] for k in WIDTHS]
            for f, field, n in sorted({key[:3] for key in expected})}


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(True, color="#cfcfcf", linewidth=.65)
    ax.tick_params(length=0, pad=4)
    for spine in ax.spines.values():
        spine.set(color="#bfbfbf", linewidth=.65)


def figure6(curves):
    fig, ax = plt.subplots(figsize=(10.3, 4.35))
    fig.subplots_adjust(left=.081, right=.984, bottom=.162, top=.963)
    style_axes(ax)
    ax.set(xlim=(0, 66), ylim=(.047, .521), xticks=range(0, 67, 2),
           yticks=[.1, .2, .3, .4, .5], xlabel=WIDTH, ylabel="测试错误率")
    for noise, color, marker in [(0, "#e7383e", "o"), (20, "#4d91c9", "X")]:
        ax.plot(WIDTHS, curves[6, "test_error", noise], lw=1.28, color=color, marker=marker,
                markeredgecolor="white", markeredgewidth=.45, markersize=8.6, label=f"{noise}%")
    ax.legend(loc="upper right", title=NOISE, framealpha=1, fontsize=14, title_fontsize=15.5)
    return fig


def truth_legend(fig, classes, y):
    parts = [TextArea("真实类别")]
    for marker, label in zip(MARKERS, classes):
        dot = DrawingArea(11, 15)
        dot.add_artist(Line2D([5.5], [6.8], marker=marker, color=PANEL_COLORS[label],
                              mec="none", ms=10, ls="none"))
        parts.append(HPacker(children=[dot, TextArea(NAMES[label])], align="center", pad=0, sep=3))
    strip = HPacker(children=parts, align="center", pad=0, sep=14)
    fig.add_artist(AnchoredOffsetbox(loc="center", child=strip, bbox_to_anchor=(.521, y),
                                     bbox_transform=fig.transFigure, pad=0, borderpad=0, frameon=False))


def load_figure7(path, widths):
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in ("labels", "anchors", "widths", "noise_percent", "true_labels", "ids")}
        labels, anchors = arrays["labels"], arrays["anchors"]
        classes, ids = arrays["true_labels"], arrays["ids"]
        valid = (np.array_equal(arrays["widths"], widths)
                 and np.array_equal(arrays["noise_percent"], [0, 20, 0, 20])
                 and np.array_equal(ids, [[672, 766, 781], [75, 32, 304]])
                 and np.array_equal(classes, [[9, 8, 6], [1, 1, 1]])
                 and labels.shape == (4, 6, 500, 500) and labels.dtype == np.uint8
                 and labels.max() < 10 and anchors.shape == (4, 6, 3, 2)
                 and np.isfinite(anchors).all() and np.all((anchors >= 0) & (anchors <= 1)))
        if not valid:
            raise ValueError("图 7 网格、样本编号或锚点与发布设置不一致。")
    return arrays


def figure7(path, widths=PANEL_WIDTHS):
    arrays = load_figure7(path, widths)
    labels, anchors, classes = arrays["labels"], arrays["anchors"], arrays["true_labels"]
    fig = plt.figure(figsize=(13.2, 8.9))
    cmap = ListedColormap(PANEL_COLORS)
    norm = BoundaryNorm(np.arange(-.5, 10.5), cmap.N)
    left, right, gap, height = .064, .987, .012, .161
    width = (right - left - gap * 5) / 6
    for row, bottom in enumerate((.737, .561, .260, .084)):
        for col, k in enumerate(widths):
            x = left + col * (width + gap)
            ax = fig.add_axes([x, bottom, width, height])
            ax.imshow(labels[row, col], cmap=cmap, norm=norm, interpolation="none", aspect="auto",
                      origin="lower", extent=(0, 1, 0, 1))
            for (px, py), marker in zip(anchors[row, col], MARKERS):
                ax.plot(px, py, marker=marker, ms=3.8, mec="none", mfc="#141414", ls="none")
            ax.set_axis_off()
            if row in (0, 2):
                fig.text(x + width / 2, bottom + height + .005, f"k = {k}",
                         fontsize=12.5, ha="center", va="bottom")
        fig.text(.038, bottom + height / 2, f"训练标签\n噪声 {(row % 2) * 20}%",
                 rotation=90, ha="center", va="center", fontsize=11)
    fig.legend([Patch(color=c) for c in PANEL_COLORS], NAMES, ncol=10, mode="expand",
               loc="upper center", bbox_to_anchor=(.055, .981, .925, .008),
               handlelength=1.65, handleheight=.75, borderaxespad=0)
    for group, caption in enumerate(CAPTIONS):
        truth_legend(fig, classes[group], (.538, .061)[group])
        fig.text(.522, (.499, .023)[group], caption, fontsize=12.4, ha="center", va="center")
    return fig


def figure10(curves):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8.4, 4.9))
    fig.subplots_adjust(left=.093, right=.986, top=.925, bottom=.132, hspace=.295)
    for ax, title, key in zip(axes, TITLES, ("random_train_mean", "same_class_correct_mean")):
        style_axes(ax)
        ax.set(xlim=(-2, 67), xticks=range(0, 61, 10), ylim=(1.55, 21.5), yticks=[5, 10, 15, 20])
        ax.set_title(title, fontsize=14.2, pad=4)
        for noise, color, fill, marker in [(0, "#4245ea", "#6465f1", "o"), (20, "#e64346", "#f36b6e", "X")]:
            ax.plot(WIDTHS, curves[10, key, noise], marker=marker, color=color, linewidth=.85,
                    markersize=6.7, markeredgewidth=.32, markerfacecolor=fill, label=f"{noise}%")
        ax.legend(loc="upper right", title=NOISE, fontsize=10.5, title_fontsize=10.5, framealpha=1)
    axes[-1].set_xlabel(WIDTH, labelpad=1.5)
    fig.text(.025, .51, FRAG, rotation=90, va="center", ha="center", fontsize=16)
    return fig


def swatch(color, text, marker=None):
    image = DrawingArea(24, 10)
    image.add_artist(Line2D([1, 22], [5, 5], color=color, linewidth=2.1, solid_capstyle="round"))
    if marker:
        image.add_artist(Line2D([11.5], [5], ls="none", marker=marker, color=color, ms=7))
    return HPacker(children=[image, TextArea(text)], align="center", pad=0, sep=4)


def figure12(curves):
    fig, ax = plt.subplots(figsize=(10.7, 5.65))
    fig.subplots_adjust(left=.091, right=.99, top=.966, bottom=.136)
    style_axes(ax)
    ax.set(xlim=(0, 66), xticks=range(0, 67, 2), ylim=(0, 60), yticks=range(0, 61, 10),
           xlabel=WIDTH, ylabel=MARGIN)
    colors = ["#0072b2", "#009e73", "#d55e00"]
    for color, field in zip(colors, ("all_median", "correct_median", "mislabeled_median")):
        for noise in ((0, 20) if field == "all_median" else (20,)):
            ax.plot(WIDTHS, curves[12, field, noise], lw=.9, marker=("o" if noise == 0 else "X"),
                    color=color, ms=9.5, mec="white", mew=.55)
    groups = [TextArea("样本分组")] + [swatch(c, name) for c, name in zip(colors, GROUPS)]
    noises = [TextArea(NOISE), swatch("#303030", "0%", "o"), swatch("#303030", "20%", "X")]
    columns = [VPacker(children=items, pad=0, sep=3.75, align="left") for items in (groups, noises)]
    block = HPacker(children=columns, align="top", pad=0, sep=23)
    legend = AnchoredOffsetbox(loc="lower right", child=block, pad=.52, borderpad=.58, frameon=True)
    legend.patch.set(boxstyle="round,pad=0,rounding_size=1.6", ec="#bfbfbf", lw=.65, fc="white")
    ax.add_artist(legend)
    return fig


def configure_style(path, stage):
    family, original = choose_font(path, stage)
    print(f"绘图字体：{original}")
    mpl.rcParams.update({"font.family": [family, "DejaVu Serif"], "font.size": 12,
                         "mathtext.fontset": "stix", "axes.labelsize": 16, "pdf.fonttype": 42,
                         "axes.unicode_minus": False, "text.color": "#222222",
                         "legend.edgecolor": "#bfbfbf", "savefig.facecolor": "white"})


def main():
    parser = argparse.ArgumentParser(description="从 500 轮实测数据重绘图 6、7、10、12 的中文 PDF")
    parser.add_argument("--results", type=Path, default=ROOT / "data", help="含 curves.csv 和 figure7.npz 的目录")
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "figures", help="输出目录")
    parser.add_argument("--font", type=Path, help="可选的简体中文字体文件（TTF、OTF、TTC、OTC）")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    curves = load_curves(args.results / "curves.csv")
    with TemporaryDirectory(dir=args.out) as tmp:
        stage = Path(tmp)
        configure_style(args.font, stage)
        combined = "figures-500epochs-zh.pdf"
        with PdfPages(stage / combined, metadata={"Title": "图6、7、10、12：500轮复现实测"}) as pdf:
            for number, draw in ((6, figure6), (7, figure7), (10, figure10), (12, figure12)):
                fig = draw(args.results / "figure7.npz") if number == 7 else draw(curves)
                try:
                    fig.savefig(stage / f"fig{number:02}.svg",
                                metadata={"Title": f"图{number}　{FIGURE_TITLES[number]}（500轮）"})
                    pdf.savefig(fig)
                finally:
                    plt.close(fig)
        for name in [*(f"fig{n:02}.svg" for n in FIGURE_TITLES), combined]:
            (stage / name).replace(args.out / name)
            print((args.out / name).resolve())


if __name__ == "__main__":
    main()

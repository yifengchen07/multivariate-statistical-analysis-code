"""Five-step diagram of the actual 224px, variable-width ResNet-18."""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/resnet-flow-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as DrawingPath
import numpy as np
from PIL import Image, ImageOps
import torch
from resnet224_models import WidthResNet18


# ---------- Editable appearance and output ----------
BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "resnet224/figures"
SOURCE_IMAGE = BASE.parent / "dataset/lulu/lulu_0023.jpg"
FIGSIZE = (22.4, 7.6)
DPI = 220
INK, MUTED = "#20384a", "#60717b"
BLUE, BLUE_FILL = "#456f89", "#dce8ed"
RUST, RUST_FILL = "#b96548", "#f0ded4"


def check_dimensions():
    """Read real tensor shapes from the model, rather than infer from the drawing."""
    torch.set_num_threads(4)
    width = 16
    model = WidthResNet18(width).eval()
    shapes, hooks = {}, []
    for name in ["conv1", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]:
        def record(module, inputs, output, key=name):
            shapes[key] = list(output.shape)
        hooks.append(getattr(model, name).register_forward_hook(record))
    with torch.inference_mode():
        model(torch.zeros(1, 3, 224, 224))
    for hook in hooks:
        hook.remove()
    expected = {
        "conv1": [1, width, 112, 112], "maxpool": [1, width, 56, 56],
        "layer1": [1, width, 56, 56], "layer2": [1, 2 * width, 28, 28],
        "layer3": [1, 4 * width, 14, 14], "layer4": [1, 8 * width, 7, 7],
        "avgpool": [1, 8 * width, 1, 1], "fc": [1, 4],
    }
    assert shapes == expected
    for number in range(1, 5):
        assert len(getattr(model, f"layer{number}")) == 2
    for number in [2, 3, 4]:
        projection = getattr(model, f"layer{number}")[0].downsample[0]
        assert projection.kernel_size == (1, 1) and projection.stride == (2, 2)
    return {"checked_base_width": width, "tensor_order_in_check": "N,C,H,W", "shapes": shapes}


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for name in ["Times New Roman.ttf", "simsun.ttc"]:
        fontManager.addfont(BASE / "resnet224/fonts" / name)
    font = FontProperties(family=["Times New Roman", "SimSun"])
    plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "path"})
    original_hash = hashlib.sha256(SOURCE_IMAGE.read_bytes()).hexdigest()
    with Image.open(SOURCE_IMAGE) as image:
        pixels = np.asarray(ImageOps.pad(image.convert("RGB"), (224, 224),
                            method=Image.Resampling.BILINEAR, color="white", centering=(.5, .5)))
    assert pixels.shape == (224, 224, 3)
    verified = check_dimensions()

    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 2240), ylim=(760, 0))
    ax.axis("off")

    def text(x, y, label, size=22, color=INK, **kwargs):
        ax.text(x, y, label, fontsize=size, color=color, ha="center", va="center",
                fontproperties=font, zorder=10, **kwargs)

    def box(x, y, width, height, fill=BLUE_FILL, edge=BLUE):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                     boxstyle="round,pad=0,rounding_size=11", facecolor=fill,
                     edgecolor=edge, linewidth=1.1, zorder=3))

    def arrow(start, end, waypoints=(), color=MUTED):
        points = [start, *waypoints, end]
        path = DrawingPath(points, [DrawingPath.MOVETO] + [DrawingPath.LINETO] * (len(points) - 1))
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=17,
                                    linewidth=1.25, color=color, zorder=5))

    # 1. Actual preprocessed input; the whole character is preserved.
    text(160, 50, "① 输入图像", size=27)
    ax.imshow(pixels, extent=(36, 284, 443, 195), interpolation="antialiased")
    ax.add_patch(Rectangle((36, 195), 248, 248, facecolor="none", edgecolor=MUTED,
                           linewidth=.8, linestyle="--"))
    text(160, 510, "224 × 224 × 3", size=28)
    text(160, 566, "等比缩放，居中补白", size=19, color=MUTED)
    text(160, 610, "整幅图像进入卷积网络", size=18, color=MUTED)

    # 2. Standard ResNet stem: 224 -> 112 -> 56, with the actual padding values.
    text(500, 50, "② 初始卷积与池化", size=27)
    box(365, 171, 270, 74)
    text(500, 208, "7 × 7 卷积", size=24)
    text(500, 274, "步长2，padding = 3", size=17, color=MUTED)
    text(500, 313, "112 × 112 × b", size=25, color=BLUE)
    arrow((500, 335), (500, 355))
    box(365, 365, 270, 74)
    text(500, 402, "3 × 3 最大池化", size=23)
    text(500, 469, "步长2，padding = 1", size=17, color=MUTED)
    text(500, 510, "56 × 56 × b", size=25, color=BLUE)
    text(500, 583, "初始卷积后接BN与ReLU", size=17, color=MUTED)
    arrow((295, 300), (355, 208), waypoints=[(330, 300), (330, 208)])

    # 3. Four stages; each contains two residual BasicBlocks.
    text(1085, 50, "③ 四组残差块", size=27)
    stage_x = [725, 915, 1105, 1295]
    dimensions = ["56 × 56 × b", "28 × 28 × 2b", "14 × 14 × 4b", "7 × 7 × 8b"]
    for number, (x, dimension) in enumerate(zip(stage_x, dimensions), start=1):
        box(x - 9, 208, 150, 150, fill="#edf3f6", edge="#a5b9c4")
        box(x - 4, 213, 150, 150, fill="#e5eef2", edge="#8ea8b6")
        box(x, 218, 150, 150)
        text(x + 75, 271, f"残差组{number}", size=21)
        text(x + 75, 321, "2个残差块", size=17, color=MUTED)
        text(x + 75, 413, dimension, size=21, color=BLUE)
        if number < 4:
            arrow((x + 157, 293), (stage_x[number] - 15, 293))
    arrow((645, 402), (710, 293), waypoints=[(680, 402), (680, 293)])
    text(1085, 466, "实验改变b；四阶段通道为b、2b、4b、8b", size=21, color=RUST)

    # A single residual block, expanded beneath the four stages.
    text(1085, 516, "每块：两次3 × 3卷积（padding = 1），与捷径相加", size=18, color=MUTED)
    box(825, 553, 170, 65, fill=RUST_FILL, edge=RUST)
    text(910, 575, "3 × 3 卷积", size=18, color=RUST)
    text(910, 602, "BN + ReLU", size=14, color=MUTED)
    box(1050, 553, 150, 65, fill=RUST_FILL, edge=RUST)
    text(1125, 575, "3 × 3 卷积", size=18, color=RUST)
    text(1125, 602, "BN", size=14, color=MUTED)
    arrow((760, 585), (815, 585))
    arrow((1003, 585), (1040, 585))
    arrow((1208, 585), (1277, 585))
    ax.add_patch(Circle((1300, 585), 18, facecolor="white", edgecolor=RUST, linewidth=1.3, zorder=7))
    text(1300, 585, "+", size=23, color=RUST)
    arrow((780, 585), (1300, 610), waypoints=[(780, 665), (1300, 665)], color=RUST)
    text(1040, 644, "捷径：保持或匹配尺寸", size=15, color=RUST)
    arrow((1325, 585), (1344, 585))
    box(1352, 559, 88, 52, fill="#f7f4ee", edge=MUTED)
    text(1396, 585, "ReLU", size=17)
    text(1085, 716, "跨组首块：主分支与1 × 1捷径均以步长2降采样。", size=17, color=MUTED)

    # 4. The average is over the 7x7 spatial grid, independently in each channel.
    text(1680, 50, "④ 全局平均池化", size=27)
    box(1560, 230, 240, 125)
    text(1680, 269, "全局平均池化", size=23)
    text(1680, 321, "7 × 7 → 1 × 1", size=24, color=BLUE)
    arrow((1457, 293), (1550, 293))
    text(1680, 422, "1 × 1 × 8b", size=26, color=BLUE)
    for column in [0, 1, 2, 3, 6, 7, 8, 9]:
        ax.add_patch(Rectangle((1566 + column * 24, 467), 18, 55,
                               facecolor=BLUE_FILL, edgecolor=BLUE, linewidth=.6))
    text(1686, 495, "···", size=23, color=MUTED)
    text(1680, 567, "展开：8b维向量", size=24)
    text(1680, 621, "每个通道对49个位置取均值", size=17, color=MUTED)

    # 5. Four unnormalized class scores; argmax produces the class prediction.
    text(2055, 50, "⑤ 四分类输出", size=27)
    box(1920, 230, 270, 125, fill=RUST_FILL, edge=RUST)
    text(2055, 269, "全连接层", size=25, color=RUST)
    text(2055, 321, "8b → 4", size=29, color=RUST)
    arrow((1810, 293), (1910, 293))
    text(1860, 260, "8b维", size=18, color=MUTED)
    arrow((2055, 365), (2055, 389))
    text(2055, 418, "4维类别得分", size=22)
    for label, x, y in [("噜噜", 1920, 455), ("奶龙", 2060, 455),
                         ("奶蛙", 1920, 524), ("牛来", 2060, 524)]:
        box(x, y, 130, 53, fill="#f6f2ec", edge="#d5c4b7")
        text(x + 65, y + 26, label, size=22)
    text(2055, 635, "取最高得分的角色作为预测", size=18, color=MUTED)

    # H x W x C is used consistently throughout the diagram (batch omitted).
    text(290, 716, "图中张量维度：高 × 宽 × 通道数", size=17, color=MUTED)
    for extension in ["png", "svg", "pdf"]:
        path = OUTPUT_DIR / f"resnet-224-flow.{extension}"
        fig.savefig(path, dpi=DPI, facecolor="white")
        print(path)
    plt.close(fig)
    assert hashlib.sha256(SOURCE_IMAGE.read_bytes()).hexdigest() == original_hash
    metadata = dict(
        architecture="WidthResNet18", blocks_per_stage=[2, 2, 2, 2],
        source_image=str(SOURCE_IMAGE.relative_to(BASE.parent)), original_sha256=original_hash,
        source_image_unchanged=True, input_shape_hwc=[224, 224, 3],
        shape_notation="H x W x C; batch dimension omitted", base_width="b (symbolic)",
        stem_conv="7x7 stride2 padding3, then BN+ReLU", stem_pool="3x3 stride2 padding1",
        stages_hwc=["56x56xb", "28x28x2b", "14x14x4b", "7x7x8b"],
        average_pool_hwc="1x1x8b", flatten_dimension="8b", class_scores=4,
        layout="horizontal five steps, no overall title", verified_model_dimensions=verified,
        fonts={"Chinese": "SimSun", "English_and_numbers": "Times New Roman"},
        residual_projection="First block of stages 2,3,4 uses a 1x1 stride2 projection shortcut.")
    (OUTPUT_DIR / "resnet-224-flow-dimensions.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

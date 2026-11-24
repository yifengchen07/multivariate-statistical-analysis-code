"""Five-step 224px / 16px patch diagram with standard evaluation preprocessing."""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/vit-diagram-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.path import Path as MPath
import numpy as np
from PIL import Image, ImageOps

BASE = Path(__file__).resolve().parent
OUT = BASE / 'vit' / 'intro'
FONT_FILES = ['/System/Library/Fonts/Supplemental/Times New Roman.ttf',
              '/Users/chan/Library/Fonts/simsun.ttc']
for font_file in FONT_FILES:
    fontManager.addfont(font_file)
FONT = FontProperties(family=['Times New Roman', 'SimSun'])
INK, MUTED = '#20384A', '#60717B'
BLUE, BLUE_FILL = '#456F89', '#DCE8ED'
RUST, RUST_FILL = '#B96548', '#F0DED4'


def main():
    OUT.mkdir(exist_ok=True)
    source = BASE.parent / 'dataset/lulu/lulu_0023.jpg'
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    with Image.open(source) as image:
        pixels = np.asarray(image.convert('RGB'))
    height, width, channels = pixels.shape
    assert (width, height, channels) == (900, 1600, 3)
    # Preserve the whole character: resize to fit, then pad to the agreed input size.
    input_side, p = 224, 16
    resized_width, resized_height = ImageOps.contain(Image.fromarray(pixels),(224,224),
                                      method=Image.Resampling.BILINEAR).size
    inputs = np.asarray(ImageOps.pad(Image.fromarray(pixels),(224,224),
                        method=Image.Resampling.BILINEAR,color='white',centering=(.5,.5)))
    assert inputs.shape == (224, 224, 3)
    rows, columns = input_side // p, input_side // p
    patches = inputs.reshape(rows, p, columns, p, 3).transpose(0, 2, 1, 3, 4)
    assert patches.shape == (14, 14, 16, 16, 3)
    variance = patches.astype(float).var(axis=(2, 3)).mean(axis=2)
    restricted = np.full_like(variance, -1)
    restricted[3:11, 3:11] = variance[3:11, 3:11]
    r, c = np.unravel_index(restricted.argmax(), restricted.shape)
    patch = patches[r, c]
    assert patch.shape == (p, p, 3) and patch.size == 3*p*p

    plt.rcParams.update({'svg.fonttype': 'path', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(19.2, 6.2), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1920), ylim=(620, 0))
    ax.axis('off')

    def text(x, y, s, size=21, color=INK, ha='center'):
        return ax.text(x, y, s, fontsize=size, ha=ha, va='center',
                       color=color, fontproperties=FONT, zorder=10)

    def arrow(start, end, waypoints=(), color=MUTED):
        points = [start, *waypoints, end]
        path = MPath(points, [MPath.MOVETO] + [MPath.LINETO] * (len(points)-1))
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle='-|>',
                     mutation_scale=19, linewidth=1.6, color=color, zorder=6))

    text(193, 48, '① 等比缩放与补白', size=25)
    text(530, 48, '② 取出一个图块', size=25)
    left, top, image_w = 53, 158, 280
    image_h = image_w
    ax.imshow(inputs, extent=(left, left+image_w, top+image_h, top),
              interpolation='antialiased')
    ax.add_patch(Rectangle((left, top), image_w, image_h, facecolor='none',
                          edgecolor=MUTED, linewidth=.9, linestyle='--'))
    # The grid has exactly fourteen rows and fourteen columns of 16px patches.
    for i in range(columns+1):
        x = left + i*image_w/columns
        ax.plot([x, x], [top, top+image_h], color=INK, alpha=.4, linewidth=.55)
    for i in range(rows+1):
        y = top + i*image_h/rows
        ax.plot([left, left+image_w], [y, y], color=INK, alpha=.4, linewidth=.55)
    hx = left + c*p/input_side*image_w
    hy = top + r*p/input_side*image_h
    ax.add_patch(Rectangle((hx, hy), image_w/columns, image_h/rows,
                          edgecolor=RUST, facecolor='none', linewidth=2.0, zorder=7))
    text(193, 494, '输入：224 × 224 × 3', size=22)
    text(193, 537, '保持长宽比，完整保留角色', size=18, color=MUTED)
    text(193, 578, '居中补白至224 × 224', size=17, color=MUTED)

    px, py, side = 416, 184, 228
    ax.imshow(patch, extent=(px, px+side, py+side, py), interpolation='nearest')
    for i in range(p+1):
        v = i*side/p
        ax.plot([px+v, px+v], [py, py+side], color='white', linewidth=.5)
        ax.plot([px, px+side], [py+v, py+v], color='white', linewidth=.5)
    arrow((hx+image_w/columns+3, hy+image_h/rows/2), (px-15, py+side/2),
          waypoints=[(340, hy+image_h/rows/2), (375, py+side/2)], color=RUST)
    text(530, 474, '16 × 16 × 3 = 768', size=24, color=RUST)
    text(530, 518, '每个像素有3个颜色值', size=18, color=MUTED)

    arrow((660, 300), (726, 300))
    text(908, 48, '③ 展开并拼接像素', size=25)
    for ch, (label, y, color) in enumerate(zip(['R', 'G', 'B'],
                     [235, 300, 365], ['#C47A75', '#80A78D', '#7D9DB9'])):
        text(756, y, label, size=22, color=color)
        for i, index in enumerate([0, 1, 2, 3, 4, 5, -6, -5, -4, -3, -2, -1]):
            value = patch[:, :, ch].flatten()[index]
            column = i if i < 6 else i+3
            ax.add_patch(Rectangle((785+column*16, y-15), 13, 30,
                            facecolor=color, alpha=.3+.65*float(value)/255, linewidth=0))
        for offset in [-8, 0, 8]:
            ax.add_patch(Circle((898+offset, y), radius=1.7, facecolor=color, edgecolor='none'))
        text(1057, y, '256', size=18, color=MUTED)
    text(908, 474, '256 + 256 + 256 = 768维', size=22)
    text(908, 518, '拼接为一条向量', size=19, color=MUTED)
    text(1312, 48, '④ 线性映射', size=25)
    box = FancyBboxPatch((1174, 223), 276, 155, boxstyle='round,pad=0,rounding_size=12',
                          facecolor=RUST_FILL, edgecolor=RUST, linewidth=1.2)
    ax.add_patch(box)
    text(1312, 300, '768 → d', size=33, color=RUST)
    text(1312, 474, '所有图块共享映射权重', size=19, color=MUTED)
    text(1312, 518, '改变的是映射后的宽度 d', size=19, color=RUST)
    arrow((1090, 300), (1155, 300))

    # Row count is fixed at 196; ellipses indicate omitted rows and variable width d.
    mx, my, cell_w, cell_h = 1593, 195, 24, 21
    for row in list(range(4)) + list(range(6, 10)):
        for col in list(range(4)) + list(range(6, 10)):
            ax.add_patch(Rectangle((mx+col*cell_w, my+row*cell_h), 20.5, 17.5,
                                  facecolor=BLUE_FILL, edgecolor=BLUE, linewidth=.25))
        for offset in [-7, 0, 7]:
            ax.add_patch(Circle((mx+4.8*cell_w+offset, my+row*cell_h+8), radius=1.2,
                                facecolor=MUTED, edgecolor='none'))
    for offset in [-8, 0, 8]:
        ax.add_patch(Circle((mx+119, my+4.8*cell_h+offset), radius=1.8,
                            facecolor=MUTED, edgecolor='none'))
    text(mx+119, my-39, 'd列', size=20, color=MUTED)
    arrow((1468, 300), (1574, 300))
    text(1712, 48, '⑤ 汇集图块向量', size=25)
    text(1712, 474, '196 × d', size=30, color=BLUE)
    text(1712, 518, '14 × 14 = 196个图块', size=19)
    text(1712, 564, '每行一个图块，每列一个特征', size=17, color=MUTED)

    for suffix in ['png', 'svg', 'pdf']:
        path = OUT / ('vit-224-patch-flow.' + suffix)
        fig.savefig(path, dpi=220, facecolor='white',
                    metadata={'Creator': '224px input, 16px patches, variable embedding width diagram'})
        print(path)
    plt.close(fig)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    metadata = dict(source=str(source.relative_to(BASE.parent)), sha256=before,
                    original_width=width, original_height=height, rgb_channels=channels,
                    preprocessing='Aspect-preserving bilinear resize to fit 224, then centered white padding to 224x224.',
                    resized_width=resized_width, resized_height=resized_height,
                    input_shape=list(inputs.shape),
                    patch_size=p, selected_patch_row=int(r), selected_patch_column=int(c),
                    selected_patch_rgb=patch.tolist(), patch_pixel_dimensions=768,
                    patch_count=rows*columns, embedding_width='d (symbolic)',
                    image_unchanged=True, layout='horizontal, five steps, no overall title',
                    fonts={'Chinese':'SimSun', 'English_and_numbers':'Times New Roman'},
                    note='Five-step 224px concept figure for the requested future experiment; does not describe or rerun the previous training.')
    (OUT / 'vit-224-patch-flow-dimensions.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

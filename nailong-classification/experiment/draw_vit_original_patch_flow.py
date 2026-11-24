"""Requested five-step original-resolution patch embedding diagram; no retraining."""
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
from PIL import Image

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
    # Only add white pixels: never resample the original image.
    padded_side = max(width, height)
    pad_left, pad_top = (padded_side-width)//2, (padded_side-height)//2
    padded = np.full((padded_side, padded_side, 3), 255, dtype=np.uint8)
    padded[pad_top:pad_top+height, pad_left:pad_left+width] = pixels
    assert np.array_equal(padded[pad_top:pad_top+height, pad_left:pad_left+width], pixels)
    # This raster inset illustrates one possible P; all displayed dimensions stay symbolic.
    raster_patch_side = 16
    p = raster_patch_side
    rows, columns = padded_side // p, padded_side // p
    patches = padded.reshape(rows, p, columns, p, 3).transpose(0, 2, 1, 3, 4)
    variance = patches.astype(float).var(axis=(2, 3)).mean(axis=2)
    restricted = np.full_like(variance, -1)
    restricted[35:70, 30:70] = variance[35:70, 30:70]
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

    text(193, 48, '① 补白，不缩放', size=25)
    text(530, 48, '② 取出一个图块', size=25)
    left, top, image_w = 53, 158, 280
    image_h = image_w
    ax.imshow(padded, extent=(left, left+image_w, top+image_h, top),
              interpolation='antialiased')
    ax.add_patch(Rectangle((left, top), image_w, image_h, facecolor='none',
                          edgecolor=MUTED, linewidth=.9, linestyle='--'))
    for boundary in [pad_left, pad_left+width]:
        bx = left+boundary/padded_side*image_w
        ax.plot([bx, bx], [top, top+image_h], color=MUTED, alpha=.5,
                linewidth=.65, linestyle='--')
    # A light grid locates the illustrated patch without committing the diagram to P=16.
    for i in range(columns+1):
        x = left + i*image_w/columns
        ax.plot([x, x], [top, top+image_h], color=INK, alpha=.08, linewidth=.16)
    for i in range(rows+1):
        y = top + i*image_h/rows
        ax.plot([left, left+image_w], [y, y], color=INK, alpha=.08, linewidth=.16)
    hx = left + c*p/padded_side*image_w
    hy = top + r*p/padded_side*image_h
    ax.add_patch(Rectangle((hx, hy), image_w/columns, image_h/rows,
                          edgecolor=RUST, facecolor=RUST, linewidth=1.5, zorder=7))
    ax.add_patch(Circle((hx, hy), radius=12, facecolor='none',
                        edgecolor=RUST, linewidth=1, zorder=7))
    text(193, 494, '原图：宽900 × 高1600', size=19)
    text(193, 537, '补白后：H′ × W′ × 3', size=19, color=MUTED)
    text(193, 578, 'H′、W′为P的整数倍', size=17, color=MUTED)

    px, py, side = 416, 184, 228
    ax.imshow(patch, extent=(px, px+side, py+side, py), interpolation='nearest')
    for i in range(p+1):
        v = i*side/p
        ax.plot([px+v, px+v], [py, py+side], color='white', linewidth=.5)
        ax.plot([px, px+side], [py+v, py+v], color='white', linewidth=.5)
    arrow((hx+14, hy), (px-15, py+side/2),
          waypoints=[(340, hy), (375, py+side/2)], color=RUST)
    text(530, 474, 'P × P × 3', size=26, color=RUST)
    text(530, 518, 'P为图块边长，可自行选择', size=18, color=MUTED)

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
        text(1055, y, 'P²', size=18, color=MUTED)
    text(908, 474, 'P² + P² + P² = 3P²维', size=22)
    text(908, 518, '拼接为一条向量', size=19, color=MUTED)
    text(1312, 48, '④ 线性映射', size=25)
    box = FancyBboxPatch((1174, 223), 276, 155, boxstyle='round,pad=0,rounding_size=12',
                          facecolor=RUST_FILL, edgecolor=RUST, linewidth=1.2)
    ax.add_patch(box)
    text(1312, 300, '3P² → d', size=33, color=RUST)
    text(1312, 474, '所有图块共享映射权重', size=19, color=MUTED)
    text(1312, 518, '改变的是映射后的宽度 d', size=19, color=RUST)
    arrow((1090, 300), (1155, 300))

    # Both the number of rows N and number of columns d remain variable.
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
    text(1712, 474, 'N × d', size=30, color=BLUE)
    text(1712, 518, 'N = (H′/P) × (W′/P)', size=19)
    text(1712, 564, '每行一个图块，每列一个特征', size=17, color=MUTED)

    for suffix in ['png', 'svg', 'pdf']:
        path = OUT / ('vit-original-patch-flow.' + suffix)
        fig.savefig(path, dpi=220, facecolor='white',
                    metadata={'Creator': 'Original-resolution patch embedding diagram'})
        print(path)
    plt.close(fig)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    metadata = dict(source=str(source.relative_to(BASE.parent)), sha256=before,
                    original_width=width, original_height=height, rgb_channels=channels,
                    padding=dict(left=pad_left, right=padded_side-width-pad_left,
                                 top=pad_top, bottom=padded_side-height-pad_top),
                    padded_width=padded_side, padded_height=padded_side,
                    original_pixels_exactly_preserved=True,
                    patch_size='P (symbolic; raster inset is only an illustration)',
                    raster_inset_patch_side=p, selected_patch_row=int(r), selected_patch_column=int(c),
                    selected_patch_rgb=patch.tolist(), patch_pixel_dimensions='3*P**2',
                    patch_count='(H_padded/P)*(W_padded/P)', embedding_width='d (symbolic)',
                    image_unchanged=True, layout='horizontal, five steps, no overall title',
                    fonts={'Chinese':'SimSun', 'English_and_numbers':'Times New Roman'},
                    note='Five-step original-resolution concept figure for the requested future experiment; does not describe or rerun the previous training.')
    (OUT / 'vit-original-patch-flow-dimensions.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

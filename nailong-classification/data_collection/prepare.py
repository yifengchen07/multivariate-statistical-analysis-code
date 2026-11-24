"""Verify downloaded images, select one GIF frame, and export reviewed classes.

Run from any directory: python3 data_collection/prepare.py
Requires Pillow, numpy, opencv-python. Existing naiwa/nailong files are read only.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json

import cv2
import numpy as np
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def rgb(image):
    image = ImageOps.exif_transpose(image)
    if image.mode in ('RGBA', 'LA', 'P'):
        rgba = image.convert('RGBA')
        canvas = Image.new('RGBA', rgba.size, 'white')
        image = Image.alpha_composite(canvas, rgba)
    return image.convert('RGB')


def feature(image):
    # Trim only for comparisons; the saved sample retains the complete picture.
    a = np.asarray(image)
    mask = np.any(a < 235, axis=2)
    yy, xx = np.where(mask)
    if len(xx) and xx.max() - xx.min() > 40 and yy.max() - yy.min() > 40:
        image = image.crop((int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1))
    hashes = []
    for box in [None, (0, 0, image.width, max(1, int(image.height * .78)))]:
        part = image if box is None else image.crop(box)
        gray = cv2.cvtColor(np.asarray(part.resize((32, 32))), cv2.COLOR_RGB2GRAY)
        dct = cv2.dct(gray.astype(np.float32))[:8, :8].ravel()[1:]
        bits = dct > np.median(dct)
        hashes.append(sum(int(v) << j for j, v in enumerate(bits)))
    small = np.asarray(image.resize((32, 32)), dtype=np.float32)
    return hashes, small


def distance(a, b):
    return min((a[0][i] ^ b[0][i]).bit_count() for i in range(2))


def main():
    if (ROOT / 'dataset_info.json').exists():
        raise SystemExit('当前数据集已经人工清洗并编号。历史采集处理脚本已停用，以免恢复已删除图片或破坏当前编号。')
    previous = json.loads((HERE / 'audit.json').read_text()) if (HERE / 'audit.json').exists() else []
    sources = json.loads((HERE / 'sources.json').read_text())
    exclusions = json.loads((HERE / 'review_exclusions.json').read_text())
    audit, decoded = [], []
    for source in sources:
        row = dict(source)
        row['retrieved_utc'] = '2026-11-19'
        path = HERE / 'raw' / row['id']
        try:
            if not path.exists():
                row['status'] = 'download_unavailable'
                audit.append(row)
                continue
            original = path.read_bytes()
            row['original_sha256'] = hashlib.sha256(original).hexdigest()
            with Image.open(path) as im:
                count = getattr(im, 'n_frames', 1)
                frame = count // 2
                im.seek(frame)
                image = rgb(im)
                row.update(original_format=im.format, original_frames=count,
                           selected_frame=frame, width=image.width, height=image.height)
            if min(image.size) < 96:
                row['status'] = 'rejected_low_resolution_or_placeholder'
                audit.append(row)
                continue
            reason = next((v for k, v in exclusions.items() if row['id'].startswith(k)), None)
            if reason:
                row.update(status='rejected_visual_review', reason=reason)
                audit.append(row)
                continue
            row['status'] = 'pending'
            audit.append(row)
            decoded.append((row, image, feature(image)))
        except Exception as error:
            row.update(status='rejected_invalid_image', reason=str(error))
            audit.append(row)

    # Prefer larger originals when the same picture appears on several websites.
    decoded.sort(key=lambda x: (-x[1].width * x[1].height, x[0]['id']))
    kept = []
    for row, image, feat in decoded:
        duplicate = next((other for other, _, old_feat in kept
                          if row['class'] == other['class']
                          and distance(feat, old_feat) <= 5
                          and np.mean(np.abs(feat[1] - old_feat[1])) < 23), None)
        if duplicate:
            row.update(status='rejected_near_duplicate', duplicate_of=duplicate['id'])
            continue
        target = ROOT / 'dataset' / row['class'] / (row['class'] + '_' + row['id'] + '.jpg')
        target.parent.mkdir(parents=True, exist_ok=True)
        exported = image.copy()
        exported.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
        exported.save(target, quality=95, subsampling=0)
        row.update(status='accepted', relative_path=target.relative_to(ROOT).as_posix(),
                   output_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                   review='visually_checked',
                   depiction='web_remix_or_sticker' if any(h in row['url'] for h in ['niulai.co', 'dps.media']) else 'web_image')
        kept.append((row, image, feat))

    # Similarity groups are deliberately broader than the duplicate filter.
    # They flag possible leakage; they are not a guarantee of independent scenes.
    parent = list(range(len(kept)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, (a, _, fa) in enumerate(kept):
        for j in range(i):
            b, _, fb = kept[j]
            if a['class'] == b['class'] and distance(fa, fb) <= 14:
                parent[find(i)] = find(j)
    for i, (row, _, _) in enumerate(kept):
        row['similarity_group'] = row['class'] + '_' + kept[find(i)][0]['id']

    (HERE / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    accepted = [row for row in audit if row['status'] == 'accepted']
    current = {row['relative_path'] for row in accepted}
    for row in previous:
        relative = row.get('relative_path')
        if relative and relative not in current:
            old = ROOT / relative
            if old.exists() and hashlib.sha256(old.read_bytes()).hexdigest() == row.get('output_sha256'):
                old.unlink()
    fields = ['relative_path', 'class', 'id', 'page_url', 'url', 'original_sha256',
              'output_sha256', 'original_format', 'original_frames', 'selected_frame',
              'width', 'height', 'similarity_group', 'depiction', 'retrieved_utc', 'alt']
    with (HERE / 'manifest.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(accepted)
    counts = {p.name: sum(f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}
                         for f in p.iterdir()) for p in (ROOT / 'dataset').iterdir() if p.is_dir()}
    report = {'checked_utc': datetime.now(timezone.utc).isoformat(),
              'candidate_count': len(sources), 'added_counts': dict(Counter(x['class'] for x in accepted)),
              'dataset_counts': counts, 'audit_status_counts': dict(Counter(x['status'] for x in audit))}
    (HERE / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

"""Prepare explicit 224px inputs with the existing split and fixed training noise."""
from pathlib import Path
import hashlib
import json
import shutil
import csv
from collections import defaultdict
import numpy as np
from PIL import Image, ImageOps

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
OLD = BASE / 'vit'
OUT = BASE / 'vit224'
WIDTHS = [2, 4, 6, 8, 10, 12, 14, 16, 24, 48, 96, 192]


def preprocess(image):
    image = image.convert('RGB')
    return ImageOps.pad(image,(224,224),method=Image.Resampling.BILINEAR,
                        color='white',centering=(.5,.5))


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'config.json').exists():
        print('Prepared configuration already exists; frozen inputs will be verified by trainer.')
        return
    with (OLD / 'splits.csv').open(encoding='utf-8-sig') as f:
        samples = list(csv.DictReader(f))
    old = np.load(OLD / 'data.npz')
    rgb = []
    for row in samples:
        image_path = ROOT / row['relative_path']
        assert hashlib.sha256(image_path.read_bytes()).hexdigest() == row['sha256']
        with Image.open(image_path) as image:
            rgb.append(np.asarray(preprocess(image)).transpose(2, 0, 1))
    rgb = np.stack(rgb)
    assert rgb.shape == (417, 3, 224, 224)
    duplicates = defaultdict(list)
    for i, image in enumerate(rgb):
        duplicates[hashlib.sha256(image.tobytes()).hexdigest()].append(i)
    for group in duplicates.values():
        assert len({samples[i]['split'] for i in group}) == 1, ('224px duplicate across splits', group)
        assert len({int(old['y_noise'][i]) for i in group}) == 1, ('224px duplicate conflicting noise', group)
    x = rgb.astype(np.float32) / 255
    mean = x[old['train']].mean(axis=(0, 2, 3), keepdims=True)
    std = x[old['train']].std(axis=(0, 2, 3), keepdims=True).clip(.05)
    x = (x-mean)/std
    np.savez_compressed(OUT/'data.npz', x=x, y=old['y'], y_noise=old['y_noise'],
                        train=old['train'], validation=old['validation'], test=old['test'], mean=mean, std=std)
    for filename in ['splits.csv', 'noise_labels.csv']:
        shutil.copy2(OLD/filename, OUT/filename)
    config = json.loads((OLD/'config.json').read_text())
    config.update(image_size=224, patch_size=16, widths=WIDTHS, noise_rates=[.2],
                  seeds=[42,43,44], eval_batch_size=64,
                  preprocessing='RGB; aspect-preserving bilinear resize to fit 224x224; centered white padding; scale to [0,1]; training-only channel mean/std.',
                  training_preprocessing='Same deterministic letterbox for all splits; no augmentation, to isolate width. Preserve the whole character. This is a small four-layer ViT, not pretrained ViT-B/16.',
                  test_protocol='Exploratory introduction. All 12 widths and 3 seeds fixed before this sweep. Evaluate final epoch once; select illustrative seed/images/columns afterward and disclose selection.',
                  comparison_note='224px rerun. Center-crop pilot stopped after visual inspection found a character head was cropped out. Pilot and original low-resolution results stored separately, never pooled.',
                  noise_protocol='Preserve the same 59 wrong training labels and all clean held-out labels from the previous experiment.')
    (OUT/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
    shutil.copy2(BASE/'models.py', OUT/'models_snapshot.py')
    shutil.copy2(BASE/'train_sweep.py', OUT/'train_sweep_snapshot.py')
    paths = [ROOT/'dataset_manifest.csv', OLD/'splits.csv', OLD/'noise_labels.csv',
             OUT/'config.json', OUT/'data.npz', OUT/'splits.csv', OUT/'noise_labels.csv',
             OUT/'models_snapshot.py', OUT/'train_sweep_snapshot.py', BASE/'models.py',
             BASE/'train_sweep.py', BASE/'prepare_vit224.py']
    fingerprints = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT/'input_fingerprints.json').write_text(json.dumps(fingerprints, indent=2))
    metadata = dict(images=417, input_shape=list(x.shape), split_counts=config['split_counts'],
                    flipped_train_labels=int(np.sum(old['y']!=old['y_noise'])),
                    unique_preprocessed_inputs=len(duplicates), token_count=197,
                    original_images_unchanged=True, identical_input_leakage=False,
                    width_grid=WIDTHS, seeds=config['seeds'], expected_runs=36,
                    channel_mean=mean.reshape(-1).tolist(), channel_std=std.reshape(-1).tolist())
    (OUT/'preparation.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata), flush=True)


if __name__ == '__main__':
    main()

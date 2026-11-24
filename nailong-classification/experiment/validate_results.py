"""Check data isolation, fixed corruption, and predictions against saved metrics."""
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import csv
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = HERE


def read_csv(path):
    with path.open(encoding='utf-8-sig') as file:
        return list(csv.DictReader(file))


def main():
    global HERE
    parser=argparse.ArgumentParser();parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--output-dir',type=Path,default=BASE);args=parser.parse_args()
    HERE=args.output_dir.resolve()
    config=json.loads((HERE/'config.json').read_text())
    samples=read_csv(HERE/'splits.csv'); noise=read_csv(HERE/'noise_labels.csv')
    with np.load(HERE/'data.npz') as cache:
        data={key:cache[key] for key in cache.files}
    fingerprints=json.loads((HERE/'input_fingerprints.json').read_text())
    assert data['x'].shape==(417,3,config['image_size'],config['image_size'])
    for relative,expected in fingerprints.items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,relative
    with (ROOT/'dataset_manifest.csv').open(encoding='utf-8-sig') as file:
        manifest=list(csv.DictReader(file))
    assert len(samples)==len(manifest)==417
    for row in manifest:
        assert hashlib.sha256((ROOT/row['relative_path']).read_bytes()).hexdigest()==row['sha256']
    known_groups=defaultdict(set); pixels=defaultdict(list)
    for i,row in enumerate(samples):
        assert row['sample_id']==noise[i]['sample_id']
        known_groups[row['group_id']].add(row['split'])
        pixels[hashlib.sha256(data['x'][i].tobytes()).hexdigest()].append(i)
        assert int(noise[i]['clean_label'])==data['y'][i]==int(row['clean_label'])
        assert int(noise[i]['noisy_label'])==data['y_noise'][i]
        if row['split']!='train':assert data['y'][i]==data['y_noise'][i]
    assert all(len(s)==1 for s in known_groups.values())
    for group in pixels.values():
        assert len({samples[i]['split'] for i in group})==1
        assert len({int(data['y_noise'][i]) for i in group})==1
    actual_flips=sum(data['y'][i]!=data['y_noise'][i] for i in data['train'])
    assert actual_flips==config['flipped_train_labels']==59
    for split in ['train','validation','test']:
        assert set(map(int,data[split]))=={i for i,r in enumerate(samples) if r['split']==split}
    results=read_csv(HERE/'results.csv');predictions=read_csv(HERE/'test_predictions.csv')
    byrun=defaultdict(list)
    for row in predictions:byrun[row['run']].append(row)
    assert len(results)==len(byrun)
    expected=len(config['widths'])*len(config['seeds'])*len(config['noise_rates'])
    if not args.allow_partial:assert len(results)==expected,(len(results),expected)
    for row in results:
        rr=byrun[row['run']];assert len(rr)==len(data['test'])==62
        assert len({p['sample_id'] for p in rr})==62
        error=np.mean([p['predicted_label']!=p['true_label'] for p in rr])
        assert abs(error-float(row['test_error']))<1e-6
        checkpoint=HERE/'runs'/row['run']/'model.pt';assert checkpoint.is_file()
        for p in rr:
            probabilities=[float(p['p_'+c]) for c in config['class_names']]
            assert abs(sum(probabilities)-1)<1e-5
            assert int(np.argmax(probabilities))==int(p['predicted_label'])
            sample=next(s for s in samples if s['sample_id']==p['sample_id'])
            assert sample['split']=='test' and sample['clean_label']==p['true_label']
    report=dict(status='passed',completed_runs=len(results),expected_runs=expected,
                dataset_images=417,split_counts=dict(Counter(r['split'] for r in samples)),
                original_image_sha256_unchanged=True,known_group_leakage=False,
                identical_preprocessed_input_leakage=False,conflicting_duplicate_noise=False,
                flipped_training_labels=int(actual_flips),held_out_labels_clean=True,
                recorded_predictions_match_metrics=True,all_checkpoint_files_present=True,
                predictions=len(predictions))
    (HERE/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()

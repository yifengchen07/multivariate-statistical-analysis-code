"""Verify final ResNet predictions and render complete curves and optional examples."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from resnet224_models import build_model
from report_helpers import select_case, choose_images, choose_columns, render_board

BASE = Path(__file__).resolve().parent
FIG = BASE / 'figures'


def ce(logits, target):
    logits = logits.astype(np.float64)
    maximum = logits.max(1)
    return maximum + np.log(np.exp(logits - maximum[:, None]).sum(1)) - logits[np.arange(len(target)), target]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = json.loads((BASE / 'config.json').read_text())
    config_hash = hashlib.sha256((BASE / 'config.json').read_bytes()).hexdigest()
    results = pd.read_csv(BASE / 'results.csv')
    predictions = pd.read_csv(BASE / 'test_predictions.csv')
    samples = pd.read_csv(BASE / 'splits.csv')
    expected = len(config['seeds']) * len(config['widths'])
    assert len(results) <= expected
    if not args.partial:
        assert len(results) == expected
    with np.load(BASE / 'data.npz') as saved:
        data = {key: saved[key] for key in saved.files}
    assert data['x'].shape == (417, 3, 224, 224)
    clean = data['y']; noisy = data['y_noise']
    assert (noisy[data['train']] != clean[data['train']]).sum() == 29
    for split in ['validation', 'test']:
        assert np.array_equal(clean[data[split]], noisy[data[split]])
    assert samples.groupby('group_id').split.nunique().max() == 1
    duplicates = {}
    for i, tensor in enumerate(data['x']):
        digest = hashlib.sha256(tensor.tobytes()).hexdigest()
        duplicates.setdefault(digest, []).append(i)
    for indices in duplicates.values():
        assert samples.iloc[indices].split.nunique() == 1 and len(set(noisy[indices])) == 1
    for name, digest in json.loads((BASE / 'input_fingerprints.json').read_text()).items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == digest, name
    checks = []
    for row in results.itertuples():
        run = BASE / 'runs' / row.run
        assert row.config_sha256 == config_hash
        cp = torch.load(run / 'model.pt', map_location='cpu', weights_only=False)
        assert cp['config_sha256'] == config_hash
        assert cp['model_config']['image_size'] == 224
        model = build_model(config, row.width)
        model.load_state_dict(cp['state_dict']); model.eval()
        assert sum(p.numel() for p in model.parameters()) == row.parameters
        assert tuple(model.conv1.weight.shape) == (row.width, 3, 7, 7)
        assert model.conv1.stride == (2, 2) and model.maxpool.stride == 2
        with np.load(run / 'logits.npz') as logits:
            train, test = logits['train'], logits['test']
        assert train.shape == (293, 4) and test.shape == (62, 4)
        assert np.isfinite(train).all() and np.isfinite(test).all()
        assert abs(ce(train, noisy[data['train']]).mean() - row.train_observed_loss) < 1e-5
        assert abs(ce(test, clean[data['test']]).mean() - row.test_loss) < 1e-5
        guess = test.argmax(1)
        pp = predictions[predictions.run == row.run].set_index('sample_id').loc[samples.iloc[data['test']].sample_id]
        assert np.array_equal(pp.predicted_label.to_numpy(), guess)
        assert np.array_equal(pp.true_label.to_numpy(), clean[data['test']])
        assert abs((guess != clean[data['test']]).mean() - row.test_error) < 1e-6
        assert abs(torch.from_numpy(test).softmax(1).max(1).values.numpy() - pp.confidence.to_numpy()).max() < 1e-6
        checks.append(dict(run=row.run, loss_and_predictions_match=True, checkpoint_shape_correct=True))
        del model, cp
    FIG.mkdir(exist_ok=True)
    summary = results.groupby('width').agg(
        runs=('run', 'count'), test_error_mean=('test_error', 'mean'), test_error_std=('test_error', 'std'),
        test_loss_mean=('test_loss', 'mean'), test_loss_std=('test_loss', 'std'),
        train_observed_error_mean=('train_observed_error', 'mean'),
        train_observed_loss_mean=('train_observed_loss', 'mean'),
        fully_fitted_runs=('interpolation_reached', 'sum'))
    summary.to_csv(BASE / 'summary.csv', encoding='utf-8-sig')
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    metrics = ['test_error', 'test_loss', 'train_observed_error', 'train_observed_loss']
    titles = ['全部62张测试图：错误率', '全部62张测试图：交叉熵',
              '293个含噪训练标签：错误率', '293个含噪训练标签：交叉熵']
    for ax, metric, title in zip(axes.flat, metrics, titles):
        for seed, rr in results.groupby('seed'):
            rr = rr.sort_values('width')
            ax.plot(rr.width, rr[metric], 'o-', linewidth=1.4, markersize=4, label=f'seed={seed}')
        ax.plot(summary.index, summary[metric + '_mean'], '--', color='#222222', linewidth=2,
                label='已完成种子均值' if args.partial else '三种子均值')
        ax.set_xscale('log', base=2)
        ax.set_xticks(config['widths']); ax.set_xticklabels(config['widths'], fontsize=9)
        ax.set_xlabel('首阶段通道数 b'); ax.set_title(title)
        ax.set_ylabel('错误率' if metric.endswith('error') else '平均交叉熵（自然对数）')
        if metric.endswith('error'):
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        else:
            ax.set_ylim(bottom=0)
        ax.grid(alpha=.15); ax.legend(fontsize=9)
    fig.suptitle('224×224 · ResNet-18 · 29/293训练标签噪声' + (' · 进行中' if args.partial else ''))
    fig.tight_layout()
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(FIG / f'complete_curves.{ext}', dpi=180)
    plt.close(fig)
    variants = {}
    complete_seeds = [int(seed) for seed, group in results.groupby('seed') if len(group) == len(config['widths'])]
    if complete_seeds and len(config['widths']) >= 4:
        selected_results = results[results.seed.isin(complete_seeds)]
        selected_predictions = predictions[predictions.seed.isin(complete_seeds)].copy()
        selected_predictions['correct'] = selected_predictions.predicted_label == selected_predictions.true_label
        case = select_case(selected_results, selected_predictions, config['widths'])
        if case['rise'] > 0 and case['fall'] > 0:
            ids = choose_images(selected_predictions, case, config['widths'])
            metrics_by_width = selected_results[selected_results.seed == int(case['seed'])].set_index('width')
            for count, tag in [(9, '9models'), (7, '7models'), (12, 'all12models')]:
                columns = config['widths'] if count == 12 else choose_columns(config['widths'], case, count)
                variants[tag] = render_board(selected_predictions, metrics_by_width, ids, columns,
                                              case, data, samples, 'resnet_predictions_' + tag)
            selected = selected_predictions[(selected_predictions.seed == int(case['seed'])) & selected_predictions.sample_id.isin(ids)]
            selected.to_csv(BASE / 'selected_predictions.csv', index=False, encoding='utf-8-sig')
            (BASE / 'selection.json').write_text(json.dumps(dict(
                selected_case=case, sample_ids=ids, variants=variants,
                note='Post-hoc introduction illustration. Full curves, all widths, seeds and predictions retained.'),
                ensure_ascii=False, indent=2) + '\n')
    report = dict(status='partial' if args.partial else 'passed', completed_runs=len(results), expected_runs=expected,
                  test_predictions=len(predictions), input_size=224, training_flips=29,
                  held_out_labels_clean=True, known_groups_isolated=True, identical_input_leakage=False,
                  all_saved_losses_and_predictions_match=True, checks=checks)
    (BASE / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    note = '\n## 结果\n\n'
    note += f'已完成{len(results)}/{expected}组。完整曲线同时展示全部测试图的错误率、交叉熵及含噪训练标签的拟合程度。模型族变化或标签噪声降低均不保证出现双下降，结论以完整曲线为准。\n\n'
    note += '![完整曲线](figures/complete_curves.png)\n\n'
    if variants:
        note += '图片对照按结果筛选，仅作介绍示例；底部计数只针对选中6张图片。\n\n![9列版](figures/resnet_predictions_9models.png)\n\n![7列版](figures/resnet_predictions_7models.png)\n\n![完整12列](figures/resnet_predictions_all12models.png)\n'
    path = BASE / 'README.md'
    path.write_text(path.read_text().split('\n## 结果')[0].rstrip() + '\n' + note, encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

"""Recompute complete-test cross entropy from all 36 saved final checkpoints' logits."""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault('MPLCONFIGDIR', '/tmp/vit224-loss-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / 'vit224'
FIG = OUT / 'figures'


def losses(logits, labels):
    """Stable natural-log cross entropy, without rounding/clipping probabilities."""
    a = np.asarray(logits, dtype=np.float64)
    assert a.shape == (len(labels), 4) and np.isfinite(a).all()
    maximum = a.max(axis=1)
    log_normalizer = maximum + np.log(np.exp(a - maximum[:, None]).sum(axis=1))
    return log_normalizer - a[np.arange(len(labels)), labels]


def plot(rows, summary, widths, metrics, titles, stem):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for ax, metric, title in zip(axes, metrics, titles):
        for color, (seed, group) in zip(colors, rows.groupby('seed')):
            group = group.sort_values('width')
            ax.plot(group.width, group[metric], 'o-', color=color,
                    linewidth=1.4, markersize=4, label=f'seed={seed}')
        ax.plot(widths, summary.loc[widths, metric + '_mean'], '--',
                color='#222222', linewidth=2, label='三种子均值')
        ax.set_xscale('log', base=2)
        ax.set_xticks(widths)
        ax.set_xticklabels(widths, fontsize=9)
        ax.set_xlabel('嵌入宽度 d')
        ax.set_ylabel('错误率' if metric.endswith('error') else '平均交叉熵（自然对数）')
        if metric.endswith('error'):
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(title)
        ax.grid(alpha=.15)
        ax.legend(fontsize=10)
    fig.tight_layout()
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(FIG / f'{stem}.{extension}', dpi=180)
    plt.close(fig)


def main():
    for path in ['/System/Library/Fonts/Supplemental/Times New Roman.ttf',
                 '/Users/chan/Library/Fonts/simsun.ttc']:
        fontManager.addfont(path)
    plt.rcParams.update({'font.family': ['Times New Roman', 'SimSun'],
                         'svg.fonttype': 'path', 'pdf.fonttype': 42,
                         'axes.unicode_minus': False})
    original = pd.read_csv(OUT / 'results.csv')
    config = json.loads((OUT / 'config.json').read_text())
    widths = config['widths']
    assert len(original) == 36
    with np.load(OUT / 'data.npz') as data:
        y_test = data['y'][data['test']].astype(int)
        y_train = data['y_noise'][data['train']].astype(int)
        test_indices = data['test'].copy()
    samples = pd.read_csv(OUT / 'splits.csv').iloc[test_indices]
    predictions = pd.read_csv(OUT / 'test_predictions.csv')
    rows, image_rows, checks = [], [], []
    for row in original.itertuples():
        path = OUT / 'runs' / row.run / 'logits.npz'
        with np.load(path) as saved:
            assert set(saved.files) >= {'train', 'test'}
            test_logits = saved['test'].astype(np.float64)
            test = losses(test_logits, y_test)
            train = losses(saved['train'], y_train)
        predicted = test_logits.argmax(1)
        wrong = predicted != y_test
        run_predictions = predictions[predictions.run == row.run].set_index('sample_id').loc[samples.sample_id]
        assert np.array_equal(run_predictions.true_label, y_test)
        assert np.array_equal(run_predictions.predicted_label, predicted)
        differences = dict(test_loss=abs(float(test.mean()) - row.test_loss),
                           train_observed_loss=abs(float(train.mean()) - row.train_observed_loss))
        assert max(differences.values()) < 1e-5, (row.run, differences)
        assert abs(wrong.mean() - row.test_error) < 1e-6
        rows.append(dict(run=row.run, seed=row.seed, width=row.width,
                         parameters=row.parameters, test_loss=test.mean(),
                         test_error=wrong.mean(), train_observed_loss=train.mean(),
                         mean_wrong_loss=test[wrong].mean(),
                         mean_correct_loss=test[~wrong].mean(),
                         mean_wrong_confidence=run_predictions.confidence.to_numpy()[wrong].mean()))
        for sid, true, pred, ce in zip(samples.sample_id, y_test, predicted, test):
            image_rows.append(dict(run=row.run, seed=row.seed, width=row.width,
                                   sample_id=sid, true_label=true, predicted_label=pred,
                                   cross_entropy=ce))
        checks.append(dict(run=row.run, logits_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           absolute_difference_from_saved_metrics=differences))
    rows = pd.DataFrame(rows).sort_values(['seed', 'width'])
    image_rows = pd.DataFrame(image_rows)
    assert len(image_rows) == 2232
    summary = rows.groupby('width').agg(
        runs=('run', 'count'), test_loss_mean=('test_loss', 'mean'),
        test_loss_std=('test_loss', 'std'), test_error_mean=('test_error', 'mean'),
        test_error_std=('test_error', 'std'),
        train_observed_loss_mean=('train_observed_loss', 'mean'),
        train_observed_loss_std=('train_observed_loss', 'std'))
    assert (summary.runs == 3).all()
    rows.to_csv(OUT / 'loss_results.csv', index=False, encoding='utf-8-sig')
    image_rows.to_csv(OUT / 'test_image_losses.csv', index=False, encoding='utf-8-sig')
    summary.to_csv(OUT / 'loss_summary.csv', encoding='utf-8-sig')
    plot(rows, summary, widths, ['test_error', 'test_loss'],
         ['全部62张测试图：错误率', '全部62张测试图：交叉熵'], 'test_error_vs_loss')
    plot(rows, summary, widths, ['test_loss', 'train_observed_loss'],
         ['全部62张测试图的交叉熵', '对293个含噪训练标签的交叉熵'], 'complete_loss_curves')
    audit = dict(status='passed', runs=len(rows), test_images_per_run=62,
                 image_loss_rows=len(image_rows), retrained=False,
                 loss='mean negative log softmax probability of clean true test label; natural log; float64 logsumexp',
                 all_saved_test_train_losses_match=True, checks=checks)
    (OUT / 'loss_validation.json').write_text(json.dumps(audit, indent=2) + '\n')
    section = '''## 测试损失复核

从全部36组末轮保存的logits重新计算自然对数交叉熵（每组完整62张测试图），未重训或重选图片；训练、测试损失均与原始results.csv一致（容差1e-5）。

测试交叉熵没有比错误率呈现更清晰、稳定的双下降：三种子均值在d=2、24、48、96、192分别为1.328、2.984、2.459、2.660、2.790，缺少最初下降，后段也有波动。seed=42在d=14→24→96仍有1.803→3.542→2.099的局部升降，但这不足以说明跨种子稳定的双下降。原6张图片对照适合展示选定图片的预测反复，不应将其称为完整测试集双下降的验证。

错误率只看最终类别；交叉熵还惩罚真实类别概率低、特别是自信的误判。因此较宽模型可以错误率更低、交叉熵更高。

![同一批测试图的错误率与交叉熵](figures/test_error_vs_loss.png)

![完整测试与训练交叉熵](figures/complete_loss_curves.png)

逐组数据：loss_results.csv；三种子汇总：loss_summary.csv；逐图片损失：test_image_losses.csv；复核：loss_validation.json。

复现：在experiment目录执行 `python3 analyze_vit224_losses.py`。
'''
    readme = OUT / 'README.md'
    existing = readme.read_text().split('## 测试损失复核')[0].rstrip()
    readme.write_text(existing + '\n\n' + section, encoding='utf-8')
    print(summary.round(6).to_string())
    print(json.dumps({k: v for k, v in audit.items() if k != 'checks'}, ensure_ascii=False))


if __name__ == '__main__':
    main()

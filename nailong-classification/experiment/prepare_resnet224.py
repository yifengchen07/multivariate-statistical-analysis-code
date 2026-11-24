"""Prepare an independent, portable 224px ResNet sweep with nested 10% noise."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
SOURCE = BASE / 'vit224'
OUT = BASE / 'resnet224'


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'preparation.json').exists() and (OUT / 'input_fingerprints.json').exists():
        print('Existing ResNet configuration retained; run.py verifies frozen inputs.')
        return
    with np.load(SOURCE / 'data.npz') as saved:
        data = {key: saved[key] for key in saved.files}
    assert data['x'].shape == (417, 3, 224, 224)
    samples = pd.read_csv(SOURCE / 'splits.csv')
    clean = data['y'].copy()
    old_noisy = data['y_noise']
    noisy = clean.copy()
    rng = np.random.default_rng(71010)
    candidates = defaultdict(list)
    for index in data['train']:
        if old_noisy[index] != clean[index]:
            candidates[hashlib.sha256(data['x'][index].tobytes()).hexdigest()].append(int(index))
    # Largest-remainder allocation of round(293*.10)=29 flips across the four classes.
    targets = [7, 11, 7, 4]
    for label, target in enumerate(targets):
        groups = [group for group in candidates.values() if clean[group[0]] == label]
        rng.shuffle(groups)
        achievable = {0: []}
        for group in groups:
            for count, chosen in list(achievable.items())[::-1]:
                n = count + len(group)
                if n <= target and n not in achievable:
                    achievable[n] = chosen + group
        assert target in achievable
        selected = achievable[target]
        noisy[selected] = old_noisy[selected]
    train = data['train']
    assert int((noisy[train] != clean[train]).sum()) == 29
    assert np.all((noisy != clean) <= (old_noisy != clean))
    for split in ['validation', 'test']:
        assert np.array_equal(noisy[data[split]], clean[data[split]])
    duplicates = defaultdict(list)
    for i, image in enumerate(data['x']):
        duplicates[hashlib.sha256(image.tobytes()).hexdigest()].append(i)
    for group in duplicates.values():
        assert len(set(samples.iloc[group].split)) == 1
        assert len(set(noisy[group])) == 1
    data['y_noise'] = noisy
    np.savez_compressed(OUT / 'data.npz', **data)
    shutil.copy2(SOURCE / 'splits.csv', OUT / 'splits.csv')
    pd.DataFrame(dict(sample_id=samples.sample_id, split=samples.split,
                      clean_label=clean, noisy_label=noisy,
                      flipped=clean != noisy)).to_csv(OUT / 'noise_labels.csv', index=False, encoding='utf-8-sig')
    config = json.loads((SOURCE / 'config.json').read_text())
    for key in ['depth', 'heads', 'patch_size', 'mlp_ratio', 'comparison_note', 'dropout']:
        config.pop(key, None)
    config.update(architecture='WidthResNet18', blocks=[2, 2, 2, 2],
                  stem='7x7 stride2 padding3; 3x3 maxpool stride2 padding1',
                  widths=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64],
                  noise_rates=[.1], flipped_train_labels=29, actual_noise_rate=29 / 293,
                  noise_seed=71010, device='auto', batch_size=64, eval_batch_size=64,
                  noise_protocol='29/293 training labels changed; class-stratified subset of prior 59 flips, preserving their incorrect class. Identical 224px inputs share labels. Held-out labels clean.',
                  noise_assignment_note='10% allocation by class: 7,11,7,4. No original image or dataset label modified.',
                  training_preprocessing='Same 224px tensors and train-only normalization as ViT224; preserve aspect ratio and whole character with white padding.',
                  test_protocol='Exploratory 12-width x 3-seed grid fixed before training. Report all final checkpoints, test error and cross entropy. Introduction example selection, if any, disclosed.',
                  width_definition='Base channels b; stages have b,2b,4b,8b channels. b=64 equals torchvision ResNet18(num_classes=4). All widths use 224x224 input.',
                  pretrained=False)
    (OUT / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
    shutil.copy2(BASE / 'resnet224_models.py', OUT / 'resnet224_models.py')
    trainer = (BASE / 'train_sweep.py').read_text()
    trainer = trainer.replace('from models import build_model', 'from resnet224_models import build_model')
    trainer = trainer.replace("choices=['auto', 'mps', 'cpu']", "choices=['auto', 'mps', 'cpu', 'cuda']")
    trainer = trainer.replace('(BASE.parent / relative)', '(HERE / relative)')
    trainer = trainer.replace("('mps' if torch.backends.mps.is_available() else 'cpu')", "('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))")
    trainer = trainer.replace("    torch.set_num_threads(4)", "    if device == 'cuda' and not torch.cuda.is_available():\n        raise RuntimeError('CUDA unavailable.')\n    torch.backends.cudnn.benchmark = False\n    torch.backends.cudnn.deterministic = True\n    torch.set_num_threads(4)")
    trainer = trainer.replace("note='MPS reproducibility is seed controlled, not guaranteed bitwise across hardware/releases.'", "note='Seed controlled; reproducibility is not guaranteed bitwise across hardware/releases.'")
    (OUT / 'train.py').write_text(trainer)
    for filename in ['run.py', 'benchmark.py', 'report.py']:
        shutil.copy2(BASE / ('resnet224_' + filename), OUT / filename)
    helpers = (BASE / 'report_vit224.py').read_text().split('def cpu_verify(')[0]
    helpers = helpers.replace('from models import build_model\n', '')
    helpers = helpers.replace("OUT=BASE/'vit224'", 'OUT=BASE')
    helpers = helpers.replace("['/System/Library/Fonts/Supplemental/Times New Roman.ttf','/Users/chan/Library/Fonts/simsun.ttc']",
                              "[str(BASE/'fonts/Times New Roman.ttf'),str(BASE/'fonts/simsun.ttc')]")
    helpers = helpers.replace('模型更大，反而认错了？', '更宽的ResNet，会认错吗？')
    helpers = helpers.replace('224×224输入 · 16×16图块 · ViT · 约20%训练标签噪声',
                              '224×224输入 · ResNet-18 · 约10%训练标签噪声')
    helpers = helpers.replace("f'd = {d}'", "f'b = {d}'")
    helpers = helpers.replace('d为嵌入宽度；其余架构、数据和训练设置固定。',
                              'b为首阶段通道数；四阶段通道为b、2b、4b、8b。')
    (OUT / 'report_helpers.py').write_text(helpers)
    (OUT / 'fonts').mkdir(exist_ok=True)
    for source in ['/System/Library/Fonts/Supplemental/Times New Roman.ttf',
                   '/Users/chan/Library/Fonts/simsun.ttc']:
        shutil.copyfile(source, OUT / 'fonts' / Path(source).name)
    (OUT / 'requirements.txt').write_text('torch>=2.8\nnumpy\npandas\nmatplotlib\nPillow\n')
    (OUT / 'README.md').write_text('''# ResNet-18 / 224×224 / 10%训练标签噪声

沿用417张图片及训练293/验证62/测试62的划分。输入与ViT224完全一致：保持长宽比缩放、居中补白到224×224，训练集通道均值/标准差归一化。只在训练中加入29/293（9.90%）错标，验证和测试保持人工标签。29张为原59张错标的分层子集，不改原始图片或清单标签。

从零训练ResNet-18，标准7×7步长2卷积和最大池化；四阶段各2个BasicBlock，通道分别b/2b/4b/8b。b=64与torchvision标准ResNet18四分类结构及计算完全一致。扫描b=1,2,3,4,6,8,12,16,24,32,48,64，种子42/43/44。Adam，学习率0.001余弦降至0.00001，400轮，batch64，无增强、dropout或权重衰减，使用末轮权重。测试错误率和自然对数交叉熵都报告；模型替换与降低噪声同时进行，因此不能单独归因于其中一个因素。

Mac本机：`python3 run.py --device mps`。

NVIDIA：先在CUDA版PyTorch环境运行 `python -m pip install numpy pandas matplotlib Pillow`，再运行 `python run.py --device cuda`。自动选择设备：`python run.py`。不需要重新下载数据或预训练权重。

进度见training.log与每个runs子目录的history.csv；仅已完成的模型可续跑，中断模型从头训练。全流程结束生成pipeline_completed.json。run.py先核验输入，自动做当前设备的速度检查，再执行固定网格及报告。
''', encoding='utf-8')
    fingerprints = {filename: hashlib.sha256((OUT / filename).read_bytes()).hexdigest()
                    for filename in ['config.json', 'data.npz', 'splits.csv', 'noise_labels.csv',
                                     'train.py', 'resnet224_models.py', 'run.py', 'benchmark.py', 'report.py',
                                     'report_helpers.py', 'fonts/Times New Roman.ttf', 'fonts/simsun.ttc']}
    (OUT / 'input_fingerprints.json').write_text(json.dumps(fingerprints, indent=2) + '\n')
    preparation = dict(images=417, shape=list(data['x'].shape), training_flips=29,
                       noise_fraction=29/293, flipped_by_class=targets,
                       nested_subset_of_previous_59=True, held_out_labels_clean=True,
                       identical_inputs_have_consistent_noise=True, expected_runs=36,
                       source_data_sha256=hashlib.sha256((SOURCE/'data.npz').read_bytes()).hexdigest(),
                       training_labels_only_changed=True)
    (OUT / 'preparation.json').write_text(json.dumps(preparation, indent=2) + '\n')
    print(json.dumps(preparation), flush=True)


if __name__ == '__main__':
    main()

# 用 CIFAR-10 重新训练和评估

以下命令都从仓库根目录运行。训练与新模型的评估使用 `experiments/`，重绘随仓库提供的测量结果请见[首页](../README.md)。图号对应[论文的 arXiv 第一版](https://arxiv.org/abs/2203.08124v1)。

## 环境与数据

原任务的云端环境使用 Linux、Python 3.12、PyTorch 2.8.0、CUDA 12.8 和单张 RTX 5090。下例建立对应的 Python 环境。已有相同 PyTorch 环境的 GPU 实例可以直接安装 `requirements-train.txt`。其他加速平台可查 [PyTorch 安装表](https://pytorch.org/get-started/previous-versions/)。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-train.txt
python -m experiments.prepare_data --data datasets
```

TorchVision 会下载和校验 CIFAR-10。已有解压数据时，将 `--data` 指向包含 `cifar-10-batches-py/` 的目录，再运行 `python -m experiments.prepare_data --data datasets --verify-only`。后续的 `--data` 参数应指向同一目录。训练入口使用 POSIX 文件锁，可使用 Linux、macOS 或 Windows 的 WSL 环境。[数据集主页](https://www.cs.toronto.edu/~kriz/cifar.html)

## 图 1、2、3：每份模型训练 100 轮

| 图 | 模型 | 初始化种子 |
| --- | --- | --- |
| 1 | 全连接、ViT、WideResNet-28×10 | 0、1 |
| 2 | ViT、DenseNet-121、ResNet-18 | 0 |
| 3 | 全连接、MLP-Mixer、ViT、VGG-19、DenseNet-121、ResNet-18、WideResNet-28×10 | 0、1、2 |

三张图共用 21 份权重。第一条命令仅列出计划，第二条开始训练。只运行某张图时，可以把两条命令中的 `--figures 1 2 3` 换成该图号，随后的评估也只需调用该图号。

```bash
python -m experiments.arch.train --figures 1 2 3 --data datasets --work runs/architectures --plan
python -u -m experiments.arch.train --figures 1 2 3 --data datasets --work runs/architectures --device cuda
for fig in 1 2 3; do
  python -u -m experiments.regions.run --figure "$fig" --data datasets --work runs/architectures --device cuda || break
done
python -m experiments.regions.solid --data datasets --work runs/architectures --figures 1 2 3
```

`runs/architectures/figures/` 包含置信度着色的 PNG、逐点预测和置信度 NPZ，以及记录权重摘要、样本索引和网格设置的 JSON。`figures-solid/` 包含按每一网格点的预测类别着色的 PNG。每个可视化平面使用 500×500 网格。图 1 和图 3 取同一组训练集索引 30、72、42。图 2 的三列各取三个训练样本。同一列的三张图像分别进行独立的像素置换，RGB 分量保持在一起；各模型使用相同的三张置换结果。

图 2 的九张原始类别网格已随仓库提供。准备好 CIFAR-10 后，可在 CPU 上直接重建含像素置换缩略图的纯色版本。

```bash
python -m experiments.regions.solid --figures 2 --data datasets --measurements architecture-results/figure2 --output runs/figure2-replot
```

每个模型均用 CIFAR-10 训练 100 轮，SGD 动量 0.9，权重衰减 0.0005，更新批量 128。重型网络使用梯度累积，DenseNet-121、ViT、MLP-Mixer 和 WideResNet 的分片为 32，VGG-19 为 64；含批归一化层的模型按分片估计训练统计量。初始学习率为 0.05，ViT 与 MLP-Mixer 改用 0.02，VGG-19 改用 0.005。每个 epoch 后保存断点；使用原命令可从最近完成的一轮继续。归档图中 ResNet-18 的种子 0、1 采用了先前训练的权重，其中一部分轮次曾移至另一设备续训。全新环境会从头训练，浮点环境、设备和训练过程可使边界细节与归档图产生差异。WideResNet-28×10 是本复现对论文宽度的显式选择。

## 图 6、7、10、12：每份模型训练 500 轮

默认配置名为 `measured`。它使用 11 个宽度 `1,2,4,6,7,8,10,12,20,40,64`、0% 和 20% 两种标签噪声、初始化种子 1，共 22 份 PreActResNet18(k) 权重。每个模型训练 500 轮，采用 Adam、固定学习率 0.0001、批量 128，训练集包含全部 50,000 张图像。使用同一 `--out` 可在图之间共享训练结果。先检查计划，再运行所需阶段。

```bash
python -m experiments.width.run_width --stage plan
python -u -m experiments.width.run_width --stage all --profile measured --data datasets --out runs/width --device cuda
```

例如，只复现图 7 时运行以下命令。它会选取六个展示宽度，共 12 份模型，再生成每个宽度的四个面板。

```bash
python -u -m experiments.width.run_width --figures 7 --stage all --data datasets --out runs/width --device cuda
```

| 图 | 输出和采样 |
| --- | --- |
| 6 | 11 个宽度，两种噪声。对 10,000 张测试图像计算分类错误率。 |
| 7 | 6 个宽度（1、4、7、10、20、64）。两组训练三元组各使用 500×500 网格，两种训练噪声均评估。 |
| 10 | 每个宽度和噪声条件下，每组采样 1,000 个训练三元组，在每个 50×50 平面统计类别区域的八邻域连通分量数，再按组取均值。 |
| 12 | 每组抽样 5,000 次，每次使用图像差向量搜索预测类别变化，目标为 10 个有效方向。每张图像对有效方向取均值，再对组内有效图像取中位数。少于 5 个有效方向的图像记为空值，逐样本结果和有效样本数保存在评估 JSON 中。 |

输出位于 `runs/width/measured/`。其中 `models/` 包含各模型权重与测试指标，`measures/` 包含每个宽度和噪声条件的原始测量，`figures/` 包含绘制的 PNG 和对应 JSON。图 7 的训练入口生成含置信度的图；归档的中文版 PDF 按预测类别着纯色。可把 `--stage all` 改为 `train`、`evaluate` 或 `plot`，分别训练、评估或用已保存的测量绘图。相同配置中断后可重复命令。训练通常每 10 轮存储一个断点，图 10、12 每组每 10 个观测写一次进度。

`--profile paper` 使用 4,000 轮和适用图的完整宽度集合，输出至独立目录。随仓库发布的这四张图对应 `measured`。图 12 的搜索使用作者公开代码中的非均匀距离网格，图 10 的同类三元组在十个类别间均衡抽样；论文用语、公开代码和这些实现选择的差别应在比较曲线时一并考虑。

## 附加结果：ResNet-18 双种子的一致率

图 2 与图 3 中的 ResNet-18 种子 0，以及图 3 中的种子 1，使用历史双种子实验的最终权重。两份 SHA-256 摘要和原设备信息记录在归档的 `agreement.json` 中；跨架构训练入口也校验这两份已知权重。用当前设备重新跑完 100 轮会得到该次运行自己的权重摘要。

在已训练的两份 ResNet-18 上，下面的命令会评估 10,000 张测试图像、一个训练三元组张成的 500×500 平面，以及 500 个不同类别测试三元组张成的 50×50 平面。结果写入新运行目录中的 `agreement.json`，各测试平面含三元组索引、逐点一致数与总点数。该命令使用与图 1–3 共用的模型定义、标准化和几何函数。

```bash
python -u -m experiments.regions.agreement --data datasets --device cuda \
  --checkpoints runs/architectures/checkpoints/resnet/seed0.pth runs/architectures/checkpoints/resnet/seed1.pth \
  --out runs/pair-agreement --cache runs/pair-agreement/cache
```

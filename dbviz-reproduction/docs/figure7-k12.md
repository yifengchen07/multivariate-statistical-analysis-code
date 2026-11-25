# 用本次双下降曲线的峰值宽度绘制图 7

论文第一版在宽度 10 观察到带噪测试错误率的中段峰值。我们训练 500 轮、固定种子 1 后，在 20% 标签噪声的曲线上测得宽度 6 的测试错误率为 22.72%，宽度 10 为 28.82%，宽度 12 为 31.04%，宽度 20 为 28.31%。因此，围绕本次实测峰值的版本展示宽度 **1、4、7、12、20、64**。这些错误率来自[本仓库图 6 数据](../data/curves.csv)，关于原实验峰值的讨论见[论文第一版第 6 页](https://arxiv.org/pdf/2203.08124v1#page=6)。

[新版图 7](../figures/fig07-k12.pdf) · [四页合订本](../figures/figures-500epochs-k12-zh.pdf) · [宽度 12 的四个原始面板](../data/figure7-k12-500-panels.zip)

## 权重与实测记录

第四列取自原 AutoDL 实例导出的两份宽度 12、初始化种子 1、第 500 轮权重。本地在 CPU 上用同一套 CIFAR-10 测试图像重新评估，再计算两个三元组上的决策区域。

| 标签噪声 | 图 6 的云端正确数 | 本地 CPU 复核正确数 | 图 6 记录的测试错误率 |
| --- | ---: | ---: | ---: |
| 0% | 8,575 / 10,000 | 8,575 / 10,000 | 14.25% |
| 20% | 6,896 / 10,000 | 6,895 / 10,000 | 31.04% |

20% 噪声模型在本地测得的测试错误率为 31.05%，与云端记录相差 0.01 个百分点。跨平台数值精度可能影响接近决策边界的预测，差异的具体原因尚未确定。图 6 使用原始云端测量值，图 7 的宽度 12 面板使用这次本地前向推理的逐点预测。[权重与原始数据校验记录](../data/figure7-k12-receipt.json)

## 在原 AutoDL 实例上生成

将运行包 `figure7-k12-autodl.zip` 上传到原实例的 `/root/autodl-tmp/dbviz-stages/`，在 JupyterLab 终端执行以下命令。

```sh
cd /root/autodl-tmp/dbviz-stages &&
unzip -o figure7-k12-autodl.zip &&
python -u figure7-k12-kit/scripts/run_figure7_k12.py
```

完整仓库的同一入口为 `python scripts/run_figure7_k12.py --project /root/autodl-tmp/dbviz-stages`。它查找两份权重，在整个 CIFAR-10 测试集上核验，并为各模型分别计算异类三元组和同类三元组的 500 × 500 预测网格。其余五列直接读取归档网格。

成功后在项目根目录下载 `figure7-k12-results.zip`。其中 `fig07-k12.pdf` 是单图，`figures-500epochs-k12-zh.pdf` 是换入该图的四页合订本，另附 PNG、可重绘的六列网格及新列的原始测量。脚本只做推理和绘图，结束后实例保持开机。

## 在已导出的面板上重绘

仓库已提供上述面板。在本仓库根目录执行以下命令即可重绘。

```sh
python scripts/figure7_peak.py --panels data/figure7-k12-500-panels.zip --curves data/curves.csv --out build/fig7-k12
```

报告图注示例为“图 7　不同网络宽度下的决策区域。每个模型训练 500 轮，选取宽度 12 展示本次 20% 标签噪声测试错误率曲线的中段峰值。” 查阅原论文选取的宽度时，可使用[归档的宽度 10 版本](../figures/fig07.pdf)。

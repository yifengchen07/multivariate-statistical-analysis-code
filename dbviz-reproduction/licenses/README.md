# 第三方模型代码

`experiments/arch/source/` 的七个模型源文件以及 `experiments/width/resnet18k.py` 逐字复制自 Somepalli 等人的 dbViz，提交为 `aeea84c42a32efafebbb29c3d944204aa74963cf`。上游的文件头、引用以及完整 Apache 2.0 许可随本仓库保留。训练、缓存、评估、续跑与绘图入口由本复现项目整理。[dbViz 固定提交](https://github.com/somepago/dbViz/tree/aeea84c42a32efafebbb29c3d944204aa74963cf) · [dbViz 许可全文](dbViz-Apache-2.0.txt)

| 本仓库位置 | dbViz 中的文件 |
| --- | --- |
| `experiments/arch/source/{fcnet,mlpmixer,vit,vgg,densenet,resnet,wideresnet}.py` | `models/` 下的同名文件 |
| `experiments/width/resnet18k.py` | `double-descent/models/resnet18k.py` |

dbViz 的致谢或文件头另列出以下来源，随附相应的[MIT 版权及许可声明](upstream-MIT.txt)。

- CIFAR 网络实现及预激活残差结构。Copyright (c) 2017 liukuang。[pytorch-cifar](https://github.com/kuangliu/pytorch-cifar) · [许可](https://github.com/kuangliu/pytorch-cifar/blob/master/LICENSE)
- 视觉 Transformer 实现。Copyright (c) 2020 Phil Wang。[vit-pytorch](https://github.com/lucidrains/vit-pytorch) · [许可](https://github.com/lucidrains/vit-pytorch/blob/main/LICENSE)
- MLP-Mixer 实现。Copyright (c) 2021 Phil Wang。[mlp-mixer-pytorch](https://github.com/lucidrains/mlp-mixer-pytorch) · [许可](https://github.com/lucidrains/mlp-mixer-pytorch/blob/main/LICENSE)

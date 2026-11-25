# 来源与许可

## 论文与图号

Gowthami Somepalli、Liam Fowl、Arpit Bansal、Ping Yeh-Chiang、Yehuda Dar、Richard Baraniuk、Micah Goldblum 和 Tom Goldstein 于 CVPR 2022 发表 *Can Neural Nets Learn the Same Model Twice? Investigating Reproducibility and Double Descent from the Decision Boundary Perspective*。本项目以 arXiv 第一版的图号组织实验。引用原研究时请引用作者论文。[arXiv 第一版](https://arxiv.org/abs/2203.08124v1) · [CVPR 论文页面及 BibTeX](https://openaccess.thecvf.com/content/CVPR2022/html/Somepalli_Can_Neural_Nets_Learn_the_Same_Model_Twice_Investigating_Reproducibility_CVPR_2022_paper.html)

## 代码与字体

- 项目新增的训练、评估、校验和绘图代码采用 [Apache-2.0](../LICENSE)。实验模型基于作者的 [dbViz 固定版本](https://github.com/somepago/dbViz/tree/aeea84c42a32efafebbb29c3d944204aa74963cf)。随库保留了[上游许可与署名](../licenses/)。
- 中文 PDF 的默认绘图字体是 Noto Sans SC Regular 的轻量字形子集，已重命名为 Plot Labels SC Regular。字体采用 SIL Open Font License 1.1。[字体来源与修改](../assets/fonts/README.md) · [字体许可](../assets/fonts/OFL.txt)

## 数据与图像

训练数据由 TorchVision 按需获取，CIFAR-10 的发布者为 Alex Krizhevsky、Vinod Nair 和 Geoffrey Hinton。数据集原始图片、训练权重与云主机环境包均不随仓库分发。[CIFAR-10 官网](https://www.cs.toronto.edu/~kriz/cifar.html)

`data/` 中的宽度实验结果、`architecture-results/` 的架构记录和 `figures/` 的复现图源自本次复现。`comparisons/` 同时包含从原论文读取的近似数值，具体来源按行记录。根目录 Apache-2.0 的授权范围为代码；第三方论文、字体和数据集依各自的来源条款，仓库内的实验数据与图表未另选开放许可证。

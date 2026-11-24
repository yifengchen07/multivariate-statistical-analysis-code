# 梗角色四分类数据集

当前版本以用户人工清洗后保留的图片为准，已于2026-11-19统一编号，共417张。此次只重命名和建立清单，全部图片内容及原始编码保持不变；未重新筛选、删除或恢复任何图片。

| 类别编号 | 目录 | 角色 | 数量 | 文件名 |
|---:|---|---|---:|---|
| 0 | `dataset/lulu` | 噜噜 | 96 | `lulu_0001.jpg` 起 |
| 1 | `dataset/nailong` | 奶龙 | 163 | `nailong_0001.jpg` 起，保留原PNG扩展名 |
| 2 | `dataset/naiwa` | 奶蛙 | 98 | `naiwa_0001.jpg` 起 |
| 3 | `dataset/niulai` | 牛来 | 60 | `niulai_0001.jpg` 起 |
| | 合计 | | 417 | |

每类从0001起连续编号，类别编号与 torchvision ImageFolder 默认的目录字母排序一致。奶龙包含159张JPEG和4张PNG，未为统一扩展名重新压缩图片。

## 实验入口

`dataset/` 下只有四个类别目录，图片可以按ImageFolder结构直接加载。全部样本信息位于 `dataset_manifest.csv`，包含样本ID、类别编号、相对路径、原路径、SHA-256、尺寸、格式及已有来源信息。该CSV使用UTF-8 BOM编码。

- `dataset_info.json`：当前类别映射、数量和版本。
- `rename_mapping.csv`：每张图片的新旧路径与校验值，可以追溯或恢复原文件名。
- `dataset_validation.json`：编号、文件完整性与重命名前后校验结果。
- `data_collection/manifest.csv`：此前网络采集图片的来源，路径已同步为当前编号。
- `data_collection/manifest_before_numbering.csv`：编号前的采集来源备份。

现已固定训练集293张、验证集62张、测试集62张，分层分组记录见 `experiment/vit224/splits.csv`，ResNet重跑沿用完全相同的划分。已知视频帧组、精确重复及识别出的近重复图片保持同组；未知同源关系可能遗漏。原始图像目录没有移动。已有相似分组以及旧文件名中的原视频帧分组信息仍保存在完整清单中。

## 历史记录

`data_collection/` 中的原始下载文件、候选记录、筛选报告及 `COLLECTION_README.md` 是人工清洗前的历史采集记录，其中的数量不能代表当前数据集。当前图片和 `dataset_manifest.csv` 是后续实验依据。旧 `prepare.py` 已阻止再次导出采集图片，避免恢复已清洗删除的文件或破坏编号。

## 宽度与标签噪声实验

当前实验统一使用224×224输入：等比缩放、居中补白，归一化统计只取训练集。

- `experiment/vit224/`：已完成的12宽度×3种子ViT，约20%训练标签噪声，含全部错误率、交叉熵、权重及筛选的介绍图片。完整曲线尚未呈现清晰、稳定的双下降。
- `experiment/resnet224/`：新ResNet-18重跑，12个首阶段通道数×3种子，29/293（9.90%）训练错标；验证和测试标签干净，400轮末轮权重。该目录内的 `README.md` 说明运行和结果入口。
- `experiment/run_resnet224.command`：Mac原生终端使用MPS启动入口。
- `experiment/cleanup_32px.json`：按要求删除旧32×32权重、缓存、预测、图表及专用脚本的逐文件记录；417张原始图片和已完成的224px结果保留。

ResNet训练进度以 `experiment/resnet224/training.log` 为准；`pipeline_completed.json` 仅在完整36组训练和结果核验结束后生成。`resnet224_smoke/` 是独立的2轮流程测试，不与正式结果混用。

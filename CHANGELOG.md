# 版本记录

## 2026-11-07 开源

- 公开代码、完整训练数据（训练池 242,921 题、开发集、留出测试）与评测记录，见 [数据说明](docs/DATA.md) 和 [评测](docs/EVALUATION.md)。
- 在 1,557 题留出测试上评测一次：84.05%（原始 4B 75.50%）；该集合可能与训练池重叠，不是盲测。
- 新增 `configs/jev/release/train.json`，可直接用公开数据复现训练。

## LoRA v1.1.0（当前版本）

- 全层 LoRA，第 5,500 步；复现配置 `configs/jev/release/train.json`。
- 权重 SHA-256：`88c64b00d23201118a9b7b4bad192aba6a5fcb5de4e65f00542529554425b4b6`。
- 外部基准 1,050/1,514（69.35%，>52 候选分块淘汰）；发布包直接打分 1,028/1,514（67.90%）。训练 26 候选，发布包 52 候选 / 8,192 token。
- [固定下载版本](https://huggingface.co/JuneYao/medical-jev-4b/tree/v1.1.0)。

详细成绩及统一结果字段见 [结果总览](docs/results/README.md)。

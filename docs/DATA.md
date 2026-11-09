# 训练数据

[返回首页](../README.md)

完整数据公开在 [Hugging Face 数据集](https://huggingface.co/datasets/JuneYao/medical-jev-4b-data)，数据卡列出每类任务和每个来源的题数与声明许可。

| 文件 | 题数 | 用途 |
| --- | ---: | --- |
| `train.jsonl` | 242,921 | 训练池（77 个来源，CMB 195,784 题） |
| `development.jsonl` | 1,566 | 选模开发集 |
| `development_v1_heldout.jsonl` | 1,151 | 开发集中 v1 未训练过的部分 |
| `test.jsonl` | 1,557 | 留出测试 |

## 数据从哪里来

1. **公开题源**：医学考试题库（CMB、MedQA、MedMCQA、CMExam、KorMedMCQA、各国执业考试等）、公开数据集和规则 / 指南映射，统一成选择题格式，并按题干映射到 27 类决策任务。
2. **补充题源**：按来源核查、去重后加入的题（分期、宣教、康复等），文件清单与 SHA-256 见 `configs/jev/data_sources.json`。
3. **来源约束病例（R60）**：约 2,150 题由 AI 依据公开指南构造、AI 审核，`label_status` 为 `synthetic_source_grounded_ai_reviewed`。

直接使用公开数据集即可训练，不需要重新构建。

## 隔离与局限

- 训练池、开发集、测试集与外部基准按来源组、精确输入、归一化文本三项隔离。
- v1 当时的原始训练文件已丢失；公开的是重建版本，与 v1 训练池各任务规模基本一致，但不逐条相同。
- 标签未经临床专家逐条审核。各来源保留上游许可，代码许可不授予数据使用权；部分来源为考试原题或仅限非商业使用，使用前请自行核对。

# 训练数据

[返回首页](../README.md)

完整数据公开在 [Hugging Face 数据集](https://huggingface.co/datasets/JuneYao/medical-jev-4b-data)，数据卡列出每类任务和每个来源的题数与声明许可。

| 文件 | 题数 | 用途 |
| --- | ---: | --- |
| `train.jsonl` | 242,921 | 训练池（77 个来源，CMB 195,784 题） |
| `development.jsonl` | 1,566 | 选模开发集 |
| `test.jsonl` | 1,557 | 留出测试 |

## 数据从哪里来

1. **公开题源**：医学考试题库（CMB、MedQA、MedMCQA、CMExam、KorMedMCQA、各国执业考试等）、公开数据集和规则 / 指南映射，统一成选择题格式，并按题干映射到 27 类决策任务。
2. **补充题源**：按来源核查、去重后加入的题（分期、宣教、康复等），文件清单与 SHA-256 见 `configs/jev/data_sources.json`。
3. **来源约束病例（R60）**：约 2,150 题由 AI 依据公开指南构造、AI 审核，`label_status` 为 `synthetic_source_grounded_ai_reviewed`。

直接使用公开数据集即可训练，不需要重新构建。

## 隔离与局限

- 公开数据是原始切分。训练前运行 `python scripts/build_clean_data.py .` 生成 `work/data/clean`：从训练池删除与开发集、留出测试、外部基准题干相同或高度相似（5-gram Jaccard ≥ 0.8）的题，以及答案互相矛盾的题；从开发集删除与留出测试重复的题。v1.2 共删除训练题 307 条、开发题 3 条，结果可逐字节复现。
- `jev split` 将同 `group_id` 或同完整输入哈希的记录连接成组件后整组件划分，并在写出前检查集合间的交集。
- 去重不能排除语义改写或底座预训练时见过的题。
- 任务归类多数由规则生成，约 96% 仍为待复核状态，部分题可能归错任务。分诊题不进入训练。
- 标签未经临床专家逐条审核。各来源保留上游许可，代码许可不授予数据使用权；部分来源为考试原题或仅限非商业使用，使用前请自行核对。

## 独立候选清单

```bash
python scripts/audit_duplicates.py --train work/data/train.jsonl \
  --evaluation work/data/test.jsonl --output work/audits/test-overlap.json
```

工具只筛查有独立 `state.clinical_question`、有效选项和答案的文字选择题，不覆盖所有结构化任务。版本 `clinical-question-sequence/1` 使用 Unicode NFKC、空白折叠和首尾去空白，保留大小写、标点、数字、单位与否定词；要求全部规范化选项文本的多重集合及正确答案文本相同，再以 `difflib.SequenceMatcher(..., autojunk=False)` 比较题干。报告列出 ≥90% 和 ≥95% 两档数量，每道评测题保留分数最高的训练候选，分数相同时取训练 ID 最小者；同时记录输入文件 SHA-256、完整题干、选项及算法参数。

**相似度候选必须人工复核。** 它可能只差剂量、小数点或否定词，选项重排还可能改变“以上皆是”等位置依赖题的含义；额外患者状态字段也不在题干分数中。清单不参与 `split` / `isolate`，不自动判定同题或实际训练暴露，也不修改输入文件。需要人工确认并处理的题组应先形成可追溯的新数据版本，再重新划分。

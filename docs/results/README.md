# 模型结果

当前版本 LoRA v1.2.0 的结果，文字说明见[评测文档](../EVALUATION.md)。

| 评测 | 原始 4B | v1.2 |
| --- | ---: | ---: |
| 外部基准，混合策略 | 925 / 61.10% | **1,038 / 68.56%** |
| 外部基准，默认接口 | 907 / 59.91% | **1,011 / 66.78%** |
| 留出测试 | 1,175 / 75.47% | **1,260 / 80.92%** |

| 文件 | 内容 |
| --- | --- |
| [summary.json](v1.2/summary.json) | 权重哈希、训练信息、数据清洗、总体与逐任务成绩、配对比较、分诊分析、其他领域结果 |
| [per-question-benchmark.jsonl](v1.2/per-question-benchmark.jsonl) | 外部基准逐题答案，不含题目原文 |
| [per-question-heldout.jsonl](v1.2/per-question-heldout.jsonl) | 留出测试逐题答案，不含题目原文 |
| [input-limit-check.json](v1.2/input-limit-check.json) | 长输入和 52 选项边界检查 |

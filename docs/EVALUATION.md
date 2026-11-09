# 评测

[返回首页](../README.md)

## 评测集

| 评测集 | 题数 | 用途 | 状态 |
| --- | ---: | --- | --- |
| 外部基准 `medical_decision_v1` | 1,514 | 跨来源能力、与其他模型对比 | 开发中反复使用，**不是盲测** |
| 训练分布内留出测试 | 1,557 | 与训练同分布的留出题，每个权重只评测一次 | v1 已评测；可能与 v1 训练池重叠 |
| 旧留出测试（v1 发布时） | 1,620 | 同上 | v1 已评测一次；文件随原基线丢失 |
| CARE-Bench validation 分诊 | 181 | 分诊结论的独立核对 | 仅评测，不训练（CC BY-NC 4.0） |
| 开发集 | 1,566 | 训练中选模 | 每类 60 题，分诊 6 题 |

所有评测集与训练池按**来源组、精确输入哈希、归一化文本**三项隔离；交集均为 0。这些检查不能排除语义近重复或预训练暴露。

外部基准来自 [jev-healthcare-lab](https://github.com/JuneYaooo/jev-healthcare-lab)，覆盖 27 类有题任务；其中 230 题（CARE-Bench、TCM-SD、PMR-Synth）为非商业许可。

## 指标

- **准确率**：答对题数 / 全部题数。无效回答、请求失败、超出接口上限都计错，不从分母中扣除。
- **配对比较**：同一批题逐题对照，按来源组 bootstrap（5,000 次）给出 95% 区间；区间跨 0 视为无显著差异。
- **ECE**：10 个等宽置信度箱，衡量概率是否可信；越小越好。v1 概率未经校准。

## 推理口径

| 口径 | ≤52 候选 | >52 候选（100 道中医证型题） |
| --- | --- | --- |
| 混合策略（本文默认） | 一次前向直接打分 | 分块淘汰：每块 26 个、保留前 2、3 种分块取平均 |
| 发布接口 | 直接打分 | 不支持，计错 |

| 模型 | 混合策略 | 发布接口 |
| --- | ---: | ---: |
| v1 | 1,050（69.35%） | 1,028（67.90%） |
| 原始 Qwen3.5-4B | 925（61.10%） | 907（59.91%） |

v1 相对原始 4B：+8.26 个百分点，95% 区间 [5.74, 10.88]；修复 264 题，答坏 139 题。ECE 0.132。

## 与其他模型对比

| 模型 | 答对 / 1,514 | 准确率 |
| --- | ---: | ---: |
| JEV 1.13（线上服务存档） | 1,092 | 72.13% |
| DeepSeek V4.1 Flash | 1,076 | 71.07% |
| **v1** | **1,050** | **69.35%** |
| DeepSeek V4 Pro | 1,041 | 68.76% |
| Qwen3.8-27B | 1,008 | 66.58% |
| 原始 Qwen3.5-4B | 925 | 61.10% |
| Qwen3.5-9B | 826 | 54.56% |

参考模型通过各自 API 以生成方式作答（关闭思考、temperature 0、输出严格校验为选项键），由 jev-healthcare-lab 在 `results/clinical-v1` 中记录；本模型为本地选项打分。作答方式、服务条件与运行时间不同，仅供参考。Qwen3.5-9B 有 16 题因限流失败计错。

## 逐任务成绩

准确率（%）。题数少于 20 的任务个别题即可造成大幅波动。

| 任务 | 题数 | 原始 4B | **v1** | DeepSeek Flash | DeepSeek Pro | Qwen3.8-27B | JEV 1.13 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 病例并发症 case_complication | 100 | 77 | **86** | 89 | 89 | 84 | 93 |
| 病例病因 case_etiology | 100 | 80 | **85** | 84 | 84 | 81 | 84 |
| 疾病分期 clinical_grade | 100 | 42 | **54** | 50 | 53 | 43 | 62 |
| 诊断选择 diagnosis_choice | 100 | 82 | **89** | 91 | 91 | 86 | 89 |
| 检查选择 examination_choice | 100 | 81 | **86** | 81 | 74 | 74 | 88 |
| 用药选择 medication_choice | 100 | 67 | **82** | 80 | 72 | 73 | 83 |
| 药物基因组建议 pgx_guideline_recommendation | 100 | 39 | **60** | 54 | 51 | 54 | 55 |
| 模拟病例诊断 synthetic_diagnosis | 100 | 55 | **60** | 77 | 71 | 68 | 70 |
| 中医病机 tcm_pathogenesis_candidate | 100 | 64 | **77** | 76 | 79 | 75 | 67 |
| 中医证型 tcm_syndrome | 100 | 18 | **24** | 37 | 30 | 31 | 38 |
| 治疗选择 treatment_choice | 100 | 67 | **77** | 82 | 74 | 72 | 78 |
| 分诊 triage_urgency | 100 | 55 | **40** | 61 | 57 | 65 | 63 |
| 临床试验转介 trial_referral | 100 | 55 | **62** | 60 | 58 | 58 | 53 |
| 患者禁忌 patient_contraindication | 54 | 67 | **80** | 74 | 67 | 65 | 80 |
| 护理优先级 nursing_priority | 42 | 76 | **86** | 74 | 81 | 83 | 86 |
| 患者消息优先级 message_urgency_pair | 30 | 77 | **70** | 50 | 63 | 50 | 63 |
| 补液方案 fluid_plan | 16 | 31 | **62** | 56 | 50 | 31 | 50 |
| 预防措施 prevention_choice | 13 | 54 | **92** | 85 | 85 | 92 | 85 |
| 营养方案 nutrition_plan | 11 | 64 | **64** | 73 | 73 | 45 | 73 |
| 健康教育 patient_counseling | 11 | 82 | **82** | 91 | 91 | 91 | 100 |
| 体位选择 positioning_choice | 10 | 50 | **60** | 80 | 90 | 70 | 90 |
| 随访计划 followup_plan | 6 | 83 | **100** | 67 | 83 | 83 | 83 |
| 感染防控 infection_precaution | 5 | 80 | **100** | 100 | 100 | 100 | 100 |
| 监测目标 monitoring_target | 5 | 80 | **80** | 100 | 80 | 60 | 80 |
| 给药途径 administration_route_choice | 4 | 50 | **100** | 75 | 75 | 50 | 75 |
| 康复方案 rehabilitation_choice | 4 | 50 | **75** | 50 | 75 | 50 | 100 |
| 知情同意 consent_action | 3 | 67 | **67** | 100 | 100 | 100 | 100 |

逐题记录：[results/v1/bench-mixed-predictions.jsonl](results/v1/bench-mixed-predictions.jsonl)；按发布接口的旧记录见 [results/v1/per-question.jsonl](results/v1/per-question.jsonl)。

## 分诊核对

基准分诊只有 100 题、且 v1 90% 预测同一类，另用 CARE-Bench 官方 validation 部分核对（181 题、93 个病例，与基准所用病例不重叠；转换方式可逐题复现基准的 100 题，见 `scripts/build_care_validation.py`）。

| 模型 | 基准分诊 / 100 | CARE validation / 181 | 最大类占比（validation） |
| --- | ---: | ---: | ---: |
| 原始 4B | 55 | 80（44.2%） | 46% |
| v1 | 40 | 94（51.9%） | 87% |

两套题结论不一致：基准上原始 4B 更好，validation 上 v1 更好，但 v1 都集中预测“非紧急就医”。分诊不应使用本模型。

## 训练分布内留出测试

2026-11-07 用 v1 和原始底座各评测一次（混合策略，输入 SHA-256 `ebf69b39…14b9e7`），之后随数据集公开，不再作为封存集。

| 模型 | 答对 / 1,557 | 准确率 | 宏平均 |
| --- | ---: | ---: | ---: |
| v1 | 1,307 | 84.05% | 84.75% |
| 原始 4B | 1,174 | 75.50% | 76.47% |

v1 相对原始 4B +8.55 个百分点，区间 [6.53, 10.42]。该集合切自重建的数据，v1 当时的训练池包含其上游来源，v1 抽样训练的 44,847 条具体是哪些已无记录，因此部分题可能被 v1 训练过。提升幅度与外部基准（+8.26）接近，但不能据此证明没有重叠。记录：[results/v1/heldout-1557.json](results/v1/heldout-1557.json)。

## 复现

```bash
jev eval --config configs/jev/release/train.json \
  --checkpoint <medical-jev-4b>/adapter.safetensors \
  --input work/benchmark/medical_decision_v1.jsonl \
  --max-options 52 --max-tokens 8192 --tournament-above 52 --batch-size 1 --output runs/bench-v1
```

不加 `--checkpoint` 即评测原始底座。基准文件由 `jev import-benchmark --benchmark <jev-healthcare-lab>/benchmarks/medical_decision_v1 --output work/benchmark/medical_decision_v1.jsonl` 生成。配对比较：`python scripts/compare.py --base runs/bench-readout --candidate runs/bench-v1`。

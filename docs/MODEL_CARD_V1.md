---
license: apache-2.0
language:
- zh
- en
base_model: Qwen/Qwen3.5-4B
library_name: pytorch
---

# medical-jev-4b

基于 Qwen3.5-4B 的医疗选择题LoRA模型，从给定候选中选出一项。用于教学题库、模型评测和离线研究，不用于临床决策。

## 成绩与限制

外部基准共 1,514 题，默认接口准确率 **66.78%**；接口可回答的 1,414 题准确率为 71.50%。开启超过 52 选项的分块淘汰后为 68.56%。1,557 道留出题上为 80.92%，比原始 4B 高 5.46 个百分点。外部基准开发中反复使用，发布前也分析过留出题错题，二者都不是盲测。

训练覆盖 2–26 个候选，发布接口允许 52 个候选 / 8,192 token，但 27–52 候选的医学准确率证据有限，默认接口不支持 148 候选中医证型题。不得用于分诊或任何临床决策：分诊 51/100，低于原始 4B 的 55/100，且第一轮对话很少追问。

概率未经校准，不能作为临床风险或复核阈值。构造示例答案由题设规则给出，只演示输入输出。训练数据未经临床专家逐条审核。

## 使用

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install huggingface-hub==1.33.0
hf download {{MODEL_ID}} --revision v1.2.0 --local-dir medical-model
pip install -r medical-model/requirements.txt
python medical-model/predict_choices.py --bundle medical-model \
  --input medical-model/examples/scenarios.jsonl --batch-size 1
```

使用随包 `ChoicePredictor` 或 CLI。约 130 MB 适配器。首次加载还需固定版本 Qwen3.5-4B 底座，并核验哈希。`--base-model` 可指定已有底座。权重使用自定义格式，不是聊天生成接口。

每行 JSONL 包含 `id` 和 `request`。`request.state` 是材料，`request.questions.decision.instructions` 是判断要求，`criteria` 将语义选项键映射到文字。输出 `answer` 与归一化的 `probabilities`，不生成解释。

## 训练与证据

44,347 条训练样本，label smoothing 0.05。全层 rank 16 / alpha 32 LoRA，选择第 5,000 步。完整训练数据不随模型分发。

[评测报告](https://huggingface.co/{{MODEL_ID}}/blob/main/EVALUATION.md) · [文件校验值](https://huggingface.co/{{MODEL_ID}}/blob/main/manifest.json) · [源码与结果](https://github.com/JuneYaooo/medical-jev-train)

代码、适配器与底座均采用 Apache-2.0，此前以 MIT 发布部分的声明见 NOTICE；数据遵循各来源许可。

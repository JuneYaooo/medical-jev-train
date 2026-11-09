---
license: mit
language:
- zh
- en
base_model: Qwen/Qwen3.5-4B
library_name: pytorch
---

# medical-jev-4b

基于 Qwen3.5-4B 的医疗选择题LoRA模型，从给定候选中选出一项。用于教学题库、模型评测和离线研究，不用于临床决策。

## 成绩与限制

开发基准共 1,514 题，本模型准确率 **67.90%**；接口可回答的 1,414 题准确率为 72.70%。未能回答的题在全部分母中计错。该基准已用于开发诊断，1,620 道封存测试题尚未评测。

训练覆盖 2–26 个候选，已发布接口允许 52 个候选 / 8,192 token，但 27–52 没有医学准确率验证，148 候选中医证型题全部无法回答。不得用于分诊或任何临床决策。LoRA 分诊 40/100，低于原始 4B 的 55/100，90% 预测同一类别。

概率未经校准，不能作为临床风险或复核阈值。构造示例答案由题设规则给出，只演示输入输出。训练数据未经临床专家逐条审核。

## 使用

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install huggingface-hub==1.33.0
hf download {{MODEL_ID}} --revision v1.1.0 --local-dir medical-model
pip install -r medical-model/requirements.txt
python medical-model/predict_choices.py --bundle medical-model \
  --input medical-model/examples/scenarios.jsonl --batch-size 1
```

使用随包 `ChoicePredictor` 或 CLI。约 130 MB 适配器。首次加载还需固定版本 Qwen3.5-4B 底座，并核验哈希。`--base-model` 可指定已有底座。权重使用自定义格式，不是聊天生成接口。

每行 JSONL 包含 `id` 和 `request`。`request.state` 是材料，`request.questions.decision.instructions` 是判断要求，`criteria` 将语义选项键映射到文字。输出 `answer` 与归一化的 `probabilities`，不生成解释。

## 训练与证据

44,847 条训练样本，label smoothing 0.05。全层 rank 16 / alpha 32 LoRA，选择第 5,500 步。完整训练数据不随模型分发。

[评测报告](https://huggingface.co/{{MODEL_ID}}/blob/main/EVALUATION.md) · [文件校验值](https://huggingface.co/{{MODEL_ID}}/blob/main/manifest.json) · [源码与结果](https://github.com/JuneYaooo/medical-jev-train)

代码与适配器采用 MIT，底座采用 Apache-2.0；数据遵循各来源许可。

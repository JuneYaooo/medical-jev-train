# 训练方法

[返回首页](../README.md)

## 环境

Python 3.12、PyTorch 2.8.0（CUDA 12.8）、Transformers 5.19+（支持 Qwen3.5），配套 `flash-linear-attention==0.4.2` 与 `fla-core==0.4.2`（FLA 0.6 与 Triton 3.4 不兼容）；`causal-conv1d==1.7.0` 用 `--no-build-isolation` 安装。在 RTX 5090 上 FP16 比 BF16 更稳定：批量与单条打分最大概率差 0.0036，BF16 超过 0.02。新机器首次反向传播会自动调优 Triton 内核，建议设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。安装后先跑自检。

## 一次完整训练

```bash
pip install -e . && pip install -r requirements-train.txt
hf download --repo-type dataset JuneYao/medical-jev-4b-data --local-dir work/data/release
hf download Qwen/Qwen3.5-4B --local-dir work/models/Qwen3.5-4B

# 1. 自检：批量与单条一致、未训练适配器等于原模型、小样本能拟合、保存后重载一致
jev verify --config configs/jev/release/train.json --input work/data/release/train.jsonl --output runs/verify

# 2. 原始底座基线（用于对比）
jev eval --config configs/jev/release/train.json --input work/data/release/development.jsonl \
  --views 1 --output runs/readout

# 3. 训练
jev train --config configs/jev/release/train.json --output runs/train

# 4. 外部基准（与发布接口相同上限，超过 52 候选分块淘汰）
jev eval --config configs/jev/release/train.json --checkpoint runs/train/best.safetensors \
  --input work/benchmark/medical_decision_v1.jsonl --max-options 52 --max-tokens 8192 \
  --tournament-above 52 --batch-size 1 --output runs/bench-candidate
```

外部基准文件由 `jev import-benchmark --benchmark <jev-healthcare-lab>/benchmarks/medical_decision_v1 --output work/benchmark/medical_decision_v1.jsonl` 生成。中断后用同一命令、同一输出目录重跑会从最近保存处继续；输入或配置变化会拒绝续训。

## 每一步做什么

| 步骤 | 做法 |
| --- | --- |
| 打分 | 问题和候选项套聊天模板，候选编为 A、B、C…；读取下一个 token 在各字母上的 logits 并归一化 |
| 抽样 | 每类任务按哈希顺序最多取 3,000 题，通用医学题取 10,000 题；v1 实际训练 44,847 题 |
| 训练 | 全 32 层 LoRA，rank 16，alpha 32，dropout 0.05；学习率 1e-4，cosine，warmup 3%；batch 8 × 累积 2；2 个 epoch，共 5,606 步；FP16 + SDPA；label smoothing 0.05；每道题随机循环移动选项顺序，避免模型记住答案位置 |
| 上限 | 训练 2–26 个候选、1,536 token；超出的题不进入训练 |
| 选模 | 每 500 步在开发集上评测并保存最好的检查点。v1 评测了 13 个检查点，第 5,500 步开发集准确率最高（82.86%），按宏平均准确率排序也是这一步 |
| 校准 | 可选 `jev calibrate`，只在开发集上拟合温度，不改变答案排序；v1 未校准 |

RTX 5090 32 GB 上约 20 题/秒，约 1.5 小时。

## 发布前验证

1. **外部基准配对比较**：与原始 Qwen3.5-4B 在 1,514 题上逐题配对，按来源组 bootstrap，提升 +8.26 个百分点，区间 [5.74, 10.88]；
2. **留出测试只评一次**：在当时的 1,620 题留出集上评测一次，1,278 题正确（78.89%），记录输入和权重哈希；
3. **发布包复验**：从 Hugging Face 匿名下载发布包，校验每个文件的 SHA-256，在 RTX 5090 上重跑全部基准题，预测与本地完全一致。

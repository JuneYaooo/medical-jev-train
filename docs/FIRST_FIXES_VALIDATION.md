# 首批修复验证记录

验证日期：2026-11-16。基线为 `fc69ed9a1c39d1075687aaf0d4d2641554d4164a`，本地分支 `fix/evaluation-portability-isolation`。未重新训练 4B 模型、修改权重、重跑历史封存考试或覆盖公开数据快照。

## 改动与兼容

- 评测和比较以原始题目为准，未答和不支持计错；校验重复、额外题号及输入/标签元数据。`summary` 新增关键字 `rows`，普通评测和开发集评测均传入原题。存在不支持题时必须提供原题。
- `n`、准确率、宏平均和准确率区间使用全部有标签题。另列 `input_n`、`unlabelled_n`、`supported_n`、`coverage` 和 `supported_accuracy`。NLL、Brier、ECE 仅对有效概率计算，以 `probability_metrics_n` 记录分母；有效答案缺概率仍可计正确。
- `jev compare` 现在必须指定 `--input`；比较脚本仍用 `--benchmark`，两个入口共用完整题集补齐与校验逻辑。
- Windows 普通评测不加载封存锁，JSON I/O 使用 UTF-8；封存功能在 Windows 明确拒绝。Linux 保留互斥、失败留痕与禁止重复运行。
- 数据划分按病例组及精确输入哈希的连通组件进行，排除规则整组件传播，写入前验证隔离。标点、选项重排及近重复只进入独立人工审核清单。

## 环境与全量结果

两端都未安装完整的 torch/transformers 等模型测试依赖，没有下载模型权重。测试使用假模型、纯 CPU 逻辑和真实文件 I/O。

| 环境 | 总数 | 通过 | 跳过 | 失败/错误 |
| --- | ---: | ---: | ---: | ---: |
| Windows CPython 3.12.14 | 52 | 42 | 7 | 3 个错误 |
| WSL Ubuntu 24.04，Python 3.12.3 | 52 | 48 | 4 | 0 |

新增回归共 28 项（计分 10、数据及候选审计 10、平台 8）：Windows 26 通过、2 个 POSIX 专属测试跳过；Linux 28 项全通过。Windows 另外跳过原有 POSIX 封存测试；两端均有 4 项模型依赖测试跳过。

Windows PowerShell 命令（`<python312>` 指实际 Python 3.12 可执行文件）：

```powershell
$env:PYTHONPATH='src'
& <python312> -m unittest discover -s tests -v
```

Linux 命令（仓库根目录）：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

本地完整输出保存在未跟踪的 `work/verification/windows312.log`、`linux312.log` 和 `windows312-baseline.log`。其中 Windows 全量退出码为 1；不能将本批结果表述为所有测试均通过。

## Windows 既有错误

对原始提交执行 `git archive`，在临时干净目录用同一 Windows Python 3.12.14 再运行原 24 项测试：14 通过、4 跳过、1 个失败和 5 个错误。修复后剩余的 3 个错误全部存在于该基线，没有新增失败项：

1. `test_lora_export_has_current_runtime_and_model_card`：导出脚本读取 UTF-8 模型卡时仍使用系统 GBK，触发 `UnicodeDecodeError`。不属于本批普通评测 I/O。
2. `test_prompt_preserves_training_serialization`：旧测试读取 UTF-8 提示词夹具时未指定编码，触发 `UnicodeDecodeError`。
3. `test_hub_snapshot_symlinks_are_supported`：本机默认临时目录 `D:\temp` 下符号链接读取触发 `OSError: [Errno 22] Invalid argument`。Linux 同项通过；没有跳过或屏蔽该失败。

本批已修正 Windows 基线中的配置路径拼接、JSONL 编码不一致和普通评测的 `fcntl` 导入问题。原 POSIX 封存测试在 Windows 改为明确的平台跳过，同时新增 Windows 必须拒绝封存且不得构造模型或创建账本的断言。

## 额外核验

- 计分回归包含 `1307/1557`、`1028/1514`、全部不支持题、无标签题、缺失或无效概率、错误元数据及两个比较入口一致性。
- 用公开固定版本的 1,514 条 v1 预测重新汇总，仍为 1,050 正确、69.352708%：计分修复不改变已有完整预测的正确数。
- [历史更正记录](results/v1/heldout-1557-correction.json) 的来源 Git blob SHA-256、各任务分母和正确数之和、准确率及宏平均均已校验。原始 `heldout-1557.json` 未改；更正结果为 83.94%/75.40%，宏平均 84.62%/76.35%，不能重算的区间保留为空。
- 数据回归覆盖多跳组件、跨组相同请求、排除传播、混合任务、输入顺序确定性、剂量/否定/选项顺序不自动合并、数量守恒及写入前失败拦截。
- 独立候选审计工具已读取公开的 242,921 条训练池和 1,557 条测试记录，按 `clinical-question-sequence/1` 规则输出 ≥90% 的 85 个、≥95% 的 39 个候选。此规则不同于早期核查，不能替代旧报告数量，也不证明实际训练暴露。完整候选在本地 `work/audits/public-test-variants.json`，输入数据未修改。

本批不包含训练配比、训练顺序或语义近重复的自动归组，也未推送或合并远端分支。

# Qwen2.5-VL LoRA 退化问题分析

## 问题现象

上一轮 RTX 5090 full run 已经跑通 Qwen2.5-VL LoRA 训练、评测和报告生成闭环，但 LoRA 后指标没有提升：

- Qwen2.5-VL base `keyword_coverage`：0.9017
- Qwen2.5-VL + LoRA `keyword_coverage`：0.5533
- Qwen2.5-VL base `average_answer_length`：233.6800
- Qwen2.5-VL + LoRA `average_answer_length`：3.2000

主要现象是 LoRA 后回答明显变短。这说明当前训练答案格式、loss mask 或 generation 配置存在问题，不能把该实验写成 LoRA 效果提升。

## 可能原因

- 训练数据中的 assistant 目标过短，很多样本只有数字或一个短词。
- 原训练 collator 只 mask 了 pad token，对用户问题、图像占位和 prompt 部分也计算 loss，不适合标准 SFT。
- 训练目标是短答案，但评测时希望模型给出更完整解释，训练和评测目标不一致。
- generation 的 `max_new_tokens`、prompt 格式和答案格式没有统一控制。
- 当前数据仍然是本地合成 demo 数据，表达模式较单一，不足以支撑稳定泛化结论。

## 已采取的代码修复

- `scripts/prepare_lora_data.py` 新增 `--answer_style explain`，默认生成：

```text
答案：{answer}
依据：{basis}
```

- `basis` 优先来自 `related_knowledge_ids` 对应的知识库内容；找不到时使用关键词或通用依据。
- `scripts/train_lora_qwen_vl_gpu.py` 的 collate 逻辑改为 assistant-only loss mask，只对 assistant 回答部分计算 loss。
- `scripts/eval_qwen_vl_lora.py` 新增 `--prompt_style answer_then_reason`，base 和 LoRA 评测使用同一提示词。
- 评测新增 `too_short_rate`、`exact_match`、`numeric_match`。
- 新增 `scripts/inspect_lora_data.py`，用于训练前检查 assistant answer 长度。
- 新增 `scripts/analyze_qwen_lora_bad_cases.py`，用于对比 base 和 LoRA 的 bad case。

## 本轮修复结果

第二轮 RTX 5090 full run 使用了相同的 train / val / test 划分，即 800 / 100 / 100，并统一使用 `answer_then_reason` 评测 prompt。

| metric | Qwen2.5-VL base | Qwen2.5-VL + LoRA | delta |
|---|---:|---:|---:|
| num_samples | 100.0000 | 100.0000 | +0.0000 |
| exact_match | 0.2100 | 0.0000 | -0.2100 |
| numeric_match | 0.6000 | 0.7600 | +0.1600 |
| too_short_rate | 0.2700 | 0.0000 | -0.2700 |
| keyword_coverage | 0.4817 | 0.8650 | +0.3833 |
| average_answer_length | 26.7400 | 56.1100 | +29.3700 |
| average_latency_seconds | 0.5091 | 0.7699 | +0.2608 |

本轮修复后，LoRA 的 `too_short_rate` 从 0.2700 降到 0.0000，说明短答退化现象得到缓解。`average_answer_length` 从 26.7400 增加到 56.1100，更符合“答案 + 一句话依据”的训练目标。`numeric_match` 和 `keyword_coverage` 也有所改善。

`exact_match` 下降到 0.0000，主要是因为 LoRA 输出变成解释型格式，而 reference answer 仍是短答案。对于这种输出，完全字符串匹配不再适合作为单独评价指标，需要结合数值匹配、关键词覆盖、短答率和 bad case 分析一起看。

## 下一次 RTX 5090 实验如何运行

在服务器项目目录中执行：

```bash
cd /root/autodl-tmp/projects/MathVision-Assistant
git pull

MODEL_NAME=/root/autodl-tmp/models/Qwen/Qwen2___5-VL-3B-Instruct \
RUN_MODE=full \
bash scripts/run_qwen_lora_gpu.sh
```

如果要先做轻量 smoke：

```bash
RUN_MODE=smoke bash scripts/run_qwen_lora_gpu.sh
```

## 如何判断是否改善

重点观察：

- LoRA 的 `too_short_rate` 是否低于上一轮短答退化现象。
- LoRA 的 `average_answer_length` 是否不再异常接近 0。
- LoRA 的 `keyword_coverage` 是否不再明显低于 base。
- `exact_match` 和 `numeric_match` 是否稳定或有所改善；解释型回答下不要单独依赖 exact match。
- `metrics/bad_cases/bad_cases.md` 中是否仍大量出现“LoRA 输出过短”。

本轮结果显示短答率、数值匹配和关键词覆盖已有改善，但这只能说明当前合成 demo 流程更合理，不能直接说明模型在正式 benchmark 上提升。

## 不能夸大结果的说明

当前实验仍然基于本地合成 demo 数据，主要用于验证多模态 LoRA 训练、评测、诊断和报告生成流程。项目亮点应写成“工程闭环跑通”和“发现并修复训练诊断问题”，不要写成“LoRA 显著提升模型能力”。

# RTX 5090 上的 Qwen2.5-VL LoRA 微调实验记录

## 实验目的

本次实验用于验证 MathVision-Assistant 在 RTX 5090 服务器上的多模态 LoRA 微调、评测与
报告生成闭环，包括数据生成、LoRA 数据转换、Qwen2.5-VL 基座模型评测、LoRA adapter 训练、
LoRA 后评测和自动生成报告。

本次实验重点是工程闭环跑通，不是证明 LoRA 带来了效果提升。

## 实验环境

- GPU：RTX 5090 32GB
- 模型：Qwen2.5-VL-3B-Instruct
- 微调方式：PEFT-LoRA
- attention：sdpa

## 第一轮实验：短答退化问题

- run_dir：`runs/20260510_194349`
- run_mode：`full`
- model_name：`/root/autodl-tmp/models/Qwen/Qwen2___5-VL-3B-Instruct`

### 数据设置

本次实验使用项目内生成的本地合成 demo 数据：

| split | count |
|---|---:|
| train | 800 |
| val | 100 |
| test | 100 |

这些数据用于功能性验证，不等同于正式 benchmark。当前结果不能代表模型在 MathVista、ChartQA、DocVQA 等公开数据集上的泛化能力。

### 训练配置

| config | value |
|---|---:|
| max_steps | 300 |
| limit_samples | 1000 |
| lora_r | 8 |
| lora_alpha | 16 |
| attention | sdpa |

### Baseline vs LoRA

| metric | Qwen2.5-VL base | Qwen2.5-VL + LoRA | delta |
|---|---:|---:|---:|
| num_samples | 100.0000 | 100.0000 | +0.0000 |
| keyword_coverage | 0.9017 | 0.5533 | -0.3483 |
| non_empty_rate | 1.0000 | 1.0000 | +0.0000 |
| average_answer_length | 233.6800 | 3.2000 | -230.4800 |
| average_latency_seconds | 1.6419 | 0.1953 | -1.4466 |

### 结果分析

基座模型在 test split 上的 `keyword_coverage` 为 0.9017，LoRA 后下降到 0.5533。
这说明当前合成数据规模、训练策略、答案格式或评测方式仍需要继续改进，
不能把这次实验表述为指标提升。

LoRA 后 `average_answer_length` 从 233.6800 下降到 3.2000，说明模型输出可能变得过短，
存在训练格式、监督信号或 generation 配置问题。平均延迟从 1.6419 秒下降到 0.1953 秒，
但由于输出长度明显变短，这个变化不能直接视为模型效果或推理能力提升。

`non_empty_rate` 在 LoRA 前后均为 1.0000，说明模型都能生成非空回答，但非空回答不代表答案正确。

## 第二轮修复实验

第一轮实验后，项目针对短答退化做了四项修复：

1. LoRA 训练数据使用 `answer_style=explain`，assistant 目标从短答案改为“答案 + 一句话依据”。
2. SFT loss 改为 assistant-only loss mask，只对 assistant 回答部分计算 loss。
3. base 和 LoRA 评测统一使用 `answer_then_reason` prompt。
4. 增加 `too_short_rate`、`exact_match`、`numeric_match` 和 bad case 分析。

第二轮 full run 设置：

- run_dir：`runs/20260510_210429`
- run_mode：`full`
- model_name：`/root/autodl-tmp/models/Qwen/Qwen2___5-VL-3B-Instruct`
- max_steps：300
- limit_samples：1000
- lora_r：8
- lora_alpha：16
- attention：sdpa
- prompt_style：answer_then_reason
- eval max_new_tokens：96
- train / val / test：800 / 100 / 100

第二轮 Baseline vs LoRA：

| metric | Qwen2.5-VL base | Qwen2.5-VL + LoRA | delta |
|---|---:|---:|---:|
| num_samples | 100.0000 | 100.0000 | +0.0000 |
| exact_match | 0.2100 | 0.0000 | -0.2100 |
| numeric_match | 0.6000 | 0.7600 | +0.1600 |
| too_short_rate | 0.2700 | 0.0000 | -0.2700 |
| keyword_coverage | 0.4817 | 0.8650 | +0.3833 |
| average_answer_length | 26.7400 | 56.1100 | +29.3700 |
| average_latency_seconds | 0.5091 | 0.7699 | +0.2608 |

第二轮结果说明短答退化被明显缓解：LoRA 的 `too_short_rate` 从 0.2700 降到 0.0000，
`average_answer_length` 从 26.7400 增加到 56.1100。同时，`numeric_match` 从 0.6000 到
0.7600，`keyword_coverage` 从 0.4817 到 0.8650。

需要注意的是，`exact_match` 下降到 0.0000，主要原因是 LoRA 输出变成“答案 + 依据”的解释型格式，
而 reference answer 仍是短答案。此时完全字符串匹配不再适合作为解释型回答质量的单独指标，
应结合 `numeric_match`、`keyword_coverage`、`too_short_rate` 和人工 bad case 分析一起看。

第二轮实验仍然是本地合成 demo 数据上的功能性评测，不等同于正式 benchmark。

## 当前结论

- 已经跑通 RTX 5090 上的 Qwen2.5-VL LoRA 工程闭环。
- 第一轮实验发现了 LoRA 短答退化问题，第二轮修复后短答率、数值匹配和关键词覆盖有改善。
- 当前实验仍不能证明模型在正式数学视觉问答 benchmark 上泛化提升。
- 当前评测是本地合成 demo 数据上的功能性评测，不等同于正式 benchmark。
- 项目亮点应表述为：完成了数据生成、LoRA 微调、adapter 加载、模型评测和报告生成的工程闭环。

## 后续改进方向

- 增加真实数学视觉问答数据，例如 MathVista、ChartQA、DocVQA 等公开数据集。
- 保证训练集、验证集和测试集严格隔离，并固定随机种子和实验配置。
- 增加更可靠的评测指标，例如 exact match、numeric match、结构化答案评测和人工抽样检查。
- 做 bad case 分析，定位 LoRA 后回答变短、关键词覆盖下降的样本类型。
- 调整 LoRA target modules、learning rate、max_steps、batch 组织方式和 generation `max_new_tokens` 等参数。
- 改进训练答案格式，让模型学习更稳定的短答案或结构化答案输出。
- 第二轮已经用 `answer_style=explain`、assistant-only loss mask、统一 `answer_then_reason`
  评测 prompt 和 bad case 分析缓解了短答退化；后续仍需要在更真实的数据和更严格指标上验证。

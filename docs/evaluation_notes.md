# 评测说明

## 当前指标

- `exact_match`
- `numeric_match`
- `keyword_coverage`
- `retrieval_recall_at_k`
- `average_latency`

## 当前指标为什么合理

- 适合可配置规模的本地合成 demo 数据上的原型阶段快速检查，默认生成 1000 条。
- 可以快速发现模型回答、数值、关键词、检索和延迟问题。
- 有助于做 before / after 对比，例如原始 SmolVLM 与 SmolVLM + LoRA adapter 的小规模对比。
- 数据划分为 train / val / test 后，可以避免 LoRA 训练默认使用测试样本，降低数据泄漏风险。

## 当前指标的不足

- `exact_match` 对等价表达不友好，例如 `(3, 2)`、`(3,2)`、`[3, 2]` 可能表达同一坐标。
- `numeric_match` 不判断单位和语义，也不能验证推理过程是否正确。
- `keyword_coverage` 不能代表完整推理，只能粗略观察关键概念是否出现。
- `retrieval_recall_at_k` 不代表证据被模型正确使用，只说明相关 evidence 是否被检索到。
- 当前 demo 数据仍然是本地合成数据，不能代表正式 benchmark 上的泛化能力。
- `average_latency` 受硬件、缓存、模型加载方式和运行状态影响，不能单独作为模型优劣结论。

## 正式 benchmark 应该怎么做

- 接入 MathVista、ChartQA、DocVQA 等公开数据集。
- 固定 train / validation / test split、随机种子、模型版本和硬件环境。
- 使用更强的答案归一化，覆盖坐标、角度、百分比、`π` 和符号表达式。
- 增加 relaxed accuracy、symbolic equivalence、evidence utilization、hallucination rate 等指标。
- 与 no-RAG、RAG、LoRA、不同 VLM 后端做对比。

# 简历与面试表述参考

## 适合放入简历的项目描述

项目名称：数学视觉问答系统 — 基于 VLM 与 RAG 的 MathVision-Assistant（个人项目）

项目描述：

- 面向函数图像、统计图表、几何图和公式截图等数学图片，构建多模态问答原型，完成合成数据构造、本地知识库检索、VLM 推理、自动评测与 Gradio 演示闭环。
- 基于 TF-IDF 构建本地知识检索模块，将用户问题召回的知识片段作为上下文输入多模态模型，支持 mock、SmolVLM 与 Qwen2.5-VL 后端切换。
- 设计 exact match、numeric match、关键词覆盖率、retrieval recall@k 与平均延迟等评测指标，并完成 SmolVLM 与 SmolVLM + LoRA adapter 的小规模 demo 对比。

## 不建议写的夸大表述

- 不要写“完成正式 benchmark 评测”。
- 不要写“LoRA 显著提升模型效果”。
- 不要写“达到工业级数学视觉问答效果”。
- 不要写“自研多模态大模型”。
- 不要写“完成 Qwen2.5-VL 大规模微调”，除非有真实结果。

## 面试时推荐说法

这个项目是小规模数学视觉问答原型，重点验证 VLM + RAG 的工程闭环，包括数据构造、检索、模型后端、评测、LoRA adapter 加载和 Gradio 演示。当前结果基于 14 条本地合成 demo 数据，不能代表正式 benchmark 泛化能力。后续可以接入 MathVista、ChartQA、DocVQA 等公开数据集做更严格评测。

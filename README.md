# MathVision-Assistant

MathVision-Assistant 是一个面向数学图片的多模态问答项目。它可以处理函数图像、统计图表、几何图、公式截图等输入，并结合本地知识库检索结果生成回答。

这个项目的重点不是追求一个很大的线上系统，而是把多模态问答里常见的几个环节做完整：数据构造、检索、模型推理、评测、可视化页面，以及可选的 LoRA 微调流程。

## 功能

- 支持图片上传和问题输入
- 支持本地 TF-IDF 检索，不依赖 FAISS
- 支持 mock、SmolVLM、Qwen2.5-VL 三种后端
- 支持自动评测并保存 CSV / Markdown 报告
- 支持 Gradio 页面演示
- 提供 SmolVLM 和 Qwen2.5-VL 的 LoRA 训练脚本
- 默认 smoke test 不需要下载外部模型，适合在 Mac 上先跑通

## 环境

我本地开发环境是 Apple Silicon MacBook Air。默认依赖没有使用 CUDA-only 库，也没有把 `flash-attn`、`bitsandbytes`、`deepspeed`、`vllm` 放进默认安装流程。

建议使用 Python 3.10 及以上。我的机器上也用 Python 3.9 跑通过，但新环境更推荐 3.10+。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 快速运行

生成 demo 数据：

```bash
python scripts/make_demo_data.py
```

跑本地 smoke test：

```bash
python scripts/run_smoke_test.py --backend mock
```

单条问答：

```bash
python scripts/ask.py \
  --backend mock \
  --image data/demo/images/line_slope_01.png \
  --question "What is the slope of the line?"
```

运行评测：

```bash
python scripts/run_eval.py --backend mock --top_k 3
```

启动 Gradio 页面：

```bash
python -m mathvision.app.gradio_app
```

运行测试：

```bash
pytest
```

## Demo 数据

`scripts/make_demo_data.py` 会生成一组小型合成数据，包含：

- 直线斜率
- 抛物线顶点
- 柱状图最大值
- 折线图趋势
- 三角形角度
- 圆面积
- 简单公式截图
- 坐标点读数
- 分数化简
- 散点图相关性

数据文件：

```text
data/demo/images/
data/demo/qa.jsonl
data/demo/knowledge_base.jsonl
```

`qa.jsonl` 示例：

```json
{
  "id": "line_slope_01",
  "image": "data/demo/images/line_slope_01.png",
  "question": "What is the slope of the line?",
  "answer": "2",
  "answer_type": "number",
  "keywords": ["slope", "2"],
  "related_knowledge_ids": ["kb_line_slope_01"]
}
```

`knowledge_base.jsonl` 示例：

```json
{
  "id": "kb_line_slope_01",
  "title": "Line slope y=2x+1",
  "content": "For a line y = mx + b, the slope is m. In y=2x+1, m=2.",
  "source_image": "data/demo/images/line_slope_01.png"
}
```

## 模型后端

### mock

`mock` 后端是本地规则后端，用来检查工程流程是否完整。它不会下载模型，适合做 smoke test 和单元测试。

```bash
python scripts/ask.py \
  --backend mock \
  --image data/demo/images/triangle_angle_01.png \
  --question "What is the third angle?"
```

### SmolVLM

SmolVLM 是默认推荐的真实多模态模型后端。首次运行会从 Hugging Face 下载模型：

```bash
python scripts/ask.py \
  --backend smolvlm \
  --image data/demo/images/line_slope_01.png \
  --question "What is the slope of the line?"
```

模型默认使用：

```text
HuggingFaceTB/SmolVLM-500M-Instruct
```

代码会按 `cuda > mps > cpu` 选择设备。Mac 上默认使用 float32 和 eager attention，避免依赖 flash attention。

### Qwen2.5-VL

Qwen2.5-VL 后端作为可选功能保留，更适合在 CUDA 云 GPU 上跑：

```bash
python scripts/ask.py \
  --backend qwen-vl \
  --image data/demo/images/line_slope_01.png \
  --question "What is the slope of the line?"
```

如果缺少依赖，需要安装：

```bash
pip install qwen-vl-utils
pip install -U transformers accelerate
```

## LoRA

### 准备数据

LoRA 训练数据由 `qa.jsonl` 转换得到：

```bash
python scripts/prepare_lora_data.py
```

输出文件：

```text
data/outputs/lora_qwen_vl.jsonl
```

### SmolVLM LoRA

SmolVLM 比 Qwen2.5-VL 小很多，更适合做轻量 LoRA 实验。正式训练仍然建议使用 CUDA GPU；Mac 上可以只跑很短的 dry run，检查流程是否能启动。

Mac 本地 dry run 示例：

```bash
pip install peft datasets

python scripts/train_lora_smolvlm_gpu.py \
  --model_name HuggingFaceTB/SmolVLM-500M-Instruct \
  --train_file data/outputs/lora_qwen_vl.jsonl \
  --output_dir checkpoints/smolvlm500m-lora-mathvision-debug \
  --allow_non_cuda \
  --max_steps 1 \
  --limit_samples 2 \
  --batch_size 1 \
  --grad_accum 1 \
  --lora_r 4 \
  --lora_alpha 8 \
  --gradient_checkpointing
```

使用训练好的 adapter 推理：

```bash
python scripts/ask.py \
  --backend smolvlm \
  --lora_adapter checkpoints/smolvlm500m-lora-mathvision-debug \
  --image data/demo/images/line_slope_01.png \
  --question "What is the slope of the line?"
```

使用 adapter 评测：

```bash
python scripts/run_eval.py \
  --backend smolvlm \
  --lora_adapter checkpoints/smolvlm500m-lora-mathvision-debug \
  --top_k 3 \
  --out_dir reports/smolvlm_lora
```

### Qwen2.5-VL LoRA

Qwen2.5-VL LoRA 脚本更适合云 GPU：

```bash
pip install peft trl datasets qwen-vl-utils

python scripts/train_lora_qwen_vl_gpu.py \
  --model_name Qwen/Qwen2.5-VL-3B-Instruct \
  --train_file data/outputs/lora_qwen_vl.jsonl \
  --output_dir checkpoints/qwen25vl-lora-mathvision
```

`requirements-gpu.txt` 里还放了 `vllm`，但它主要用于后续部署或高吞吐推理，不是本地必须安装的依赖。

## 评测

评测脚本会保存两类文件：

```text
reports/eval_results.csv
reports/eval_summary.md
```

指标包括：

- `exact_match`
- `numeric_match`
- `keyword_coverage`
- `retrieval_recall_at_k`
- `average_latency`

运行：

```bash
python scripts/run_eval.py --backend mock --top_k 3
```

### 当前结果

下面是一次本地评测结果，使用 SmolVLM-500M-Instruct 加载 LoRA adapter，在 14 条 demo 样本上运行：

```bash
python scripts/run_eval.py \
  --backend smolvlm \
  --lora_adapter checkpoints/smolvlm500m-lora-mathvision-all \
  --top_k 3 \
  --out_dir reports/smolvlm_lora_all
```

| metric | value |
|---|---:|
| num_samples | 14 |
| exact_match | 0.7143 |
| numeric_match | 0.5714 |
| keyword_coverage | 0.6548 |
| retrieval_recall_at_k | 1.0000 |
| average_latency | 5.9828s |

这组结果来自本地合成 demo 数据，主要用于检查完整流程和观察错误样本。更严格的效果对比需要接入更大的公开数据集。

## Gradio 页面

页面示例：

![Gradio demo](docs/images/gradio-demo.png)

启动：

```bash
python -m mathvision.app.gradio_app
```

页面支持：

- 上传图片
- 输入问题
- 选择 backend
- 设置检索 top-k
- 填写可选 SmolVLM LoRA adapter 路径
- 查看回答、证据和耗时

## 项目结构

```text
MathVision-Assistant/
├── configs/
├── data/
├── reports/
├── scripts/
├── src/mathvision/
│   ├── app/
│   ├── data/
│   ├── evaluation/
│   ├── rag/
│   ├── retrieval/
│   └── vlm/
└── tests/
```

## 备注

- `mock` 后端只用于本地测试，不代表真实模型能力。
- demo 数据是合成数据，适合验证流程，不适合作为正式 benchmark。
- Mac 上可以跑 SmolVLM 推理，也可以做 LoRA dry run，但不适合长时间训练。
- 如果要做更可靠的效果对比，可以接入 MathVista、ChartQA、DocVQA 等公开数据集。

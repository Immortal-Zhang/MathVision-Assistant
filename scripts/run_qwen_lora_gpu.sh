#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/root/autodl-tmp/projects/MathVision-Assistant}"
if [[ -d "${PROJECT_DIR}" ]]; then
  cd "${PROJECT_DIR}"
fi

export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export TOKENIZERS_PARALLELISM=false

RUN_MODE="${RUN_MODE:-smoke}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-3B-Instruct}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

if [[ "${RUN_MODE}" != "smoke" && "${RUN_MODE}" != "full" ]]; then
  echo "RUN_MODE 只能是 smoke 或 full，当前为：${RUN_MODE}" >&2
  exit 2
fi

if [[ "${RUN_MODE}" == "smoke" ]]; then
  DEFAULT_NUM_SAMPLES=100
  DEFAULT_LIMIT_SAMPLES=2
  DEFAULT_MAX_STEPS=1
  DEFAULT_BATCH_SIZE=1
  DEFAULT_GRAD_ACCUM=1
  DEFAULT_LORA_R=4
  DEFAULT_LORA_ALPHA=8
  DEFAULT_LORA_DROPOUT=0.0
  DEFAULT_LEARNING_RATE=2e-4
  DEFAULT_EVAL_LIMIT=3
  DEFAULT_EVAL_MAX_NEW_TOKENS=96
else
  DEFAULT_NUM_SAMPLES=1000
  DEFAULT_LIMIT_SAMPLES=1000
  DEFAULT_MAX_STEPS=300
  DEFAULT_BATCH_SIZE=1
  DEFAULT_GRAD_ACCUM=4
  DEFAULT_LORA_R=8
  DEFAULT_LORA_ALPHA=16
  DEFAULT_LORA_DROPOUT=0.05
  DEFAULT_LEARNING_RATE=1e-4
  DEFAULT_EVAL_LIMIT=100
  DEFAULT_EVAL_MAX_NEW_TOKENS=96
fi

NUM_SAMPLES="${NUM_SAMPLES:-${DEFAULT_NUM_SAMPLES}}"
LIMIT_SAMPLES="${LIMIT_SAMPLES:-${DEFAULT_LIMIT_SAMPLES}}"
MAX_STEPS="${MAX_STEPS:-${DEFAULT_MAX_STEPS}}"
BATCH_SIZE="${BATCH_SIZE:-${DEFAULT_BATCH_SIZE}}"
GRAD_ACCUM="${GRAD_ACCUM:-${DEFAULT_GRAD_ACCUM}}"
LORA_R="${LORA_R:-${DEFAULT_LORA_R}}"
LORA_ALPHA="${LORA_ALPHA:-${DEFAULT_LORA_ALPHA}}"
LORA_DROPOUT="${LORA_DROPOUT:-${DEFAULT_LORA_DROPOUT}}"
LEARNING_RATE="${LEARNING_RATE:-${DEFAULT_LEARNING_RATE}}"
EVAL_LIMIT="${EVAL_LIMIT:-${DEFAULT_EVAL_LIMIT}}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-${DEFAULT_EVAL_MAX_NEW_TOKENS}}"
PROMPT_STYLE="${PROMPT_STYLE:-answer_then_reason}"
ANSWER_STYLE="${ANSWER_STYLE:-explain}"
KNOWLEDGE_FILE="${KNOWLEDGE_FILE:-data/demo/knowledge_base.jsonl}"
MIN_AVERAGE_ANSWER_LENGTH="${MIN_AVERAGE_ANSWER_LENGTH:-20}"
BF16="${BF16:-true}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="runs/${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
METRICS_DIR="${RUN_DIR}/metrics"
DATA_DIR="${RUN_DIR}/data"
ADAPTER_DIR="${RUN_DIR}/checkpoints/qwen25vl-lora"
REPORT_FILE="${RUN_DIR}/report.md"
LOG_FILE="${LOG_DIR}/run_qwen_lora.log"

mkdir -p "${LOG_DIR}" "${METRICS_DIR}" "${DATA_DIR}" "${ADAPTER_DIR}" /root/autodl-tmp/hf_cache
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "== MathVision Qwen2.5-VL LoRA run =="
echo "run_dir=${RUN_DIR}"
echo "run_mode=${RUN_MODE}"
echo "model_name=${MODEL_NAME}"
echo "num_samples=${NUM_SAMPLES}"
echo "attn_implementation=${ATTN_IMPLEMENTATION}"
echo "prompt_style=${PROMPT_STYLE}"
echo "eval_max_new_tokens=${EVAL_MAX_NEW_TOKENS}"
echo "answer_style=${ANSWER_STYLE}"
echo "knowledge_file=${KNOWLEDGE_FILE}"

python scripts/check_gpu_env.py --out_file "${METRICS_DIR}/env.json"

python scripts/make_demo_data.py --num_samples "${NUM_SAMPLES}" --train_ratio 0.8 --val_ratio 0.1
python scripts/run_smoke_test.py --backend mock
python scripts/run_eval.py --backend mock --top_k 3 --out_dir "${METRICS_DIR}/mock_all"
python scripts/run_eval.py \
  --backend mock \
  --top_k 3 \
  --qa_file data/demo/qa_test.jsonl \
  --out_dir "${METRICS_DIR}/mock_test"

LORA_TRAIN_FILE="${DATA_DIR}/lora_qwen_vl_train.jsonl"
python scripts/prepare_lora_data.py \
  --qa_file data/demo/qa_train.jsonl \
  --out_file "${LORA_TRAIN_FILE}" \
  --answer_style "${ANSWER_STYLE}" \
  --knowledge_file "${KNOWLEDGE_FILE}"

python scripts/inspect_lora_data.py \
  --train_file "${LORA_TRAIN_FILE}" \
  --min_average_length "${MIN_AVERAGE_ANSWER_LENGTH}" \
  --fail_on_short

BF16_ARGS=()
if [[ "${BF16}" == "true" ]]; then
  BF16_ARGS=(--bf16)
fi

GC_ARGS=()
if [[ "${GRADIENT_CHECKPOINTING}" == "true" ]]; then
  GC_ARGS=(--gradient_checkpointing)
fi

python scripts/train_lora_qwen_vl_gpu.py \
  --model_name "${MODEL_NAME}" \
  --train_file "${LORA_TRAIN_FILE}" \
  --output_dir "${ADAPTER_DIR}" \
  --max_steps "${MAX_STEPS}" \
  --limit_samples "${LIMIT_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accum "${GRAD_ACCUM}" \
  --learning_rate "${LEARNING_RATE}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --attn_implementation "${ATTN_IMPLEMENTATION}" \
  "${BF16_ARGS[@]}" \
  "${GC_ARGS[@]}"

python scripts/eval_qwen_vl_lora.py \
  --model_name "${MODEL_NAME}" \
  --qa_file data/demo/qa_test.jsonl \
  --out_dir "${METRICS_DIR}/qwen_base" \
  --limit_samples "${EVAL_LIMIT}" \
  --prompt_style "${PROMPT_STYLE}" \
  --max_new_tokens "${EVAL_MAX_NEW_TOKENS}" \
  --attn_implementation "${ATTN_IMPLEMENTATION}" \
  "${BF16_ARGS[@]}"

python scripts/eval_qwen_vl_lora.py \
  --model_name "${MODEL_NAME}" \
  --adapter_dir "${ADAPTER_DIR}" \
  --qa_file data/demo/qa_test.jsonl \
  --out_dir "${METRICS_DIR}/qwen_lora" \
  --limit_samples "${EVAL_LIMIT}" \
  --prompt_style "${PROMPT_STYLE}" \
  --max_new_tokens "${EVAL_MAX_NEW_TOKENS}" \
  --attn_implementation "${ATTN_IMPLEMENTATION}" \
  "${BF16_ARGS[@]}"

python scripts/analyze_qwen_lora_bad_cases.py \
  --base_csv "${METRICS_DIR}/qwen_base/eval_results.csv" \
  --lora_csv "${METRICS_DIR}/qwen_lora/eval_results.csv" \
  --out_dir "${METRICS_DIR}/bad_cases"

python - "${RUN_DIR}" "${RUN_MODE}" "${METRICS_DIR}/qwen_base/summary.json" \
  "${METRICS_DIR}/qwen_lora/summary.json" "${ADAPTER_DIR}/train_config.json" \
  "${REPORT_FILE}" "${ANSWER_STYLE}" "${METRICS_DIR}/bad_cases" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
run_mode = sys.argv[2]
base = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
lora = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8"))
train_config = json.loads(Path(sys.argv[5]).read_text(encoding="utf-8"))
report_file = Path(sys.argv[6])
answer_style = sys.argv[7]
bad_cases_dir = Path(sys.argv[8])
split_counts = {}
for split_name, split_path in {
    "train": Path("data/demo/qa_train.jsonl"),
    "val": Path("data/demo/qa_val.jsonl"),
    "test": Path("data/demo/qa_test.jsonl"),
}.items():
    if split_path.exists():
        split_counts[split_name] = sum(1 for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip())
    else:
        split_counts[split_name] = 0

metric_keys = [
    "num_samples",
    "exact_match",
    "numeric_match",
    "too_short_rate",
    "keyword_coverage",
    "average_answer_length",
    "average_latency_seconds",
]
lines = [
    "# MathVision Qwen2.5-VL LoRA Run Report",
    "",
    "本报告由 `scripts/run_qwen_lora_gpu.sh` 自动生成。当前评测是小规模功能性评测，不等同于正式 benchmark。",
    "",
    f"- run_dir: `{run_dir}`",
    f"- run_mode: `{run_mode}`",
    f"- model_name: `{train_config['model_name']}`",
    f"- train_file: `{train_config['train_file']}`",
    f"- adapter_dir: `{train_config['output_dir']}`",
    f"- max_steps: {train_config['max_steps']}",
    f"- limit_samples: {train_config['limit_samples']}",
    f"- lora_r: {train_config['lora_r']}",
    f"- lora_alpha: {train_config['lora_alpha']}",
    f"- attention: `{train_config['attn_implementation']}`",
    f"- prompt_style: `{base.get('prompt_style', '')}`",
    f"- eval max_new_tokens: {base.get('max_new_tokens', '')}",
    f"- answer_style: `{answer_style}`",
    f"- train samples: {split_counts['train']}",
    f"- val samples: {split_counts['val']}",
    f"- test samples: {split_counts['test']}",
    "",
    "## Data Split",
    "",
    "| split | count |",
    "|---|---:|",
    f"| train | {split_counts['train']} |",
    f"| val | {split_counts['val']} |",
    f"| test | {split_counts['test']} |",
    "",
    "## Baseline vs LoRA",
    "",
    "| metric | Qwen2.5-VL base | Qwen2.5-VL + LoRA | delta |",
    "|---|---:|---:|---:|",
]
for key in metric_keys:
    base_value = base.get(key, 0)
    lora_value = lora.get(key, 0)
    if isinstance(base_value, (int, float)) and isinstance(lora_value, (int, float)):
        lines.append(f"| {key} | {base_value:.4f} | {lora_value:.4f} | {lora_value - base_value:+.4f} |")
    else:
        lines.append(f"| {key} | {base_value} | {lora_value} | - |")

notes = []
if run_mode == "full" and (train_config["max_steps"] < 300 or train_config["lora_r"] < 8):
    notes.append("本次 full run 使用了降级参数，可能是为了显存或速度控制；结果应按功能性验证解读。")
if not notes:
    notes.append("未检测到 full 参数降级。")

lines.extend(
    [
        "",
        "## 输出文件",
        "",
        f"- 环境信息：`{run_dir}/metrics/env.json`",
        f"- mock all 评测：`{run_dir}/metrics/mock_all/`",
        f"- mock test 评测：`{run_dir}/metrics/mock_test/`",
        f"- Qwen base 评测：`{run_dir}/metrics/qwen_base/`",
        f"- Qwen LoRA 评测：`{run_dir}/metrics/qwen_lora/`",
        f"- bad case 分析：`{bad_cases_dir}`",
        f"- LoRA adapter：`{train_config['output_dir']}`",
        f"- 训练日志：`{train_config['output_dir']}/train_log.json`",
        "",
        "## 诊断说明",
        "",
        "- 如果 LoRA 的 too_short_rate 明显高于 base，说明 LoRA 可能出现短答退化。",
        "- 如果 LoRA 的 keyword_coverage 下降，不能写成指标提升。",
        "- 当前仍然是本地合成 demo 数据，不等同于正式 benchmark。",
        "",
        "## 说明",
        "",
    ]
)
lines.extend(f"- {note}" for note in notes)
lines.append("- 当前数据是本地合成 demo 数据，结果不能代表正式数学视觉问答 benchmark 泛化能力。")
report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"报告已生成：{report_file}")
PY

echo "== Run completed =="
echo "Report: ${REPORT_FILE}"
echo "Adapter: ${ADAPTER_DIR}"
echo "Log: ${LOG_FILE}"

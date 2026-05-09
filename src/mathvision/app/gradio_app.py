"""Gradio demo for MathVision-Assistant."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from mathvision.data.synthetic import generate_demo_dataset
from mathvision.io_utils import read_jsonl
from mathvision.rag.pipeline import MathVisionRAGPipeline
from mathvision.retrieval.text_index import TextRetriever
from mathvision.vlm.smolvlm_backend import SmolVLMBackend


DEFAULT_KB = Path("data/demo/knowledge_base.jsonl")


def _prepare_retriever(kb_file: Path = DEFAULT_KB) -> TextRetriever:
    if not kb_file.exists():
        generate_demo_dataset("data/demo")
    retriever = TextRetriever()
    retriever.build(read_jsonl(kb_file))
    return retriever


def _evidence_to_markdown(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "未检索到证据。"
    lines = ["| id | title | score | content |", "|---|---|---:|---|"]
    for item in evidence:
        content = str(item.get("content", "")).replace("|", "\\|")
        title = str(item.get("title", "")).replace("|", "\\|")
        lines.append(
            f"| {item.get('id', '')} | {title} | {float(item.get('score', 0.0)):.4f} | {content} |"
        )
    return "\n".join(lines)


def answer_question(
    image_path: str | None,
    question: str,
    backend: str,
    top_k: int,
    lora_adapter: str,
) -> tuple[str, str, str]:
    if not image_path:
        return "请先上传图片。", "", ""
    if not question.strip():
        return "请输入问题。", "", ""

    try:
        retriever = _prepare_retriever()
        backend_obj: str | SmolVLMBackend = backend
        adapter_path = lora_adapter.strip()
        if backend == "smolvlm" and adapter_path:
            backend_obj = SmolVLMBackend(lora_adapter=adapter_path)
        pipeline = MathVisionRAGPipeline(backend=backend_obj, retriever=retriever)
        result = pipeline.answer(image_path=image_path, question=question, top_k=int(top_k))
        evidence_md = _evidence_to_markdown(result["evidence"])
        latency = f"{result['latency_seconds']:.3f} 秒，backend={result['backend']}"
        return str(result["answer"]), evidence_md, latency
    except Exception as exc:
        return f"运行失败：{exc}", "", ""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="MathVision-Assistant：数学图表多模态问答系统") as demo:
        gr.Markdown("# MathVision-Assistant：数学图表多模态问答系统")
        gr.Markdown("mock 后端用于本地快速跑通；SmolVLM 后端会下载真实模型。")
        with gr.Row():
            with gr.Column():
                image = gr.Image(label="上传数学图片", type="filepath")
                question = gr.Textbox(
                    label="输入问题",
                    value="What is the slope of the line?",
                    lines=2,
                )
                backend = gr.Dropdown(
                    label="选择 backend",
                    choices=["mock", "smolvlm", "qwen-vl"],
                    value="mock",
                )
                top_k = gr.Slider(
                    label="检索证据 Top-K",
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                )
                lora_adapter = gr.Textbox(
                    label="SmolVLM LoRA adapter 路径（可选）",
                    placeholder="例如 checkpoints/smolvlm500m-lora-mathvision-debug",
                    lines=1,
                )
                submit = gr.Button("开始问答", variant="primary")
            with gr.Column():
                answer = gr.Textbox(label="模型回答", lines=8)
                evidence = gr.Markdown(label="检索证据")
                latency = gr.Textbox(label="耗时")

        submit.click(
            fn=answer_question,
            inputs=[image, question, backend, top_k, lora_adapter],
            outputs=[answer, evidence, latency],
        )
    return demo


def main() -> None:
    demo = build_demo()
    demo.launch()


if __name__ == "__main__":
    main()

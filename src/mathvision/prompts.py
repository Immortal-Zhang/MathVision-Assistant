"""Prompt builders for multimodal math VQA."""

from __future__ import annotations


SYSTEM_PROMPT = (
    "You are MathVision-Assistant, a multimodal assistant for math diagrams, "
    "charts, geometry figures, formulas, and textbook screenshots. Answer "
    "concisely, show the key reasoning, and use retrieved evidence when useful."
)


def build_vqa_prompt(question: str, context: str | None = None) -> str:
    """Build a prompt shared by real VLM backends."""

    context_block = context.strip() if context else "No external evidence provided."
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Retrieved evidence:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer in Chinese or English following the user's question. "
        "If the answer is numeric, include the final value clearly."
    )

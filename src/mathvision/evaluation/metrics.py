"""Evaluation metrics for math visual QA."""

from __future__ import annotations

import math
import re
import unicodedata


def normalize_text(text: str) -> str:
    """Lowercase and remove punctuation-like spacing differences."""

    normalized = unicodedata.normalize("NFKC", str(text)).lower().strip()
    normalized = normalized.replace("π", "pi")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^a-z0-9一-龥./%+\-() ]", "", normalized)
    return normalized.strip()


def exact_match(prediction: str, reference: str) -> float:
    """Return 1 when normalized strings match or the answer appears clearly."""

    pred = normalize_text(prediction)
    ref = normalize_text(reference)
    if not ref:
        return 0.0
    if pred == ref:
        return 1.0
    if ref in pred:
        return 1.0
    return 0.0


def extract_numbers(text: str) -> list[float]:
    """Extract integers, decimals, simple fractions, and percentages."""

    normalized = normalize_text(text)
    values: list[float] = []

    fraction_matches = re.findall(r"(?<!\d)([-+]?\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", normalized)
    occupied: set[str] = set()
    for numerator, denominator in fraction_matches:
        den = float(denominator)
        if not math.isclose(den, 0.0):
            values.append(float(numerator) / den)
            occupied.add(f"{numerator}/{denominator}")

    without_fractions = normalized
    for token in occupied:
        without_fractions = without_fractions.replace(token, " ")

    for match in re.finditer(r"[-+]?\d+(?:\.\d+)?%?", without_fractions):
        token = match.group(0)
        if token.endswith("%"):
            values.append(float(token[:-1]) / 100.0)
        else:
            values.append(float(token))
    return values


def numeric_match(prediction: str, reference: str, tol: float = 1e-3) -> float:
    """Return 1 when any extracted predicted number matches a reference number."""

    pred_numbers = extract_numbers(prediction)
    ref_numbers = extract_numbers(reference)
    if not ref_numbers:
        return 0.0
    for ref in ref_numbers:
        if any(abs(pred - ref) <= tol for pred in pred_numbers):
            return 1.0
    return 0.0


def keyword_coverage(prediction: str, keywords: list[str]) -> float:
    """Fraction of keywords covered by prediction."""

    if not keywords:
        return 1.0
    pred = normalize_text(prediction)
    hits = 0
    for keyword in keywords:
        if normalize_text(keyword) in pred:
            hits += 1
    return hits / len(keywords)


def retrieval_recall_at_k(
    retrieved_ids: list[str], related_knowledge_ids: list[str], k: int
) -> float:
    """Recall@K for retrieved evidence ids."""

    if not related_knowledge_ids:
        return 1.0
    retrieved = set(retrieved_ids[:k])
    related = set(str(item) for item in related_knowledge_ids)
    return len(retrieved & related) / len(related)

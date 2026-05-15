"""
평가 지표

GSM8K: 마지막 숫자(#### 뒤) 비교
cosine_similarity: task vector 간 cosine similarity (논문 핵심 지표)
"""

import re
from typing import Optional

import torch
from safetensors.torch import load_file
from pathlib import Path
import json


# ──────────────────────────────────────────────
# GSM8K 정확도
# ──────────────────────────────────────────────

def extract_answer(text: str) -> Optional[str]:
    """
    GSM8K 정답 추출 — LoRI 포맷 + 원본 포맷 모두 지원
    1) "The final answer is: X" (LoRI 학습 포맷, 예측값)
    2) "#### X" (원본 GSM8K 포맷, 참조값)
    3) 마지막 숫자 (fallback)
    """
    m = re.search(r"[Tt]he final answer is:?\s*([\d,.\-]+)", text)
    if m:
        return m.group(1).strip().replace(",", "")
    m = re.search(r"####\s*([\d,.\-]+)", text)
    if m:
        return m.group(1).strip().replace(",", "")
    nums = re.findall(r"[\d,]+(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None


def gsm8k_accuracy(predictions: list[str], references: list[str]) -> dict:
    correct = 0
    for pred, ref in zip(predictions, references):
        p = extract_answer(pred)
        r = extract_answer(ref)
        if p is not None and r is not None and p == r:
            correct += 1
    acc = correct / len(predictions) if predictions else 0.0
    return {
        "accuracy": acc,
        "correct": correct,
        "total": len(predictions),
    }


# ──────────────────────────────────────────────
# Task Vector Cosine Similarity
# ──────────────────────────────────────────────

def compute_tv_cosine_similarity(
    adapter_path_a: str,
    adapter_path_b: str,
) -> float:
    """
    두 LoRA 어댑터의 task vector cosine similarity 계산
    cos(ΔW_a, ΔW_b) = ⟨ΔW_a, ΔW_b⟩_F / (||ΔW_a||_F · ||ΔW_b||_F)

    ΔW = B @ A * scaling (model 로드 없이 계산)
    """
    def load_deltas(adapter_path):
        with open(Path(adapter_path) / "adapter_config.json") as f:
            config = json.load(f)
        scaling = config["lora_alpha"] / config["r"]
        weights = load_file(Path(adapter_path) / "adapter_model.safetensors")

        deltas = {}
        for key in weights:
            if not key.endswith("lora_A.weight"):
                continue
            b_key = key.replace("lora_A.weight", "lora_B.weight")
            if b_key not in weights:
                continue
            A = weights[key].float()
            B = weights[b_key].float()
            name = key.removeprefix("base_model.model.").replace(".lora_A.weight", ".weight")
            deltas[name] = (B @ A * scaling).flatten()
        return deltas

    deltas_a = load_deltas(adapter_path_a)
    deltas_b = load_deltas(adapter_path_b)

    common = sorted(set(deltas_a) & set(deltas_b))

    # layer별로 내적/노름 누적 (메모리 절약)
    dot = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    for k in common:
        a, b = deltas_a[k], deltas_b[k]
        dot += (a * b).sum().item()
        norm_a_sq += (a * a).sum().item()
        norm_b_sq += (b * b).sum().item()
        del a, b

    denom = (norm_a_sq ** 0.5) * (norm_b_sq ** 0.5)
    return dot / (denom + 1e-8)


def compute_b_cosine_similarity(
    adapter_path_a: str,
    adapter_path_b: str,
) -> float:
    """
    B 행렬 subspace cosine similarity
    ||B_a^T @ B_b||_F / (||B_a||_F · ||B_b||_F)  — 레이어 평균
    """
    weights_a = load_file(Path(adapter_path_a) / "adapter_model.safetensors")
    weights_b = load_file(Path(adapter_path_b) / "adapter_model.safetensors")

    scores = []
    for key in weights_a:
        if not key.endswith("lora_B.weight"):
            continue
        if key not in weights_b:
            continue
        B_a = weights_a[key].float()   # (d_out, r)
        B_b = weights_b[key].float()   # (d_out, r)

        C = B_a.T @ B_b                # (r, r)
        norm_a = (B_a ** 2).sum().sqrt()
        norm_b = (B_b ** 2).sum().sqrt()
        score = C.norm() / (norm_a * norm_b + 1e-8)
        scores.append(score.item())

    return sum(scores) / len(scores) if scores else 0.0

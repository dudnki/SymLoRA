"""
데이터 로더 — Math / Code / NLU SFT

GSM8K train        : LoRI 논문 포맷 (question → step-by-step, "The final answer is: X")
CodeAlpaca         : LoRI 논문 포맷 (Alpaca instruction format)
Commonsense-170k   : LoRI NLU 학습 데이터 (8개 commonsense task mix, Alpaca format)
"""

from datasets import load_dataset, Dataset


def _math_prompt(question: str) -> str:
    return f"{question}\nAnswer the above question. First think step by step and then answer the final number.\n"


def _math_completion(answer: str) -> str:
    return answer.replace("####", "The final answer is:")


def _code_prompt(instruction: str, inp: str) -> str:
    if inp.strip():
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
        )
    else:
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Response:\n"
        )


def load_math(tokenizer, n_samples: int) -> Dataset:
    """
    openai/gsm8k train split
    LoRI 논문과 동일한 포맷 / 설정 (7,473개 전체 사용)
    prompt / completion 분리 저장 → completion-only loss
    """
    print("GSM8K train 로드 중...")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=42)
    if n_samples < len(ds):
        ds = ds.select(range(n_samples))

    def fmt(ex):
        return {
            "prompt":     _math_prompt(ex["question"]),
            "completion": _math_completion(ex["answer"]),
        }

    ds = ds.map(fmt, remove_columns=ds.column_names, desc="포맷 변환")
    print(f"  math: {len(ds):,}개")
    return ds


def load_code(tokenizer, n_samples: int) -> Dataset:
    """
    sahil2801/CodeAlpaca-20k
    LoRI 논문과 동일한 포맷 / 설정
    prompt / completion 분리 저장 → completion-only loss
    """
    print("CodeAlpaca-20k 로드 중...")
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    ds = ds.filter(lambda x: len(x["output"].strip()) > 50, desc="짧은 응답 제거")
    ds = ds.shuffle(seed=42)
    if n_samples < len(ds):
        ds = ds.select(range(n_samples))

    def fmt(ex):
        return {
            "prompt":     _code_prompt(ex["instruction"], ex.get("input", "")),
            "completion": ex["output"],
        }

    ds = ds.map(fmt, remove_columns=ds.column_names, desc="포맷 변환")
    print(f"  code: {len(ds):,}개")
    return ds


def load_nlu(tokenizer, n_samples: int) -> Dataset:
    """
    zwhe99/commonsense_170k
    LoRI가 NLU 학습에 사용한 Commonsense-170k 믹스
    (BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA 의 train split 병합)

    원본 스키마: instruction / input / output / answer
    Alpaca instruction 포맷으로 변환 → completion-only loss
    """
    print("Commonsense-170k 로드 중...")
    ds = load_dataset("zwhe99/commonsense_170k", split="train")
    ds = ds.shuffle(seed=42)
    if n_samples < len(ds):
        ds = ds.select(range(n_samples))

    def fmt(ex):
        instruction = ex.get("instruction", "")
        inp = ex.get("input", "") or ""
        return {
            "prompt":     _code_prompt(instruction, inp),   # Alpaca 포맷 재사용
            "completion": ex["output"],
        }

    ds = ds.map(fmt, remove_columns=ds.column_names, desc="포맷 변환")
    print(f"  nlu: {len(ds):,}개")
    return ds


TASK_CONFIG = {
    "math": {
        "loader": load_math,
        "default_n": 7473,
        "description": "GSM8K train (#### X 형식)",
    },
    "code": {
        "loader": load_code,
        "default_n": 10000,
        "description": "CodeAlpaca-20k SFT",
    },
    "nlu": {
        "loader": load_nlu,
        "default_n": 20000,
        "description": "Commonsense-170k (LoRI NLU 학습 믹스, 20k 서브셋)",
    },
}

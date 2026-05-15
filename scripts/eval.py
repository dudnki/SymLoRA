"""
eval.py
=======
GSM8K 정확도 + HumanEval pass@1 평가

사용법:
    # GSM8K (math)
    python scripts/eval.py --model outputs/base/merged/plain_math1.0_code1.0 --task gsm8k

    # HumanEval (code)
    python scripts/eval.py --model outputs/base/merged/plain_math1.0_code1.0 --task humaneval

    # 둘 다
    python scripts/eval.py --model outputs/base/merged/plain_math1.0_code1.0 --task all

    # 단일 어댑터 평가
    python scripts/eval.py --model outputs/base/adapters/math_sft --task gsm8k --is_adapter
"""

import argparse
import json
import sys
import warnings
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.metrics import gsm8k_accuracy


UNSLOTH_CACHE = str(Path.home() / ".cache/huggingface/hub/models--unsloth--Llama-3.2-3B/snapshots/d4446454d87d51aa42e1fb174f25acc5f8762331")


def _resolve(name: str) -> str:
    """unsloth 허브 모델은 로컬 캐시 경로로 대체."""
    if name == "unsloth/Llama-3.2-3B":
        return UNSLOTH_CACHE
    return name


def _normalize_layer_key(key: str) -> str:
    """모듈 키에서 공통 suffix 추출 (prefix 차이 무시).

    예: 'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight'
      → 'layers.0.self_attn.q_proj.lora_A.weight'
    """
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "layers":
            return ".".join(parts[i:])
    return key


def _apply_init_correction(model, adapter_path: str, scaling: float = 1.0):
    """
    Symmetric/PiSSA init에서 저장된 init weights로 base weight 보정.

    학습 시: W_eff = W_pre - scaling * B_init @ A_init
    PEFT 로드 시: W_pre + scaling * B_final @ A_final (보정 누락)
    올바른 결과: W_pre - scaling * B_init @ A_init + scaling * B_final @ A_final

    따라서 merge 전에 base weight에서 scaling * B_init @ A_init을 빼줘야 한다.

    LoRA 모듈을 직접 순회하여 base_layer에 접근하므로,
    Unsloth/HF 간 모듈 이름 차이에 영향받지 않는다.
    """
    from safetensors.torch import load_file

    init_path = Path(adapter_path) / "init_weights.safetensors"
    if not init_path.exists():
        return False

    init_w = load_file(str(init_path))
    # 정규화된 키로 인덱싱
    init_by_norm = {}
    for k, v in init_w.items():
        norm_k = _normalize_layer_key(k)
        init_by_norm[norm_k] = v

    corrected = 0
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        if len(module.lora_A) == 0:
            continue

        # 정규화된 키로 init weights 매칭
        norm_name = _normalize_layer_key(name)
        a_key = f"{norm_name}.lora_A.weight"
        b_key = f"{norm_name}.lora_B.weight"

        if a_key not in init_by_norm or b_key not in init_by_norm:
            continue

        base_layer = module.base_layer if hasattr(module, "base_layer") else module
        W = base_layer.weight

        A = init_by_norm[a_key].to(device=W.device, dtype=torch.float32)
        B = init_by_norm[b_key].to(device=W.device, dtype=torch.float32)
        delta = (scaling * (B @ A)).to(W.dtype)
        W.data.sub_(delta)
        corrected += 1

    print(f"[init_correction] {corrected} layers corrected from {init_path}")
    return True


def _apply_residual_lora_eval(model, adapter_path: str):
    """residual_R.safetensors가 있으면 residual lora를 적용 + R weights 로드.

    Returns True if applied.
    """
    from safetensors.torch import load_file
    r_path = Path(adapter_path) / "residual_R.safetensors"
    if not r_path.exists():
        return False

    from src.models.residual_lora import apply_residual_lora
    apply_residual_lora(model, verbose=False)

    r_weights = load_file(str(r_path))
    # 키 정규화 (prefix 차이 무시)
    r_by_norm = {_normalize_layer_key(k): v for k, v in r_weights.items()}

    loaded = 0
    for name, module in model.named_modules():
        if not hasattr(module, "residual_R"):
            continue
        norm_name = _normalize_layer_key(name)
        key = f"{norm_name}.residual_R.weight"
        if key in r_by_norm:
            W = module.residual_R.weight
            v = r_by_norm[key].to(device=W.device, dtype=W.dtype)
            W.data.copy_(v)
            loaded += 1

    print(f"[residual_lora] {loaded} R weights loaded")
    return True


def load_model(model_path: str, is_adapter: bool, device: str):
    if is_adapter:
        from peft import PeftModel
        with open(Path(model_path) / "adapter_config.json") as f:
            adapter_cfg = json.load(f)
        base_name = _resolve(adapter_cfg["base_model_name_or_path"])

        # scaling 계산 (alpha / r)
        lora_alpha = adapter_cfg.get("lora_alpha", 16)
        lora_r = adapter_cfg.get("r", 16)
        scaling = lora_alpha / lora_r

        print(f"Base 모델 로드: {base_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_name, dtype=torch.bfloat16, device_map=device,
        )
        model = PeftModel.from_pretrained(model, model_path)

        # Symmetric/PiSSA init 보정 (init_weights.safetensors가 있으면 적용)
        _apply_init_correction(model, model_path, scaling=scaling)

        # Residual LoRA: forward 변경되므로 merge 불가
        has_residual = _apply_residual_lora_eval(model, model_path)
        if not has_residual:
            model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            _resolve(model_path), dtype=torch.bfloat16, device_map=device,
        )
    tokenizer = AutoTokenizer.from_pretrained(_resolve(model_path))
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> str:
    """LoRI 논문 포맷: plain text (base 모델, chat template 없음)"""
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = output_ids[0, input_ids.shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def eval_gsm8k(model, tokenizer, n_samples: int, device: str) -> dict:
    print("GSM8K 로드 중...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    if n_samples < len(ds):
        ds = ds.select(range(n_samples))

    predictions, references = [], []
    for ex in tqdm(ds, desc="GSM8K 평가"):
        # LoRI 논문 포맷
        prompt = f"{ex['question']}\nAnswer the above question. First think step by step and then answer the final number.\n"
        pred = generate(model, tokenizer, prompt, max_new_tokens=256, device=device)
        predictions.append(pred)
        references.append(ex["answer"])

    result = gsm8k_accuracy(predictions, references)
    print(f"GSM8K accuracy: {result['accuracy']:.4f}  ({result['correct']}/{result['total']})")
    return result


def eval_humaneval(model_path: str, is_adapter: bool, n_samples: int) -> dict:
    """
    HumanEval은 human-eval 패키지로 별도 실행
    """
    import subprocess
    import tempfile
    import os

    print("HumanEval 평가 중...")

    script = f"""
import json, sys, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness

model_path = "{model_path}"
is_adapter = {is_adapter}
UNSLOTH_CACHE = str(Path.home() / ".cache/huggingface/hub/models--unsloth--Llama-3.2-3B/snapshots/d4446454d87d51aa42e1fb174f25acc5f8762331")

def resolve(name):
    if name == "unsloth/Llama-3.2-3B":
        return UNSLOTH_CACHE
    return name

if is_adapter:
    import json as _json
    from peft import PeftModel
    from safetensors.torch import load_file as _load_safetensors
    with open(Path(model_path) / "adapter_config.json") as f:
        adapter_cfg = _json.load(f)
    base_name = resolve(adapter_cfg["base_model_name_or_path"])
    lora_alpha = adapter_cfg.get("lora_alpha", 16)
    lora_r = adapter_cfg.get("r", 16)
    scaling = lora_alpha / lora_r
    model = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, model_path)
    # symmetric/pissa init correction
    init_path = Path(model_path) / "init_weights.safetensors"
    if init_path.exists():
        init_w = _load_safetensors(str(init_path))
        def _norm_key(k):
            parts = k.split(".")
            for i, p in enumerate(parts):
                if p == "layers": return ".".join(parts[i:])
            return k
        init_by_norm = {{_norm_key(k): v for k, v in init_w.items()}}
        corrected = 0
        for name, module in model.named_modules():
            if not hasattr(module, "lora_A") or len(module.lora_A) == 0: continue
            norm_name = _norm_key(name)
            a_key = f"{{norm_name}}.lora_A.weight"
            b_key = f"{{norm_name}}.lora_B.weight"
            if a_key not in init_by_norm or b_key not in init_by_norm: continue
            base_layer = module.base_layer if hasattr(module, "base_layer") else module
            W = base_layer.weight
            A = init_by_norm[a_key].to(device=W.device, dtype=torch.float32)
            B = init_by_norm[b_key].to(device=W.device, dtype=torch.float32)
            W.data.sub_((scaling * (B @ A)).to(W.dtype))
            corrected += 1
        print(f"[init_correction] {{corrected}} layers corrected", file=sys.stderr)
    # residual_R 로드 (있으면)
    r_path = Path(model_path) / "residual_R.safetensors"
    has_residual = r_path.exists()
    if has_residual:
        import sys as _sys
        _sys.path.insert(0, "{Path(__file__).parent.parent}")
        from src.models.residual_lora import apply_residual_lora
        apply_residual_lora(model, verbose=False)
        r_w = _load_safetensors(str(r_path))
        r_by_norm = {{_norm_key(k): v for k, v in r_w.items()}}
        for name, module in model.named_modules():
            if not hasattr(module, "residual_R"):
                continue
            norm_name = _norm_key(name)
            key = f"{{norm_name}}.residual_R.weight"
            if key in r_by_norm:
                W = module.residual_R.weight
                v = r_by_norm[key].to(device=W.device, dtype=W.dtype)
                W.data.copy_(v)
    if not has_residual:
        model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(base_name)
else:
    resolved = resolve(model_path)
    model = AutoModelForCausalLM.from_pretrained(resolved, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(resolved)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = model.to("cuda")
model.eval()

problems = read_problems()
samples = []
for task_id, problem in list(problems.items())[:{"None" if n_samples <= 0 else n_samples}]:
    # prompt를 그대로 입력 — chat template 없이 직접 completion
    input_ids = tokenizer(problem['prompt'], return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model.generate(input_ids, max_new_tokens=512, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    completion = tokenizer.decode(out[0, input_ids.shape[-1]:], skip_special_tokens=True)
    samples.append({{"task_id": task_id, "completion": completion}})

out_file = "/tmp/humaneval_samples.jsonl"
write_jsonl(out_file, samples)
result = evaluate_functional_correctness(out_file)
print(json.dumps(result))
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        tmp = f.name

    try:
        proc = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"HumanEval 오류:\n{proc.stderr}")
            return {}
        last_line = proc.stdout.strip().split("\n")[-1]
        result = json.loads(last_line)
        print(f"HumanEval pass@1: {result.get('pass@1', 'N/A'):.4f}")
        return result
    finally:
        Path(tmp).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="모델 경로 (merged 또는 adapter)")
    parser.add_argument("--task", default="all", choices=["gsm8k", "humaneval", "all"])
    parser.add_argument("--is_adapter", action="store_true", help="LoRA adapter 경로인 경우")
    parser.add_argument("--n_samples", type=int, default=0, help="평가 샘플 수 (0=전체)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    n = args.n_samples if args.n_samples > 0 else 10**9
    results = {}

    if args.task in ("gsm8k", "all"):
        model, tokenizer = load_model(args.model, args.is_adapter, args.device)
        gsm_n = min(n, 1319)  # GSM8K test 전체
        results["gsm8k"] = eval_gsm8k(model, tokenizer, gsm_n, args.device)
        del model

    if args.task in ("humaneval", "all"):
        humaneval_n = min(n, 164)
        results["humaneval"] = eval_humaneval(args.model, args.is_adapter, humaneval_n)

    # 결과 저장
    out_path = Path(args.model) / "eval_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n결과 저장: {out_path}")


if __name__ == "__main__":
    main()

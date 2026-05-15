"""
smoke_test.py
=============
Sweep 시작 전 검증:
  - hydra override가 정확히 먹히는지 (output_dir, random_state, freeze_a)
  - LoRA-FA setup (standard + freeze_a)이 trainable params를 절반으로 만드는지
  - Frozen-A (a_only_svd + freeze_a)이 init_weights를 만들고 freeze하는지

실제 학습은 하지 않음 (모델 로드 + LoRA setup만).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM


CACHE = str(Path.home() / ".cache/huggingface/hub/models--unsloth--Llama-3.2-3B/snapshots/d4446454d87d51aa42e1fb174f25acc5f8762331")


def make_model(cfg):
    model = AutoModelForCausalLM.from_pretrained(CACHE, dtype=torch.bfloat16)
    peft_cfg = LoraConfig(
        r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        target_modules=list(cfg.model.lora_target_modules),
        lora_dropout=cfg.model.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_cfg).to("cuda")
    return model


def trainable_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_setup(name, overrides):
    print(f"\n=== {name} ===")
    print(f"overrides: {overrides}")
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="train_sft", overrides=overrides)
    print(f"  init.mode = {cfg.init.mode}")
    print(f"  init.freeze_a = {cfg.init.get('freeze_a', False)}")
    print(f"  model.random_state = {cfg.model.random_state}")
    print(f"  training.output_dir = {cfg.training.output_dir}")
    print(f"  training.run_name = {cfg.training.run_name}")
    print(f"  data.task = {cfg.data.task}")

    model = make_model(cfg)
    total_before = trainable_count(model)

    # init_mode 적용
    if cfg.init.mode != "standard":
        from src.models.sym_lora import apply_symmetric_init
        info = apply_symmetric_init(model, mode=cfg.init.mode, verbose=False)
        n_init_weights = len(info.get("init_weights", {}))
        print(f"  init_weights saved: {n_init_weights}")
    else:
        print(f"  init_weights saved: 0 (standard mode)")

    # freeze_a 적용
    if cfg.init.get("freeze_a", False):
        n = 0
        for _, mod in model.named_modules():
            if hasattr(mod, "lora_A") and len(mod.lora_A) > 0:
                for k in mod.lora_A:
                    mod.lora_A[k].weight.requires_grad_(False)
                    n += 1
        print(f"  frozen lora_A modules: {n}")

    total_after = trainable_count(model)
    pct = total_after / total_before * 100
    print(f"  trainable: {total_before/1e6:.2f}M → {total_after/1e6:.2f}M ({pct:.1f}%)")

    del model
    torch.cuda.empty_cache()
    return total_before, total_after


def main():
    cases = [
        ("LoRA-FA (standard + freeze_a)", [
            "init.mode=standard",
            "init.freeze_a=true",
            "data.task=math",
            "model.random_state=123",
            "training.output_dir=outputs/multiseed/lora_fa/math_seed123",
            "training.run_name=lora_fa_math_s123",
        ]),
        ("Frozen-A (a_only_svd + freeze_a)", [
            "init.mode=a_only_svd",
            "init.freeze_a=true",
            "data.task=math",
            "model.random_state=123",
            "training.output_dir=outputs/multiseed/frozen_a/math_seed123",
            "training.run_name=frozen_a_math_s123",
        ]),
        ("PiSSA (no freeze)", [
            "init.mode=pissa",
            "data.task=math",
            "model.random_state=123",
            "training.output_dir=outputs/multiseed/pissa/math_seed123",
            "training.run_name=pissa_math_s123",
        ]),
    ]

    results = []
    for name, ov in cases:
        before, after = check_setup(name, ov)
        results.append((name, before, after))

    print("\n=== Summary ===")
    for name, b, a in results:
        print(f"  {name:40s}  {b/1e6:6.2f}M → {a/1e6:6.2f}M  ({a/b*100:.1f}%)")
    print("\nOK if:")
    print("  LoRA-FA / Frozen-A: trainable ≈ 51% (B만)")
    print("  PiSSA: trainable ≈ 100% (A+B 둘 다)")


if __name__ == "__main__":
    main()

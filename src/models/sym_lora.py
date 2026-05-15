"""
Symmetric LoRA initialization.

Seed ablation 결과: A_sim ≈ 0.077 (≈ random baseline √(r/d)=0.072)
→ LoRA의 A는 B=0 init 때문에 gradient를 못 받아 init에서 사실상 안 움직임.
→ 이전 cross-task A_sim ~0.95는 같은 seed=42 artifact.

이 모듈은 A, B 둘 다 nonzero로 초기화하되, t=0에서 pretrained forward를
정확히 보존하도록 base weight를 보정하는 방식을 구현.

Modes
-----
  "standard":  PEFT default (A ~ Kaiming, B = 0). Baseline.
  "symmetric": Gradient-balanced random init.
               σ_A = (d_out/(r·d_in²))^(1/4), σ_B = (1/(r·d_out))^(1/4)
               → forward 분산 보존 + A/B gradient magnitude 균형.
               W_eff = W - scaling * B @ A  (t=0 보존)
  "pissa":     A, B from top-r SVD of W. (Meng et al., 2024 style)
               W_eff = W - scaling * B @ A  = residual after top-r.
  "b_only_svd": B의 direction만 SVD (U[:, :r]), A는 random. 크기는 symmetric과 동일.
               → "B init이 중요한가"만 분리 검증.
  "a_only_svd": A의 direction만 SVD (Vh[:r]), B는 random. 크기는 symmetric과 동일.
               → "A init이 중요한가"만 분리 검증.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


VALID_MODES = ("standard", "symmetric", "pissa", "b_only_svd", "a_only_svd")


def _iter_lora_modules(model: nn.Module):
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        if len(module.lora_A) == 0:
            continue
        yield name, module


def _get_scaling(module: nn.Module, adapter_key: str) -> float:
    if hasattr(module, "scaling") and adapter_key in module.scaling:
        val = module.scaling[adapter_key]
        return float(val) if not isinstance(val, torch.Tensor) else val.item()
    return 1.0


@torch.no_grad()
def apply_symmetric_init(
    model: nn.Module,
    mode: str = "standard",
    verbose: bool = True,
) -> dict:
    """
    LoRA A/B를 재초기화하고 base weight를 보정하여 t=0 forward를 보존.

    forward(x) = W_eff @ x + scaling * B(t) @ A(t) @ x
    t=0: W_eff + scaling * B_init @ A_init = W_pretrained (정확히)
    delta from pretrained: scaling * (B(t)A(t) - B_init A_init)

    Returns
    -------
    dict with keys:
      - mode, n_layers: 기본 정보
      - init_weights: {name.lora_A.weight: Tensor, name.lora_B.weight: Tensor, ...}
        eval 시 base weight 보정 및 삼각 측량에 사용.
        standard mode에서는 빈 dict.
    """
    mode = mode.lower()
    if mode not in VALID_MODES:
        raise ValueError(f"init mode must be one of {VALID_MODES}, got {mode!r}")

    if mode == "standard":
        if verbose:
            print(f"[sym_lora] mode=standard — keeping PEFT default (B=0)")
        return {"mode": "standard", "n_layers": 0, "init_weights": {}}

    n_layers = 0
    delta_norms = []
    base_norms = []
    init_weights = {}

    for name, module in _iter_lora_modules(model):
        adapter_key = next(iter(module.lora_A.keys()))
        A_mod = module.lora_A[adapter_key]  # nn.Linear(d_in, r)
        B_mod = module.lora_B[adapter_key]  # nn.Linear(r, d_out)

        r = A_mod.out_features
        d_in = A_mod.in_features
        d_out = B_mod.out_features

        base_layer = module.base_layer if hasattr(module, "base_layer") else module
        W_base = base_layer.weight.data  # (d_out, d_in)
        scaling = _get_scaling(module, adapter_key)

        dtype = A_mod.weight.dtype
        device = A_mod.weight.device

        if mode == "symmetric":
            # Gradient-balanced init:
            #   조건1 (forward 분산 보존): r * d_in * σ_A² * σ_B² = 1
            #   조건2 (gradient 균형):     d_out * σ_B² = d_in * σ_A²
            # → σ_A = (d_out / (r * d_in²))^(1/4)
            #   σ_B = (1 / (r * d_out))^(1/4)
            std_A = (d_out / (r * d_in ** 2)) ** 0.25
            std_B = (1.0 / (r * d_out)) ** 0.25
            A_new = torch.randn(r, d_in, device=device, dtype=torch.float32) * std_A
            B_new = torch.randn(d_out, r, device=device, dtype=torch.float32) * std_B

        elif mode == "pissa":
            W_fp = W_base.detach().to(device=device, dtype=torch.float32)
            U, S, Vh = torch.linalg.svd(W_fp, full_matrices=False)
            sqrt_S = torch.sqrt(S[:r])
            B_new = U[:, :r] * sqrt_S.unsqueeze(0)
            A_new = (sqrt_S.unsqueeze(1) * Vh[:r, :]) / max(scaling, 1e-12)

        elif mode == "b_only_svd":
            # B만 data-driven (SVD 방향), A는 random. 크기는 symmetric과 동일.
            # → "B의 direction이 중요한가"만 분리 검증.
            std_A = (d_out / (r * d_in ** 2)) ** 0.25
            std_B = (1.0 / (r * d_out)) ** 0.25
            A_new = torch.randn(r, d_in, device=device, dtype=torch.float32) * std_A
            target_B_norm = torch.randn(d_out, r, device=device, dtype=torch.float32).mul(std_B).norm()

            W_fp = W_base.detach().to(device=device, dtype=torch.float32)
            U, _, _ = torch.linalg.svd(W_fp, full_matrices=False)
            B_dir = U[:, :r]                                        # (d_out, r), orthonormal cols, ‖·‖_F = √r
            B_new = B_dir * (target_B_norm / math.sqrt(r))

        elif mode == "a_only_svd":
            # A만 data-driven (SVD 방향), B는 random. 크기는 symmetric과 동일.
            # → "A의 direction이 중요한가"만 분리 검증.
            std_A = (d_out / (r * d_in ** 2)) ** 0.25
            std_B = (1.0 / (r * d_out)) ** 0.25
            B_new = torch.randn(d_out, r, device=device, dtype=torch.float32) * std_B
            target_A_norm = torch.randn(r, d_in, device=device, dtype=torch.float32).mul(std_A).norm()

            W_fp = W_base.detach().to(device=device, dtype=torch.float32)
            _, _, Vh = torch.linalg.svd(W_fp, full_matrices=False)
            A_dir = Vh[:r, :]                                       # (r, d_in), orthonormal rows, ‖·‖_F = √r
            A_new = A_dir * (target_A_norm / math.sqrt(r))

        # init weights 수집 (eval 보정 + 삼각 측량용)
        init_weights[f"{name}.lora_A.weight"] = A_new.cpu().contiguous()
        init_weights[f"{name}.lora_B.weight"] = B_new.cpu().contiguous()

        A_mod.weight.data.copy_(A_new.to(dtype))
        B_mod.weight.data.copy_(B_new.to(dtype))

        delta = (scaling * (B_new @ A_new)).to(W_base.dtype)
        W_base.data.sub_(delta)

        n_layers += 1
        delta_norms.append(delta.norm().item())
        base_norms.append(W_base.norm().item())

    if verbose:
        avg_delta = sum(delta_norms) / len(delta_norms)
        avg_base = sum(base_norms) / len(base_norms)
        print(
            f"[sym_lora] mode={mode}, {n_layers} layers reinit, "
            f"avg ||delta||={avg_delta:.2f}, avg ||W_base||={avg_base:.2f}, "
            f"ratio={avg_delta/avg_base:.4e}"
        )

    return {"mode": mode, "n_layers": n_layers, "init_weights": init_weights}

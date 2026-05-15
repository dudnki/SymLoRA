"""
anchor_freedom_analysis.py
==========================
"Anchor + Freedom" 가설 정량 검증.

각 method × task × seed에 대해 학습된 A, B와 init A, B를 비교하여 측정:

  1. rel_A    = ‖A_f - A_i‖_F / ‖A_i‖_F        (A의 상대 변화도)
  2. rel_B    = ‖B_f - B_i‖_F / ‖B_i‖_F        (B의 상대 변화도)
  3. asym     = rel_B / rel_A                    (anchor-freedom asymmetry)
  4. cos_A    = 1 - cos(A_f.flat, A_i.flat)     (A 방향 변화, scale-invariant)
  5. cos_B    = 1 - cos(B_f.flat, B_i.flat)     (B 방향 변화, scale-invariant)
  6. W_chg    = ‖scaling*(B_f@A_f - B_i@A_i)‖_F  (W에 대한 effective 변화)

사용 가설:
  - PiSSA: A,B 둘 다 SVD anchor → asym ≈ 1, 둘 다 적게 변함
  - a_only_svd: A=anchor, B=freedom → asym > 1 (B가 더 변함)
  - b_only_svd: B=anchor, A=freedom → asym < 1 (A가 더 변함)
  - frozen_a: A frozen (rel_A ≈ 0) → asym 무한, B만 freedom

각 metric은 모든 LoRA layer (196개)의 평균.
"""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

import torch
from safetensors.torch import load_file


ROOT = Path(__file__).parent.parent
MULTISEED = ROOT / "outputs" / "multiseed"

# init_weights.safetensors가 있는 method만 분석
CONFIGS = ["pissa", "a_only_svd", "b_only_svd", "frozen_a"]
TASKS = ["math", "code"]
SCALING = 1.0  # lora_alpha=16, r=16 → α/r=1

UNSLOTH_CACHE = str(
    Path.home()
    / ".cache/huggingface/hub/models--unsloth--Llama-3.2-3B/snapshots/d4446454d87d51aa42e1fb174f25acc5f8762331"
)


# ─── base weight 로드 (W_pretrained normalize 용) ───
_BASE_WEIGHTS_CACHE: dict[str, torch.Tensor] | None = None


def load_base_weights() -> dict[str, torch.Tensor]:
    """Llama-3.2-3B base linear weights를 layer 이름별 dict로 반환.
    키 정규화: 'layers.X.module_name'."""
    global _BASE_WEIGHTS_CACHE
    if _BASE_WEIGHTS_CACHE is not None:
        return _BASE_WEIGHTS_CACHE

    from transformers import AutoModelForCausalLM

    print(f"[base] loading {UNSLOTH_CACHE} (CPU, fp32)...")
    model = AutoModelForCausalLM.from_pretrained(
        UNSLOTH_CACHE, torch_dtype=torch.float32, device_map="cpu"
    )
    weights: dict[str, torch.Tensor] = {}
    target_suffixes = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        parts = name.split(".")
        if "layers" not in parts:
            continue
        idx = parts.index("layers")
        norm_name = ".".join(parts[idx:])
        if norm_name.endswith(target_suffixes):
            weights[norm_name] = module.weight.detach().clone().float()
    del model
    print(f"[base] {len(weights)} target layers loaded")
    _BASE_WEIGHTS_CACHE = weights
    return weights


# ─────────────────────────────────────────────────────────────────
def _normalize_key(key: str) -> str:
    parts = key.split(".")
    for i, p in enumerate(parts):
        if p == "layers":
            return ".".join(parts[i:])
    return key


def _pair_AB(weights_norm: dict[str, torch.Tensor]):
    """{layer_base: {'A': tensor, 'B': tensor}} 로 짝짓기."""
    pairs: dict[str, dict] = {}
    for k, v in weights_norm.items():
        if k.endswith(".lora_A.weight"):
            base = k[: -len(".lora_A.weight")]
            pairs.setdefault(base, {})["A"] = v.float()
        elif k.endswith(".lora_B.weight"):
            base = k[: -len(".lora_B.weight")]
            pairs.setdefault(base, {})["B"] = v.float()
    return {b: (ab["A"], ab["B"]) for b, ab in pairs.items() if "A" in ab and "B" in ab}


def _rel_change(final: torch.Tensor, init: torch.Tensor) -> float:
    init_norm = init.flatten().norm().item()
    if init_norm < 1e-12:
        return float("nan")
    return (final - init).flatten().norm().item() / init_norm


def _cos_change(final: torch.Tensor, init: torch.Tensor) -> float:
    f, i = final.flatten(), init.flatten()
    fn, in_ = f.norm().item(), i.norm().item()
    if fn < 1e-12 or in_ < 1e-12:
        return float("nan")
    cs = (f @ i).item() / (fn * in_)
    return 1.0 - cs


def _w_level_change(A_f, B_f, A_i, B_i, scaling: float = SCALING) -> float:
    return float((scaling * (B_f @ A_f - B_i @ A_i)).norm())


def _w_relative_change(A_f, B_f, A_i, B_i, W_pre, scaling: float = SCALING) -> float:
    """‖scaling*(B_f@A_f - B_i@A_i)‖ / ‖W_pre‖ — W에 대한 normalized 변화."""
    delta_W = scaling * (B_f @ A_f - B_i @ A_i)
    w_norm = W_pre.norm().item()
    if w_norm < 1e-12:
        return float("nan")
    return delta_W.norm().item() / w_norm


def _mean(xs):
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return statistics.mean(xs) if xs else float("nan")


# ─────────────────────────────────────────────────────────────────
def analyze_run(run_dir: Path, base_weights: dict | None = None) -> dict | None:
    final_p = run_dir / "adapter_model.safetensors"
    init_p = run_dir / "init_weights.safetensors"
    if not final_p.exists() or not init_p.exists():
        return None

    final = {_normalize_key(k): v for k, v in load_file(str(final_p)).items()}
    init = {_normalize_key(k): v for k, v in load_file(str(init_p)).items()}

    final_pairs = _pair_AB(final)
    init_pairs = _pair_AB(init)

    rel_A, rel_B, cos_A, cos_B, W_chg, W_rel = [], [], [], [], [], []
    for base in init_pairs:
        if base not in final_pairs:
            continue
        A_i, B_i = init_pairs[base]
        A_f, B_f = final_pairs[base]
        rel_A.append(_rel_change(A_f, A_i))
        rel_B.append(_rel_change(B_f, B_i))
        cos_A.append(_cos_change(A_f, A_i))
        cos_B.append(_cos_change(B_f, B_i))
        W_chg.append(_w_level_change(A_f, B_f, A_i, B_i))
        if base_weights is not None and base in base_weights:
            W_rel.append(_w_relative_change(A_f, B_f, A_i, B_i, base_weights[base]))

    s = {
        "rel_A": _mean(rel_A),
        "rel_B": _mean(rel_B),
        "cos_A": _mean(cos_A),
        "cos_B": _mean(cos_B),
        "W_chg": _mean(W_chg),
        "W_rel": _mean(W_rel) if W_rel else float("nan"),
        "n_layers": len(rel_A),
    }
    s["asym"] = (
        s["rel_B"] / s["rel_A"] if s["rel_A"] > 1e-12 else float("inf")
    )
    return s


def collect(config: str, task: str, base_weights: dict | None = None) -> dict[int, dict]:
    out = {}
    cfg_dir = MULTISEED / config
    if not cfg_dir.exists():
        return out
    for sub in sorted(cfg_dir.iterdir()):
        name = sub.name
        if not name.startswith(f"{task}_seed"):
            continue
        try:
            seed = int(name.split("seed")[-1])
        except ValueError:
            continue
        m = analyze_run(sub, base_weights=base_weights)
        if m is not None:
            out[seed] = m
    return out


def _agg(per_seed: dict, key: str) -> tuple[float, float]:
    """(mean, std) over seeds."""
    vals = [m[key] for m in per_seed.values()]
    vals = [
        v
        for v in vals
        if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
    ]
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)


def _fmt(v: float, prec: int = 4) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "—"
        if math.isinf(v):
            return "∞"
    return f"{v:.{prec}f}"


# ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--no-base", action="store_true", help="base model 로드 skip (W_rel 측정 안 함)")
    args = ap.parse_args()

    base_weights = None if args.no_base else load_base_weights()

    # 수집
    grid = {cfg: {t: collect(cfg, t, base_weights=base_weights) for t in TASKS} for cfg in CONFIGS}

    # ── 표 1: 핵심 metric (mean over seeds) ──
    headers = [
        "Method",
        "Task",
        "rel_A",
        "rel_B",
        "asym (B/A)",
        "cos_A",
        "cos_B",
        "W_chg",
        "W_rel",
        "n",
    ]
    rows = []
    for cfg in CONFIGS:
        for task in TASKS:
            ps = grid[cfg][task]
            if not ps:
                rows.append([cfg, task, "—", "—", "—", "—", "—", "—", "—", "0"])
                continue
            row = [cfg, task]
            for k in ["rel_A", "rel_B", "asym", "cos_A", "cos_B", "W_chg", "W_rel"]:
                m, _s = _agg(ps, k)
                row.append(_fmt(m, 4))
            row.append(str(len(ps)))
            rows.append(row)

    if args.markdown:
        print(f"| {' | '.join(headers)} |")
        print(f"| {' | '.join(['---'] * len(headers))} |")
        for r in rows:
            print(f"| {' | '.join(r)} |")
    else:
        col_w = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        sep = "  "
        line = sep.join(h.ljust(col_w[i]) for i, h in enumerate(headers))
        print(line)
        print("-" * len(line))
        for r in rows:
            print(sep.join(r[i].ljust(col_w[i]) for i in range(len(r))))

    # ── 표 2: Asymmetry 핵심 비교 (가설 검증) ──
    print()
    print("# Anchor-Freedom Asymmetry (B의 상대 변화 / A의 상대 변화)")
    print("# 가설: pissa≈1 (둘 다 anchor), a_only>1 (B=freedom), b_only<1 (A=freedom),")
    print("#       frozen_a=∞ (A frozen, B만 학습)")
    print()
    for cfg in CONFIGS:
        for task in TASKS:
            ps = grid[cfg][task]
            if not ps:
                continue
            mean_asym, std_asym = _agg(ps, "asym")
            seeds_str = ", ".join(
                f"s{s}={ps[s]['asym']:.2f}" for s in sorted(ps.keys())
                if not math.isinf(ps[s]['asym'])
            ) or "(all inf, A frozen)"
            print(f"  {cfg:12s} {task:5s}: asym = {_fmt(mean_asym, 3)} ± {_fmt(std_asym, 3)}    [{seeds_str}]")

    # ── 표 3: PiSSA vs a_only_svd 직접 비교 (사용자 가설 핵심) ──
    print()
    print("# 사용자 가설 직접 검증: PiSSA vs a_only_svd")
    print("# 만약 사용자 가설이 맞으면:")
    print("#   - a_only_svd의 cos_A < pissa의 cos_A (A는 둘 다 SVD anchor니까 비슷)")
    print("#   - a_only_svd의 cos_B > pissa의 cos_B (a_only는 B가 freedom)")
    print()
    for task in TASKS:
        p_ps = grid["pissa"].get(task, {})
        a_ps = grid["a_only_svd"].get(task, {})
        if not p_ps or not a_ps:
            continue
        p_relA, _ = _agg(p_ps, "rel_A")
        p_relB, _ = _agg(p_ps, "rel_B")
        a_relA, _ = _agg(a_ps, "rel_A")
        a_relB, _ = _agg(a_ps, "rel_B")
        p_cosA, _ = _agg(p_ps, "cos_A")
        p_cosB, _ = _agg(p_ps, "cos_B")
        a_cosA, _ = _agg(a_ps, "cos_A")
        a_cosB, _ = _agg(a_ps, "cos_B")
        print(f"  [{task}]")
        print(f"    PiSSA       : rel_A={_fmt(p_relA)}, rel_B={_fmt(p_relB)}, cos_A={_fmt(p_cosA, 6)}, cos_B={_fmt(p_cosB, 6)}")
        print(f"    A-only SVD  : rel_A={_fmt(a_relA)}, rel_B={_fmt(a_relB)}, cos_A={_fmt(a_cosA, 6)}, cos_B={_fmt(a_cosB, 6)}")
        # 비대칭 비교
        if math.isfinite(p_relA) and math.isfinite(a_relA) and a_relA > 0:
            ratio_relB = a_relB / p_relB if p_relB > 0 else float('nan')
            print(f"    → A-only가 PiSSA 대비 B 변화량 ×{_fmt(ratio_relB, 2)}")


if __name__ == "__main__":
    main()

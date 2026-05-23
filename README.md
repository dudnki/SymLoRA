# SymLoRA: SVD 초기화와 A 행렬 동결을 통한 효율적 LoRA 미세조정

KCC 2026 투고 논문 *"SVD 초기화와 A 행렬 동결을 통한 효율적 LoRA 미세조정"* 의 공식 구현체입니다.

본 저장소는 LoRA의 두 어댑터 행렬 `A`, `B`를 **사전 학습 가중치 W의 절단 SVD** 로 초기화한 뒤, `A`를 **동결**하고 `B`만 학습하는 단순한 기법(**Frozen-A LoRA**)을 제안합니다. Llama-3.2-3B 기준으로 표준 LoRA 대비 **학습 파라미터를 49% 줄이면서도 GSM8K 정확도를 +3.54%p, HumanEval pass@1을 +1.62%p 향상**시킵니다.

---

## 1. 연구 동기

LoRA[Hu+22]는 사전 학습 가중치 `W ∈ ℝ^(d_out × d_in)` 에 저랭크 갱신 `ΔW = (α/r)·BA` 를 더해 LLM을 미세조정하는 사실상 표준 PEFT 기법입니다. 그러나 표준 구현은 두 가지 한계를 가집니다.

1. **무작위 초기화는 W의 구조 정보를 활용하지 않는다.** A를 Kaiming, B를 0으로 두면 학습이 무작위 부분공간에서 시작합니다.
2. **A와 B를 모두 학습하므로, 파라미터의 절반이 A에 묶인다.** Llama-3.2-3B(r=16, 7개 모듈)에서 A는 11.93M, B는 12.39M로 거의 동일합니다.

이를 보완하려는 두 갈래의 선행 연구가 있습니다.

| 접근 | 대표 연구 | 한계 |
| --- | --- | --- |
| SVD 기반 초기화 | PiSSA[Meng+24] | A, B 둘 다 학습 → 파라미터 절약 X |
| A 동결 | LoRA-FA[Zhang+23], LoRI[Zhang+25] | A가 무작위 → W의 구조와 무관 → 성능 손실 |

**핵심 질문**: SVD 초기화의 *구조 활용* 과 LoRA-FA의 *파라미터 절약* 을 동시에 얻을 수 없는가?

본 연구는 PiSSA-style SVD로 A를 초기화한 뒤 **동결**하는 단순한 결합이 정답임을 실증합니다. 더 나아가 LoRA를 `A=구조 anchor` × `B=task freedom`의 분업 관점에서 해석하고, 학습으로 인한 유효 가중치 변화 ΔW = sBA 의 크기를 사전학습 W로 정규화한 지표 **W_rel** 을 도입해 그 메커니즘을 정량화합니다.

---

## 2. 제안 방법: Frozen-A LoRA

### 2.1 PiSSA-style SVD 초기화

각 대상 선형 층의 사전 학습 가중치 `W` 에 대해 상위 r개의 절단 SVD를 수행합니다.

```
W ≈ U_r Σ_r V_rᵀ,    U_r ∈ ℝ^(d_out × r),  V_r ∈ ℝ^(d_in × r)

A_init = √Σ_r · V_rᵀ      ∈ ℝ^(r × d_in)
B_init = U_r · √Σ_r       ∈ ℝ^(d_out × r)
W_res  = W − B_init A_init   (학습 시 새로운 base)
```

이 초기화는 시작 시점에 `ΔW = B_init A_init` 이 `W` 의 주성분과 정확히 일치하도록 합니다.

### 2.2 A 동결

학습 시 `A.requires_grad = False` 로 설정하여 그래디언트를 차단합니다. AdamW 옵티마이저는 B의 모멘텀 상태만 추적하므로 옵티마이저 메모리도 절반으로 줄어듭니다.

```python
for name, module in model.named_modules():
    if hasattr(module, "lora_A"):
        for key in module.lora_A:
            module.lora_A[key].weight.requires_grad_(False)
```

### 2.3 파라미터 비교 (Llama-3.2-3B, r=16, 7개 모듈)

| 구분 | A 파라미터 | B 파라미터 | 학습 합계 |
| --- | ---: | ---: | ---: |
| Standard LoRA | 11.93M (49.1%) | 12.39M (50.9%) | 24.31M |
| **Frozen-A (제안)** | **0 (동결)** | 12.39M | **12.39M (−49.0%)** |

### 2.4 Anchor-Freedom 프레임워크와 측정 지표

본 연구는 LoRA의 두 행렬을 **anchor (구조 보존)** 와 **freedom (task 적응)** 의 분업으로 해석합니다. 학습 후 어댑터 `A_f, B_f`, 초기 어댑터 `A_i, B_i`, 사전학습 가중치 `W`, scaling `s = α/r = 1` 를 사용해 다음 지표들을 정의합니다 ([scripts/anchor_freedom_analysis.py](scripts/anchor_freedom_analysis.py)).

**개별 행렬의 상대 변화량** — 각 어댑터가 초기값에서 얼마나 멀어졌는지.

```
rel_A = ||A_f − A_i||_F / ||A_i||_F
rel_B = ||B_f − B_i||_F / ||B_i||_F
asym  = rel_B / rel_A        (anchor-freedom 비대칭도)
```

`rel`이 작은 쪽이 anchor, 큰 쪽이 freedom 역할로 해석됩니다. Frozen-A는 `rel_A = 0`이므로 `asym = ∞` 로 가장 깔끔한 분업입니다.

**유효 가중치 변화 W_rel** — ΔW = sBA 의 학습 전후 변화량을 사전학습 가중치 `W` 로 정규화한 값.

```
W_rel = ||s·(B_f A_f − B_i A_i)||_F / ||W||_F
```

즉 "이 LoRA가 W 대비 몇 % 크기의 유효 변화를 만들었는가" 를 나타냅니다. W_rel이 너무 작으면(`PiSSA = 1.34%`) 학습이 주성분 근처를 거의 벗어나지 못한 over-anchored 상태이고, anchor/freedom 분업이 명확할 때 적절한 변화량이 확보됩니다.

모든 지표는 196개 LoRA layer(28층 × 7 모듈) 평균 후 seed 평균입니다.

---

## 3. 실험 결과

### 3.1 메인 결과 — Llama-3.2-3B, Multi-seed 평균 (seed 42/123/777)

| 방법 | 학습 비율 | GSM8K (%) | HumanEval pass@1 (%) |
| --- | ---: | ---: | ---: |
| Standard LoRA | 100% | 43.62 | 27.85 |
| LoRA-FA (Kaiming A 동결) | 51% | 33.31 | 25.41 |
| PiSSA | 100% | 47.49 | 29.07 |
| A-only SVD | 100% | **47.99** | **30.49** |
| B-only SVD | 100% | 46.55 | 30.08 |
| **Frozen-A (제안)** | **51%** | **47.16** | **29.47** |

핵심 관찰:

1. **Frozen-A는 학습 파라미터를 절반으로 줄이면서도 표준 LoRA 대비 GSM8K +3.54%p, HumanEval +1.62%p 향상.**
2. **무작위 A를 동결하는 LoRA-FA(33.31%)는 표준 LoRA보다 GSM8K에서 −10.31%p로 크게 뒤처짐.** 동결 자체가 아니라 **무엇을 동결하느냐**가 결정적임을 보여줍니다.
3. **Frozen-A는 PiSSA(파라미터 2배)와 비교해도 거의 동등한 성능.** A를 학습하는 것이 큰 이득이 없음을 의미합니다.

### 3.2 메커니즘 분석 — W_rel 지표 (표 2)

| 방법 | rel(A) | rel(B) | W_rel | 해석 |
| --- | ---: | ---: | ---: | --- |
| PiSSA | 0.043 | 0.043 | 1.34% | A, B 모두 anchor → over-anchored, 표현 범위 협소 |
| A-only SVD | 0.020 | 0.018 | 2.39% | A=anchor, B=freedom |
| B-only SVD | 0.020 | 0.020 | 2.56% | B=anchor, A=freedom |
| **Frozen-A** | **0.000** | 0.019 | 1.70% | A=anchor 고정, B=freedom (가장 명확한 분업) |

**해석**: PiSSA는 A와 B 둘 다 SVD anchor에 묶여 ΔW의 유효 변화가 W 대비 1.34%에 그칩니다(over-anchored). 한쪽을 freedom으로 풀어주면 W_rel이 2%대로 증가하면서 더 넓은 변화가 일어나고 성능도 개선됩니다. Frozen-A는 `rel_A = 0` 으로 anchor/freedom 분업이 가장 명확하면서도, B 한쪽만 학습해 충분한 W_rel(1.70%)을 확보합니다.

### 3.3 추가 관찰

- **입력 부분공간의 태스크 비종속성**: GSM8K와 HumanEval에서 학습된 ΔW의 V 부분공간 사이 principal angle 평균은 0.06으로 거의 직교에 가까웠습니다. 그럼에도 동일한 SVD-init A로 두 태스크에서 모두 우수한 성능이 나온다는 사실은, SVD로 추출된 입력 기저가 다운스트림 태스크에 의존하지 않는 충분히 풍부한 표현임을 시사합니다.
- **Frozen-A의 시드 분산이 PiSSA보다 큼** (GSM8K σ=1.62 vs 0.12). B가 0에서 시작해 더 다양한 최적화 경로를 탐색하기 때문이며, 버그가 아니라 의도된 거동입니다.

---

## 4. 실행 방법

### 4.1 환경

- Python 3.13, CUDA 12.1
- 단일 GPU 가정 (16GB 이상 권장, RTX 4070 Ti SUPER 16GB에서 검증)

```bash
git clone https://github.com/dudnki/SymLoRA.git
cd SymLoRA

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

주요 의존성: `unsloth`, `peft`, `trl`, `transformers`, `hydra-core`, `mlflow`.

### 4.2 데이터 준비

GSM8K, CodeAlpaca-20k는 HuggingFace에서 자동으로 받습니다.

```bash
python scripts/prepare_dataset.py
```

캐시 위치는 기본 `cache/`. 외부 디렉터리와 공유하려면 프로젝트 루트에 `data` symlink를 만들면 됩니다.

### 4.3 학습

모든 설정은 [configs/train_sft.yaml](configs/train_sft.yaml)에 모여 있으며, Hydra override 문법으로 변경합니다.

```bash
# 표준 LoRA (baseline, math)
python scripts/train_sft.py

# PiSSA
python scripts/train_sft.py init.mode=pissa

# Frozen-A (제안 방법)
python scripts/train_sft.py init.mode=a_only_svd init.freeze_a=true

# LoRA-FA (Kaiming A 동결)
python scripts/train_sft.py init.mode=standard init.freeze_a=true

# 코드 태스크로 전환
python scripts/train_sft.py init.mode=a_only_svd init.freeze_a=true data.task=code
```

초기화 모드 일람:

| `init.mode` | A 초기화 | B 초기화 | 비고 |
| --- | --- | --- | --- |
| `standard` | Kaiming | 0 | PEFT 기본값 |
| `symmetric` | gradient-balanced | gradient-balanced | `W_eff = W − sBA` 로 t=0 보존 |
| `pissa` | √Σ · Vᵀ | U · √Σ | A, B 모두 학습 |
| `a_only_svd` | √Σ · Vᵀ | random | `freeze_a=true` 와 결합 시 **Frozen-A** |
| `b_only_svd` | random | U · √Σ | |

### 4.4 평가

```bash
# GSM8K (1,319 문제, greedy)
python scripts/eval.py --adapter outputs/a_only_svd/math_sft --task math

# HumanEval (164 문제, pass@1)
python scripts/eval.py --adapter outputs/a_only_svd/code_sft --task code
```

### 4.5 논문 표/그림 재현

전체 sweep (초기화 모드 × 태스크 × 시드) 실행:

```bash
bash scripts/run_all_experiments.sh
```

표/그림 생성:

```bash
python scripts/aggregate_results.py        # 표 1 (메인 결과)
python scripts/anchor_freedom_analysis.py  # 표 2 (rel_A, rel_B, W_rel)
python scripts/plot_w_rel.py               # 그림 1 (W_rel vs 성능)
```

학습 메트릭은 MLflow(`sqlite:///mlflow.db`)에 기록됩니다.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## 5. 저장소 구성

```
SymLoRA/
├── src/
│   ├── models/sym_lora.py        # 6가지 초기화 모드 + freeze_a 플래그
│   ├── training/sft.py           # SFT trainer
│   ├── data/loaders.py           # GSM8K / CodeAlpaca / Commonsense 로더
│   └── evaluation/metrics.py     # GSM8K accuracy, HumanEval pass@1
├── scripts/
│   ├── train_sft.py              # 학습 entrypoint (Hydra)
│   ├── eval.py                   # 평가
│   ├── prepare_dataset.py        # 데이터셋 다운로드/전처리
│   ├── aggregate_results.py      # MLflow → 표 1 집계
│   ├── anchor_freedom_analysis.py # 표 2 계산 (rel_A, rel_B, W_rel)
│   ├── plot_w_rel.py             # 그림 1 (W_rel vs 성능)
│   ├── smoke_test.py             # sweep 전 환경 검증
│   └── run_all_experiments.sh    # multi-seed sweep
├── configs/train_sft.yaml        # 모든 hyperparameter
├── requirements.txt
└── pyproject.toml
```

git에서 제외되는 항목(모두 재생성 가능): `outputs/`, `cache/`, `logs/`, `mlflow.db`, `unsloth_compiled_cache/`, `.venv/`, `data/` symlink.

---

## 6. 실험 환경 요약

| 항목 | 값 |
| --- | --- |
| 모델 | Llama-3.2-3B (Unsloth, bf16) |
| LoRA rank / α | 16 / 16 (scaling=1.0) |
| 대상 모듈 | q, k, v, o, gate, up, down (28층 × 7 = 196개 모듈) |
| 옵티마이저 | AdamW 8-bit |
| 학습률 / 스케줄 | 5e-5, cosine, warmup 5% |
| Batch | 2 × grad_accum 8 (effective 16) |
| Epoch | 3 |
| Seed | 42 / 123 / 777 (multi-seed 평균) |
| GPU | RTX 4070 Ti SUPER 16GB, CUDA 12.1, WSL2 |

---

## 7. 인용

```bibtex
@inproceedings{symlora2026,
  title     = {SVD 초기화와 A 행렬 동결을 통한 효율적 LoRA 미세조정},
  author    = {Anonymous},
  booktitle = {한국정보과학회 학술발표논문집 (KCC)},
  year      = {2026}
}
```

## 8. 참고문헌

- [Hu+22] E. J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR, 2022.
- [Meng+24] F. Meng et al., "PiSSA: Principal Singular Values and Singular Vectors Adaptation," NeurIPS, 2024.
- [Zhang+23] L. Zhang et al., "LoRA-FA: Memory-efficient Low-rank Adaptation," arXiv:2308.03303, 2023.
- [Zhang+25] J. Zhang et al., "LoRI: Reducing Cross-Task Interference in Multi-Task Low-Rank Adaptation," arXiv, 2025.
- [Kopiczko+24] D. J. Kopiczko et al., "VeRA: Vector-based Random Matrix Adaptation," ICLR, 2024.
- [Liu+24] S.-Y. Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation," ICML, 2024.

학습 백엔드로 [unsloth](https://github.com/unslothai/unsloth)를 사용하였습니다.

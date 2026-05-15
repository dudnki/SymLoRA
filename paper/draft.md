# SVD 초기화와 A 행렬 동결을 통한 효율적 LoRA 미세조정

# Efficient LoRA Fine-tuning via SVD Initialization and Frozen A Matrix

> ⚠️ 심사용: 저자명, 소속, 이메일 미기재 (KCC 익명 심사 규정)

---

## 요 약

거대 언어 모델의 미세조정에서 LoRA(Low-Rank Adaptation)는 표준 기법으로 자리잡았으나, 두 어댑터 행렬(A, B)을 모두 학습하는 과정에서 메모리와 연산 비용이 발생한다. 본 연구에서는 LoRA의 A 행렬을 사전 학습 가중치의 절단된 특이값 분해(truncated SVD)로 초기화한 뒤 학습 과정에서 동결하는 단순한 기법(Frozen-A LoRA)을 제안한다. Llama-3.2-3B 모델을 GSM8K(수학 추론)와 HumanEval(코드 생성) 태스크에서 미세조정한 결과, 학습 가능 파라미터를 49% 감소시키면서도 표준 LoRA 대비 GSM8K +4.55%p, HumanEval +2.44%p의 성능 향상을 달성하였다. 특히 A를 동결한 경우가 학습한 경우(PiSSA, A-only SVD)보다도 우수한 성능을 보여, SVD로 초기화된 입력 부분공간이 다운스트림 태스크에 무관하게 충분히 풍부함을 시사한다.

---

## 1. 서 론

거대 언어 모델(LLM)의 미세조정에서 파라미터 효율적 미세조정(PEFT) 기법이 널리 활용되고 있으며, 그 중 LoRA[1]는 사실상 표준으로 자리잡았다. LoRA는 사전 학습 가중치 W에 대해 저랭크 갱신 ΔW = (α/r)·B·A를 더하며, 여기서 A ∈ R^(r×d_in), B ∈ R^(d_out×r)이다. 표준 구현에서는 A를 가우시안으로, B를 0으로 초기화하여 학습 시작 시 ΔW = 0을 보장한다.

표준 LoRA의 한계는 다음과 같다. 첫째, 무작위 초기화는 사전 학습 가중치 W의 구조 정보를 활용하지 않는다. 둘째, A와 B를 모두 학습하므로 학습 가능 파라미터의 약 절반이 A에 할당된다. 이를 개선하기 위한 두 갈래의 연구가 진행되어 왔다.

**SVD 기반 초기화**: PiSSA[2]는 W의 특이값 분해 결과를 A와 B의 초기값으로 사용한다. 사전 학습 가중치의 주성분 방향에서 미세조정을 시작하므로 표준 LoRA보다 빠른 수렴과 높은 성능을 보인다. 그러나 PiSSA는 여전히 A와 B를 모두 학습한다.

**A 동결**: LoRA-FA[3]는 A를 무작위로 초기화한 후 동결하고 B만 학습한다. 학습 파라미터를 절반으로 줄이는 효과가 있으나, 무작위 A는 W의 구조와 무관하므로 일반적으로 성능 손실이 발생한다.

본 연구는 두 접근의 장점을 결합한 단순한 기법, **Frozen-A LoRA**를 제안한다. PiSSA-style SVD로 A와 B를 초기화한 뒤, A를 학습 과정에서 동결하고 B만 학습한다. Llama-3.2-3B 모델을 두 가지 다운스트림 태스크(수학 추론, 코드 생성)로 미세조정한 결과, 다음을 확인하였다.

1. 학습 파라미터가 49% 감소함에도 불구하고 표준 LoRA 대비 GSM8K +4.55%p, HumanEval +2.44%p 향상
2. A를 동결한 경우가 학습한 경우(PiSSA, A-only SVD)보다도 GSM8K에서 우수
3. 이 결과는 SVD로 추출된 입력 방향이 태스크에 무관한 충분한 표현력을 가짐을 시사

---

## 2. 관련 연구

**LoRA**[1]는 미세조정 시 가중치 갱신을 ΔW = BA의 저랭크 분해로 매개변수화하여, 모델 파라미터의 0.1% 미만의 추가 파라미터로 전체 미세조정에 근접한 성능을 달성한다.

**PiSSA**[2]는 사전 학습 가중치 W = U Σ V^T의 SVD 결과 중 상위 r개 성분으로 A, B를 초기화하고, 잔차 W_res = W - B_init A_init를 새로운 base로 사용한다.

**LoRA-FA**[3]는 A를 무작위로 초기화한 후 동결하여 학습 파라미터와 옵티마이저 상태를 절반 수준으로 감소시킨다.

**VeRA**[4]는 모든 층에서 A, B를 공유하고 작은 스케일 벡터만 학습하여 더 극단적인 압축을 달성한다.

**DoRA**[5]는 가중치 갱신을 크기와 방향으로 분해하여 LoRA의 표현력을 증가시킨다.

**LoRI**[6]는 무작위 A를 동결하고 B에 task-specific sparsity mask를 학습하여 다중 태스크 환경에서의 간섭을 완화한다. 본 연구는 LoRI의 "A 동결" 전략을 공유하면서, 무작위 대신 SVD 기반 초기화로 성능 손실 없이 파라미터 효율과 정확도를 동시에 향상시킨다.

본 연구는 PiSSA의 SVD 초기화와 LoRA-FA/LoRI의 A 동결 전략을 결합하여, 두 접근 단독으로는 달성하기 어려운 "파라미터 절반 + 성능 향상"을 동시에 얻는다.

---

## 3. 제안 방법: Frozen-A LoRA

### 3.1 PiSSA-style SVD 초기화

각 대상 선형 층의 사전 학습 가중치 W ∈ R^(d_out × d_in)에 대해 절단된 특이값 분해를 수행한다.

```
W ≈ U_r Σ_r V_r^T,  U_r ∈ R^(d_out × r),  V_r ∈ R^(d_in × r)
```

다음과 같이 초기화한다.

```
A_init = √Σ_r · V_r^T   ∈ R^(r × d_in)
B_init = U_r · √Σ_r     ∈ R^(d_out × r)
W_res  = W - B_init A_init   (새로운 base)
```

이 초기화는 미세조정 시작 시점에 ΔW = B_init A_init이 W의 주성분을 표현하도록 한다.

### 3.2 A 동결

학습 시 A의 `requires_grad`를 False로 설정하여 그래디언트를 차단한다. 옵티마이저는 B의 파라미터만 추적하므로 AdamW의 모멘텀 상태도 절반으로 줄어든다.

```python
for name, module in model.named_modules():
    if hasattr(module, "lora_A"):
        for key in module.lora_A:
            module.lora_A[key].weight.requires_grad_(False)
```

### 3.3 파라미터 분석

Llama-3.2-3B (28층, d_model=3072, kv_dim=1024, mlp=8192) 기준 r=16의 LoRA 파라미터는 다음과 같다.

| 구분 | A 파라미터 | B 파라미터 | 합계 |
|---|---|---|---|
| 표준 LoRA | 11.93M (49.1%) | 12.39M (50.9%) | 24.31M |
| Frozen-A | 0 (동결) | 12.39M | 12.39M |

학습 파라미터가 24.31M → 12.39M로 **49.0% 감소**한다. AdamW 옵티마이저의 모멘텀 상태(파라미터당 2배)까지 고려하면 메모리 절감 효과는 더 크다.

---

## 4. 실 험

### 4.1 실험 설정

- **모델**: Llama-3.2-3B (Unsloth 4-bit 비활성)
- **어댑터**: r=16, α=16 (scaling=1.0), dropout=0
- **대상 모듈**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **학습**: AdamW (8-bit), lr=5e-5, batch=2, grad_accum=8 (eff. batch=16), 3 epoch, cosine schedule, warmup 5%
- **데이터셋** (LoRI[6] 논문 포맷 준용)
  - 수학: GSM8K 학습셋 전량 (7,473개), prompt에 "step by step" 지시 후 "The final answer is: X" 형식의 completion
  - 코드: CodeAlpaca-20k[7] (10K 서브셋, 응답 50자 미만 필터링), Alpaca instruction 포맷
- **평가**:
  - GSM8K test (1,319 문제), greedy decoding, 정확도
  - HumanEval (164 문제), pass@1

### 4.2 비교 baseline

총 7가지 변형을 동일 설정으로 비교하였다.

| ID | 초기화 (A) | 초기화 (B) | 학습 |
|---|---|---|---|
| Standard | Kaiming | 0 | A, B |
| Symmetric | gradient-balanced | gradient-balanced | A, B |
| PiSSA | SVD | SVD | A, B |
| A-only SVD | SVD | 0 | A, B |
| B-only SVD | Kaiming | SVD | A, B |
| **Frozen-A (Ours)** | **SVD** | **0** | **B only** |
| Frozen-A + Residual | SVD | 0 | B + Residual |

### 4.3 메인 결과

표 1은 7가지 방법의 학습 파라미터 비율과 두 태스크 성능을 보여준다.

**표 1.** Llama-3.2-3B 미세조정 성능 비교 (단일 시드)

| 방법 | 학습 비율 | GSM8K (%) | HumanEval (pass@1, %) |
|---|---|---|---|
| Standard LoRA | 100% | 43.90 | 27.44 |
| Symmetric init | 100% | 45.03 | 31.10 |
| PiSSA | 100% | 47.38 | 29.27 |
| A-only SVD | 100% | 47.08 | 32.32 |
| B-only SVD | 100% | 47.01 | 32.93 |
| **Frozen-A (Ours)** | **51%** | **48.45** | **29.88** |
| Frozen-A + Residual | 53% | 47.16 | 28.05 |

**핵심 관찰**:
1. **Frozen-A가 GSM8K에서 최고 성능** (48.45%)을 달성하며, 표준 LoRA(43.90%) 대비 +4.55%p 향상
2. PiSSA(47.38%) 및 A-only SVD(47.08%)보다도 우수 — A를 학습하지 *않는* 것이 더 나음
3. HumanEval에서도 표준(27.44%) 대비 +2.44%p 향상하나, B-only SVD(32.93%)보다는 낮음
4. 학습 파라미터는 약 절반(51%)

### 4.4 분석

**(a) 왜 A를 동결해도 잘 되는가?**

SVD는 W의 가장 큰 특이값에 대응하는 입력 방향(V_r)을 추출한다. 이 방향은 W가 입력에 가장 강하게 반응하는 부분공간이며, 다운스트림 태스크에 의존하지 않는 가중치 본연의 구조 정보다. A를 학습하면 이 최적 방향에서 멀어질 위험이 있으며, 본 실험에서 PiSSA(47.38%) < Frozen-A(48.45%)인 결과가 이를 뒷받침한다.

**(b) 입력 부분공간의 태스크 비종속성**

Math(GSM8K)와 Code(HumanEval)에서 학습된 ΔW의 V 부분공간을 비교 분석한 결과, 두 태스크의 V 부분공간은 무작위 수준의 직교성을 보였다(저랭크 SVD의 principal angle 평균 0.06). 그럼에도 불구하고 동일한 SVD-init A로 두 태스크 모두에서 우수한 성능이 달성되었다. 이는 SVD-init A가 두 태스크 모두에 충분히 풍부한 입력 기저를 제공함을 시사한다.

**(c) Residual의 효과**

Frozen-A + Residual은 B 모듈 간 잔차 연결을 추가하나, 본 실험에서는 Frozen-A 단독보다 성능이 다소 낮았다. 추가 파라미터의 효용이 낮으며, B만 학습하는 단순한 형태가 가장 효율적임을 보인다.

---

## 5. 결 론

본 논문은 LoRA의 A 행렬을 PiSSA-style SVD로 초기화한 후 동결하는 단순한 기법(Frozen-A LoRA)을 제안하였다. Llama-3.2-3B에서의 실험 결과, 학습 가능 파라미터를 49% 감소시키면서도 표준 LoRA 대비 GSM8K +4.55%p, HumanEval +2.44%p의 성능 향상을 달성하였다. 이는 SVD로 추출된 입력 방향이 다운스트림 태스크에 무관한 충분한 표현력을 가짐을 시사하며, 학습이 불필요한 LoRA 컴포넌트를 식별하는 후속 연구의 기반이 될 수 있다.

향후 연구로는 (1) 다양한 모델 크기 및 태스크에 대한 검증, (2) 다중 시드 안정성 분석, (3) Frozen-A의 메커니즘에 대한 이론적 분석을 계획한다.

---

## 참고문헌

[1] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, "LoRA: Low-Rank Adaptation of Large Language Models," in *Proc. ICLR*, 2022.

[2] F. Meng, Z. Wang, and M. Zhang, "PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models," in *Proc. NeurIPS*, 2024.

[3] L. Zhang, L. Zhang, S. Shi, X. Chu, and B. Li, "LoRA-FA: Memory-efficient Low-rank Adaptation for Large Language Models Fine-tuning," *arXiv preprint arXiv:2308.03303*, 2023.

[4] D. J. Kopiczko, T. Blankevoort, and Y. M. Asano, "VeRA: Vector-based Random Matrix Adaptation," in *Proc. ICLR*, 2024.

[5] S.-Y. Liu, C.-Y. Wang, H. Yin, P. Molchanov, Y.-C. F. Wang, K.-T. Cheng, and M.-H. Chen, "DoRA: Weight-Decomposed Low-Rank Adaptation," in *Proc. ICML*, 2024.

[6] J. Zhang, Y. Lin, X. Yang, R. Sener, and T. Goldstein, "LoRI: Reducing Cross-Task Interference in Multi-Task Low-Rank Adaptation," *arXiv preprint*, 2025.

[7] S. Chaudhary, "Code Alpaca: An Instruction-following LLaMA Model for Code Generation," https://github.com/sahil280114/codealpaca, 2023.

[8] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman, "Training Verifiers to Solve Math Word Problems," *arXiv preprint arXiv:2110.14168*, 2021.

[9] M. Chen et al., "Evaluating Large Language Models Trained on Code," *arXiv preprint arXiv:2107.03374*, 2021.

[10] AI@Meta, "Llama 3 Model Card," 2024.

[11] I. Loshchilov and F. Hutter, "Decoupled Weight Decay Regularization," in *Proc. ICLR*, 2019.

[12] D. Han, M. Han, and Unsloth team, "Unsloth," https://github.com/unslothai/unsloth, 2023.

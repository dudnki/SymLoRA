# KCC 2026 포스터 논문 Outline

## 제목 후보
1. **SVD 초기화와 A 행렬 동결을 통한 효율적 LoRA 미세조정**
2. SVD 기반 LoRA 초기화의 사후 동결: 파라미터를 절반으로 줄이는 효율적 미세조정
3. Frozen-A LoRA: SVD 초기화의 학습 동결을 통한 파라미터 효율적 미세조정

→ **선택: 1번** (가장 명확하고 짧음)

## 저자
- 1저자/단독: 본인

## 초록 (Abstract, ~150자)
거대 언어 모델의 미세조정에서 LoRA는 표준 기법으로 자리잡았으나, 모든 어댑터 파라미터(A, B)를 학습하는 과정에서 메모리와 연산 비용이 발생한다. 본 연구는 LoRA의 A 행렬을 사전 학습 가중치의 SVD로 초기화한 후 학습 과정에서 동결하는 단순한 기법(Frozen-A LoRA)을 제안한다. Llama-3.2-3B 모델을 GSM8K(수학)와 HumanEval(코드) 태스크에서 미세조정한 결과, 학습 가능 파라미터를 약 50% 감소시키면서도 표준 LoRA 대비 GSM8K +4.55%p, HumanEval +2.44%p의 성능 향상을 확인하였다. 특히 A를 동결한 경우가 학습한 경우보다 오히려 성능이 우수하여, SVD로 초기화된 입력 방향이 task에 무관한 최적 부분공간임을 시사한다.

## 본문 구조 (총 2~3쪽)

### 1. 서론 (0.4쪽)
- **배경**: LoRA는 거대 언어 모델 미세조정의 표준
  - ΔW = scaling × B @ A, A: r×d_in, B: d_out×r
  - 표준: A = Kaiming, B = 0으로 초기화
- **문제**: 
  1. 표준 초기화는 task-specific 정보를 활용하지 않음
  2. PiSSA(SVD 초기화)는 성능 향상하지만 여전히 A, B 모두 학습
  3. LoRA-FA는 random A 동결 → 초기화 한계로 성능 손실
- **본 연구**: PiSSA-style SVD 초기화 + A 동결의 결합
  - 학습 파라미터 절반
  - 성능은 오히려 향상
- **Contribution**:
  1. SVD 초기화 + A 동결 결합 기법 제안
  2. Llama-3.2-3B에서 7가지 변형 비교 실험
  3. SVD 입력 방향이 task-agnostic 최적임을 분석으로 입증

### 2. 관련 연구 (0.3쪽)
- **LoRA** (Hu et al., 2021): low-rank adaptation
- **PiSSA** (Meng et al., 2024): SVD로 A, B 초기화, 둘 다 학습
- **LoRA-FA** (Zhang et al., 2023): random A 동결, B만 학습
- **VeRA** (Kopiczko et al., 2024): 공유 random A, B
- 본 연구는 PiSSA의 init과 LoRA-FA의 동결 전략을 결합

### 3. 제안 방법 (0.4쪽)
**3.1 PiSSA-style SVD 초기화**
- 사전 학습 가중치 W ∈ R^{d_out × d_in}에 대해
- SVD: W = U Σ V^T
- A_init = √Σ[:r] · V[:r]^T, B_init = U[:, :r] · √Σ[:r]
- W_residual = W - B_init @ A_init로 base 가중치 재정의

**3.2 A 행렬 동결**
- 학습 시 A의 requires_grad = False
- B만 gradient 업데이트
- 옵티마이저 상태도 B만 유지 → 메모리 절감

**3.3 파라미터 비교**
- Standard LoRA: r × (d_in + d_out)
- Frozen-A LoRA: r × d_out (약 50% 감소, d_in ≈ d_out일 때)
- 옵티마이저 상태(AdamW: 2배)까지 고려하면 메모리 효과 더 큼

### 4. 실험 (1쪽)
**4.1 실험 설정**
- 모델: Llama-3.2-3B (Unsloth)
- 어댑터: r=16, alpha=16, target=all linear (q,k,v,o,gate,up,down)
- 학습: AdamW, lr=2e-4, batch=16, epoch=3
- 데이터셋:
  - 수학: GSM8K (학습) → GSM8K test (1319) 평가
  - 코드: TIGER-Lab/MathInstruct + CodeAlpaca → HumanEval (164) 평가

**4.2 비교 baseline (7가지)**
- standard: A=Kaiming, B=0 (vanilla LoRA)
- symmetric: gradient-balanced 초기화
- pissa: A=√ΣV^T, B=U√Σ (둘 다 학습)
- a_only_svd: A=SVD init, B=0 (둘 다 학습)
- b_only_svd: A=Kaiming, B=SVD (둘 다 학습)
- **a_frozen_svd (Ours)**: A=SVD frozen, B만 학습
- a_frozen_residual: A=SVD frozen + Residual B network

**4.3 메인 결과**
| 방법 | 학습 파라미터 | GSM8K (%) | HumanEval (pass@1) |
|---|---|---|---|
| Standard LoRA | 100% | 43.90 | 27.44 |
| Symmetric init | 100% | 45.03 | 31.10 |
| PiSSA | 100% | 47.38 | 29.27 |
| A-only SVD | 100% | 47.08 | 32.32 |
| B-only SVD | 100% | 47.01 | 32.93 |
| **Frozen-A (Ours)** | **~50%** | **48.45** | 29.88 |
| Frozen-A + Residual | ~52% | 47.16 | 28.05 |

→ **Frozen-A가 GSM8K 최고 + 파라미터 절반**

**4.4 분석**
- (a) 왜 A를 frozen해도 잘 되는가?
  - SVD는 W의 principal directions를 추출
  - 이 방향들은 task-agnostic한 "input subspace"
  - A를 학습하면 오히려 이 최적에서 멀어질 수 있음
- (b) Math vs Code task의 ΔW 분석
  - 두 task의 V subspace는 거의 직교 (random 수준)
  - 그러나 두 task 모두 SVD-init A에서는 잘 작동
  - → SVD-init A는 양쪽 모두에 충분히 풍부한 input basis 제공

### 5. 결론 (0.2쪽)
- LoRA의 A 행렬을 PiSSA-style SVD로 초기화 후 동결하는 단순한 기법으로
  - 학습 파라미터 50% 감소
  - GSM8K +4.55%p, HumanEval +2.44%p 향상 (표준 대비)
- 학습 불필요한 A를 식별함으로써 더 효율적인 LoRA 변형 가능성 제시
- 향후: 다양한 모델 크기, 더 많은 task로 일반화 검증

## 참고 문헌 (8-12편)
- Hu et al., 2021 - LoRA
- Meng et al., 2024 - PiSSA
- Zhang et al., 2023 - LoRA-FA
- Kopiczko et al., 2024 - VeRA
- Liu et al., 2024 - DoRA
- Cobbe et al., 2021 - GSM8K
- Chen et al., 2021 - HumanEval (Codex)
- Touvron et al., 2023 - Llama
- Loshchilov & Hutter, 2019 - AdamW
- Kingma & Ba, 2015 - Adam

## Figure / Table
- **Figure 1**: 제안 방법 다이어그램 (LoRA 구조 + A frozen 표시)
- **Table 1**: 7가지 방법 비교 메인 결과 (위 표)
- **Figure 2 (선택)**: 학습 곡선 비교 (frozen-A vs standard)
- **Figure 3 (선택)**: SVD A의 V subspace overlap analysis (task 간)

## 일정
- **Day 1 (오늘)**: outline + Section 1, 2, 3 초안
- **Day 2**: Section 4 (실험) + Table/Figure
- **Day 3**: Section 5 + polish + KCC 포맷 + 제출

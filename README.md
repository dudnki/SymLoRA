# SymLoRA

LoRA 학습 기하학(geometry) 분석과 A-Anchor Init에 관한 연구 코드입니다.

> **한 줄 요약**: LoRA의 A와 B는 init에서 ~2%만 움직이며, 그 움직임은 init subspace에
> orthogonal하고 task-specific하다. A의 init direction(SVD)이 성능을 결정하고,
> B의 init direction은 무관하다.

---

## Repo 구조

```
SymLoRA/
├── src/
│   ├── models/sym_lora.py  # 6개 init 모드 (standard/symmetric/pissa/a_only_svd/b_only_svd) + freeze_a
│   ├── training/sft.py     # SFT trainer
│   ├── data/loaders.py     # GSM8K, CodeAlpaca 로더
│   └── evaluation/metrics.py  # GSM8K accuracy, HumanEval pass@1
├── scripts/
│   ├── train_sft.py            # 학습 entrypoint (Hydra)
│   ├── eval.py                 # 평가
│   ├── prepare_dataset.py
│   ├── aggregate_results.py    # MLflow 결과 → 표 집계
│   ├── plot_w_rel.py           # Wrel vs 성능 산점도 (그림 1)
│   ├── anchor_freedom_analysis.py  # rel(A), rel(B), Wrel 계산 (표 2)
│   ├── smoke_test.py           # sweep 시작 전 setup 검증
│   └── run_all_experiments.sh  # multi-seed sweep
├── configs/train_sft.yaml
├── pyproject.toml
└── requirements.txt
```

> Git에서 제외: `outputs/`, `cache/`, `logs/`, `mlflow.db`, `unsloth_compiled_cache/`,
> `.venv/`, `data/` symlink. 모두 재생성 가능합니다.

---

## 환경

- Python 3.13 (`.python-version`)
- CUDA 12.1 환경 가정 (unsloth 의존)
- 학습은 단일 GPU(예: RTX 4070 Ti SUPER 16GB) 가정

### 설치

```bash
uv venv .venv --python 3.13      # 또는 python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 데이터 준비

GSM8K, CodeAlpaca-20k, Commonsense-170k은 HuggingFace에서 가져옵니다.

```bash
python scripts/prepare_dataset.py
```

기본적으로 `cache/`에 저장됩니다. 외부 디렉터리(예: 다른 프로젝트와 공유) 사용 시
프로젝트 루트에 `data` symlink를 만들어 쓸 수 있습니다.

---

## 학습

[configs/train_sft.yaml](configs/train_sft.yaml)에 모든 hyperparameter가 정의돼
있고, Hydra override 문법으로 변경합니다.

```bash
# math, standard init (baseline)
python scripts/train_sft.py

# math, symmetric init
python scripts/train_sft.py init.mode=symmetric

# code, PiSSA init
python scripts/train_sft.py init.mode=pissa data.task=code

# Frozen-A (논문 제안): A=SVD anchor freeze + B만 학습
python scripts/train_sft.py init.mode=a_only_svd init.freeze_a=true

# LoRA-FA baseline: A=Kaiming freeze + B만 학습
python scripts/train_sft.py init.mode=standard init.freeze_a=true
```

### Init 모드

| Mode | 설명 |
|---|---|
| `standard` | PEFT 기본 (A ~ Kaiming, B = 0) |
| `symmetric` | A, B 둘 다 nonzero, gradient-balanced. `W_eff = W - sBA`로 t=0 보존 |
| `pissa` | A, B를 W의 top-r SVD로 초기화 |
| `a_only_svd` | A direction만 SVD (B는 random) |
| `b_only_svd` | B direction만 SVD (A는 random) |

---

## 평가

```bash
# GSM8K
python scripts/eval.py --adapter outputs/standard/math_sft --task math

# HumanEval
python scripts/eval.py --adapter outputs/standard/code_sft --task code
```

### 결과 집계 / 분석

```bash
python scripts/aggregate_results.py        # mlflow.db에서 메트릭 모으기 → 표 1
python scripts/anchor_freedom_analysis.py  # rel(A), rel(B), Wrel 계산 → 표 2
python scripts/plot_w_rel.py               # Wrel vs 성능 산점도 → 그림 1
```

### 전체 실험 reproduction

```bash
bash scripts/run_all_experiments.sh
```

(다수의 seed × init mode × task 조합을 순차 실행)

---

## 실험 추적

학습 중 모든 메트릭은 MLflow(`sqlite:///mlflow.db`)에 기록됩니다.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`mlflow.db` 자체는 git에서 제외됩니다.

---

## 라이선스

(작성 예정)

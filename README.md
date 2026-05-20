# SymLoRA

Code for the paper *"Efficient LoRA Fine-tuning via SVD Initialization and Frozen A Matrix"* (KCC 2026).

We initialize the LoRA `A` matrix with the truncated SVD of the pretrained weight `W` and **freeze** it during fine-tuning, training only `B`. The result is a LoRA variant with **half the trainable parameters of standard LoRA** that also outperforms it.

## Main results

Llama-3.2-3B, LoRA rank 16, multi-seed mean (3 seeds).

| Method            | Trainable %  | GSM8K | HumanEval |
| ----------------- | -----------: | ----: | --------: |
| Standard LoRA     | 100%         | 43.62 |     27.85 |
| LoRA-FA           |  51%         | 33.31 |     25.41 |
| PiSSA             | 100%         | 47.49 |     29.07 |
| A-only SVD        | 100%         | 47.99 |     30.49 |
| B-only SVD        | 100%         | 46.55 |     30.08 |
| **Frozen-A (ours)** | **51%**    | 47.16 |     29.47 |

Standard LoRA 대비 학습 파라미터 49% 감소, GSM8K +3.54%p, HumanEval +1.62%p.
무작위 anchor를 동결하는 LoRA-FA보다 GSM8K +13.85%p로, 동결 자체가 아니라 *무엇을 동결하느냐*가 중요함을 보인다.

## Setup

학습은 단일 GPU(16GB+) + CUDA 12.1 환경을 가정합니다. RTX 4070 Ti SUPER에서 검증되었습니다.

```bash
git clone https://github.com/dudnki/SymLoRA.git
cd SymLoRA

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.13 / `unsloth` / `peft` / `trl` / `mlflow` / `hydra-core`.

### Data

GSM8K, CodeAlpaca-20k는 HuggingFace에서 자동으로 받습니다.

```bash
python scripts/prepare_dataset.py
```

기본 캐시 위치는 `cache/`. 다른 프로젝트와 데이터셋을 공유하려면 프로젝트 루트에
`data` symlink를 만들면 됩니다.

## Quick start

설정은 [configs/train_sft.yaml](configs/train_sft.yaml)에 모여 있고, Hydra override로 바꿉니다.

```bash
# Standard LoRA (baseline)
python scripts/train_sft.py

# PiSSA
python scripts/train_sft.py init.mode=pissa

# Frozen-A (proposed)
python scripts/train_sft.py init.mode=a_only_svd init.freeze_a=true

# LoRA-FA (Kaiming A frozen)
python scripts/train_sft.py init.mode=standard init.freeze_a=true

# Code task
python scripts/train_sft.py init.mode=a_only_svd init.freeze_a=true data.task=code
```

### Init modes

| `init.mode`   | A                 | B           | Notes                                 |
| ------------- | ----------------- | ----------- | ------------------------------------- |
| `standard`    | Kaiming           | 0           | PEFT default                          |
| `symmetric`   | scaled nonzero    | scaled nonzero | gradient-balanced, `W_eff = W - sBA` |
| `pissa`       | √Σ · Vᵀ           | U · √Σ      | both trainable                        |
| `a_only_svd`  | √Σ · Vᵀ           | random      | with `freeze_a=true` → Frozen-A       |
| `b_only_svd`  | random            | U · √Σ      |                                       |

## Evaluation

```bash
# GSM8K
python scripts/eval.py --adapter outputs/a_only_svd/math_sft --task math

# HumanEval
python scripts/eval.py --adapter outputs/a_only_svd/code_sft --task code
```

## Reproducing the paper

전체 sweep (init mode × task × seed) 실행:

```bash
bash scripts/run_all_experiments.sh
```

표/그림 생성:

```bash
python scripts/aggregate_results.py        # Table 1
python scripts/anchor_freedom_analysis.py  # Table 2 (rel_A, rel_B, W_rel)
python scripts/plot_w_rel.py               # Figure 1
```

학습 메트릭은 MLflow(`sqlite:///mlflow.db`)에 저장됩니다.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Repo layout

```
src/
  models/sym_lora.py       6 init modes + freeze_a
  training/sft.py          SFT trainer
  data/loaders.py          GSM8K / CodeAlpaca / Commonsense
  evaluation/metrics.py    GSM8K acc, HumanEval pass@1
scripts/
  train_sft.py             Hydra entrypoint
  eval.py
  prepare_dataset.py
  aggregate_results.py
  anchor_freedom_analysis.py
  plot_w_rel.py
  run_all_experiments.sh
  smoke_test.py
configs/train_sft.yaml
```

`outputs/`, `cache/`, `logs/`, `mlflow.db`, `.venv/`, `unsloth_compiled_cache/`는 모두
재생성 가능하므로 git에서 제외됩니다.

## Citation

```bibtex
@inproceedings{symlora2026,
  title  = {Efficient LoRA Fine-tuning via SVD Initialization and Frozen A Matrix},
  author = {Anonymous},
  booktitle = {Proceedings of KCC},
  year   = {2026}
}
```

## Acknowledgements

LoRA [Hu et al., 2021], PiSSA [Meng et al., 2024], LoRA-FA [Zhang et al., 2023], LoRI [Zhang et al., 2024]의 구현을 참고하였다. 학습 백엔드로 [unsloth](https://github.com/unslothai/unsloth)를 사용한다.

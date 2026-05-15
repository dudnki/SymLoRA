"""
prepare_dataset.py
==================
Math / Code SFT 데이터 준비 및 저장

사용법:
    python scripts/prepare_dataset.py --task math
    python scripts/prepare_dataset.py --task code
    python scripts/prepare_dataset.py --task math --n_samples 5000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loaders import TASK_CONFIG

SAVE_DIR = Path("data/processed")


def run(task: str, n_samples: int):
    if task not in TASK_CONFIG:
        print(f"[오류] 지원하지 않는 태스크: {task}")
        print(f"       지원 태스크: {list(TASK_CONFIG.keys())}")
        return

    cfg = TASK_CONFIG[task]
    print(f"\n{'='*50}")
    print(f"태스크: {task} — {cfg['description']}")
    print(f"{'='*50}")

    # loaders는 tokenizer를 받지만 base 모델에서는 포맷에 사용 안 함
    ds = cfg["loader"](tokenizer=None, n_samples=n_samples)

    # 길이 필터 (prompt + completion 합계 기준)
    before = len(ds)
    ds = ds.filter(
        lambda x: 50 < len(x["prompt"]) + len(x["completion"]) < 4000,
        desc="길이 필터",
    )
    print(f"길이 필터 후: {len(ds):,}개 (제거: {before - len(ds):,}개)")

    # train/test 분할 (95:5)
    split = ds.train_test_split(test_size=0.05, seed=42)
    print(f"train: {len(split['train']):,}  /  test: {len(split['test']):,}")

    # 샘플 미리보기
    print("\n--- 샘플 미리보기 ---")
    sample = split["train"][0]
    preview = sample["prompt"] + sample["completion"]
    print(preview[:400] + "..." if len(preview) > 400 else preview)

    # 저장
    save_path = SAVE_DIR / task
    save_path.mkdir(parents=True, exist_ok=True)
    split.save_to_disk(str(save_path))
    print(f"\n저장 완료: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASK_CONFIG.keys()))
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()

    n = args.n_samples or TASK_CONFIG[args.task]["default_n"]
    run(args.task, n)

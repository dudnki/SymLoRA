"""
aggregate_results.py
====================
outputs/multiseed/<config>/<task>_seed<seed>/eval_results.json 을 모두 읽어서
평균 ± std 표를 출력한다.

사용:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --markdown    # markdown 표
    python scripts/aggregate_results.py --latex       # LaTeX 표 (paper 붙여넣기용)
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).parent.parent
MULTISEED_DIR = ROOT / "outputs" / "multiseed"

# 출력 순서 고정 (paper 표 순서)
CONFIGS = ["standard", "lora_fa", "pissa", "a_only_svd", "b_only_svd", "frozen_a"]
DISPLAY = {
    "standard":   "Standard LoRA",
    "lora_fa":    "LoRA-FA",
    "pissa":      "PiSSA",
    "a_only_svd": "A-only SVD",
    "b_only_svd": "B-only SVD",
    "frozen_a":   "Frozen-A (Ours)",
}
TASKS = ["math", "code"]
TASK_METRIC = {"math": ("gsm8k", "accuracy"), "code": ("humaneval", "pass@1")}


def collect(config: str, task: str) -> dict[int, float]:
    """{seed: score(0~1)} dict 반환."""
    out = {}
    cfg_dir = MULTISEED_DIR / config
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

        eval_path = sub / "eval_results.json"
        if not eval_path.exists():
            continue
        try:
            d = json.loads(eval_path.read_text())
        except json.JSONDecodeError:
            continue

        bench, key = TASK_METRIC[task]
        score = d.get(bench, {}).get(key)
        if score is None:
            continue
        out[seed] = float(score)
    return out


def fmt_cell(scores: dict[int, float]) -> str:
    if not scores:
        return "—"
    vals = list(scores.values())
    pct = [v * 100 for v in vals]
    if len(pct) == 1:
        return f"{pct[0]:.2f}"
    mean = statistics.mean(pct)
    std = statistics.stdev(pct) if len(pct) >= 2 else 0.0
    return f"{mean:.2f} ± {std:.2f}"


def fmt_seeds(scores: dict[int, float]) -> str:
    if not scores:
        return ""
    return ", ".join(f"s{s}={v*100:.2f}" for s, v in sorted(scores.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--latex", action="store_true")
    args = ap.parse_args()

    # 데이터 수집
    grid = {cfg: {task: collect(cfg, task) for task in TASKS} for cfg in CONFIGS}

    # ── 1) 메인 표 (mean ± std) ──
    headers = ["Method", "GSM8K (%)", "HumanEval (%)", "n seeds"]
    rows = []
    for cfg in CONFIGS:
        math_scores = grid[cfg]["math"]
        code_scores = grid[cfg]["code"]
        n_seeds = max(len(math_scores), len(code_scores))
        rows.append([
            DISPLAY[cfg],
            fmt_cell(math_scores),
            fmt_cell(code_scores),
            str(n_seeds),
        ])

    if args.markdown:
        print(f"| {' | '.join(headers)} |")
        print(f"| {' | '.join(['---'] * len(headers))} |")
        for r in rows:
            print(f"| {' | '.join(r)} |")
    elif args.latex:
        print("\\begin{tabular}{lrrr}")
        print("\\toprule")
        print(" & ".join(headers) + " \\\\")
        print("\\midrule")
        for r in rows:
            print(" & ".join(r) + " \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")
    else:
        # 평문 표
        col_w = [max(len(h), max(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        sep = "  "
        line = sep.join(h.ljust(col_w[i]) for i, h in enumerate(headers))
        print(line)
        print("-" * len(line))
        for r in rows:
            print(sep.join(r[i].ljust(col_w[i]) for i in range(len(r))))

    # ── 2) seed별 상세 (debug용) ──
    print()
    print("[per-seed details]")
    for cfg in CONFIGS:
        for task in TASKS:
            s = grid[cfg][task]
            if s:
                print(f"  {cfg:14s} {task:5s}: {fmt_seeds(s)}")
            else:
                print(f"  {cfg:14s} {task:5s}: (no results)")


if __name__ == "__main__":
    main()

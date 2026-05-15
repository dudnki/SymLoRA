"""
plot_w_rel.py — paper Figure 1 생성
W_rel vs task 성능 (math, code) 상관관계 plot.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

# 깔끔한 폰트 (영어만)
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.7,
})

# anchor_freedom_analysis.py 결과 + aggregate_results.py 결과
methods = ["PiSSA", "A-only SVD", "B-only SVD", "Frozen-A"]
w_rel_math = [1.34, 2.39, 2.56, 1.70]   # %
w_rel_code = [1.38, 2.58, 2.71, 1.74]   # %
gsm8k = [47.49, 47.99, 46.55, 47.16]    # %
humaneval = [29.07, 30.49, 30.08, 29.47]  # %

colors = {
    "PiSSA":       "#d62728",  # red — over-anchored
    "A-only SVD":  "#1f77b4",  # blue
    "B-only SVD":  "#2ca02c",  # green
    "Frozen-A":    "#9467bd",  # purple — ours
}
markers = {
    "PiSSA":       "s",
    "A-only SVD":  "^",
    "B-only SVD":  "v",
    "Frozen-A":    "o",
}

# column width (~85mm = 3.35 inch). 2 panels을 가로로.
# 가로비를 줄여서 column에 들어갔을 때 height 더 크게 (라벨 가독성 ↑).
fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.0))

def plot_panel(ax, x, y, xlabel, ylabel, title):
    for i, m in enumerate(methods):
        ax.scatter(
            x[i], y[i],
            color=colors[m], marker=markers[m],
            s=70, label=m, edgecolor="black", linewidth=0.6,
            zorder=3,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle=":")
    # x축 약간 여유
    xmin, xmax = min(x), max(x)
    ax.set_xlim(xmin - 0.25, xmax + 0.25)
    ax.legend(loc="best", fontsize=6.5, framealpha=0.85)

plot_panel(axes[0], w_rel_math, gsm8k,
           r"$W_{\mathrm{rel}}$ (%)", "GSM8K (%)",
           "(a) Math: anchor strength vs reasoning")
plot_panel(axes[1], w_rel_code, humaneval,
           r"$W_{\mathrm{rel}}$ (%)", "HumanEval pass@1 (%)",
           "(b) Code: anchor strength vs generation")

plt.tight_layout()

out_dir = Path(__file__).parent.parent / "paper" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "w_rel_correlation.pdf", bbox_inches="tight")
fig.savefig(out_dir / "w_rel_correlation.png", dpi=200, bbox_inches="tight")
print(f"Saved → {out_dir}/w_rel_correlation.{{pdf,png}}")

"""Grokking scaling plot: eval accuracy vs step for n=4, n=8, n=16 runs.

Usage:
    python3 analysis/plot_scaling.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS = [
    ("runs/R000h_n4_init0.1/metrics.json",       "n=4 (q=7, h=2)",  "C0"),
    ("runs/R015_lwe_sparse_n8/metrics.json",     "n=8 (q=7, h=2)",  "C1"),
    ("runs/R013_modeB_n16_patience/metrics.json","n=16 (q=7, h=4)", "C3"),
]


def main():
    fig, ax = plt.subplots(figsize=(9, 5))

    for path, label, color in RUNS:
        p = Path(path)
        if not p.exists():
            print(f"skip {p} (missing)")
            continue
        with open(p) as f:
            m = json.load(f)
        # Use soft-gate eval for sparse, else eval_acc_soft
        if "eval_acc_soft" in m:
            acc = m["eval_acc_soft"]
        elif "eval_acc" in m:
            acc = m["eval_acc"]
        else:
            print(f"skip {p} — no eval_acc key")
            continue
        ax.plot(m["steps"], acc, "-", color=color, label=label)

    ax.axhline(0.5, linestyle=":", color="gray", alpha=0.5, label="chance")
    ax.set_xscale("log")
    ax.set_xlabel("step (log scale)")
    ax.set_ylabel("eval accuracy")
    ax.set_ylim(0.45, 1.05)
    ax.set_title("LWE Mode B grokking: scaling with n (fixed 2L d=64)")
    ax.legend()
    ax.grid(alpha=0.3)

    out = Path("runs/scaling_curve.png")
    plt.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

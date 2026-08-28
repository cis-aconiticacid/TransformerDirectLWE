"""Plot training curves from a run's metrics.json.

Usage:
    python3 analysis/plot_metrics.py runs/R000_sanity
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=str)
    p.add_argument("--out", type=str, default=None,
                   help="output PNG path (default: <run_dir>/curves.png)")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    with open(run_dir / "metrics.json") as f:
        m = json.load(f)

    out_path = Path(args.out) if args.out else run_dir / "curves.png"

    has_sparsity = "density_hard" in m and any(d is not None for d in m["density_hard"])

    if has_sparsity:
        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    steps = m["steps"]

    # Panel 1: loss
    axes[0].plot(steps, m["train_loss"], "-", color="C0", label="train CE")
    axes[0].axhline(0.693, color="gray", linestyle=":", alpha=0.5, label="log(2)")
    axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title(f"{run_dir.name}")

    # Panel 2: accuracy (LWE: eval_acc_soft/hard; SWIFFT: eval_coef_soft/hard)
    if "eval_acc_soft" in m:
        axes[1].plot(steps, m["eval_acc_soft"], "o-", color="C1", label="eval acc (soft gate)")
        if has_sparsity and "eval_acc_hard" in m:
            hard = [h for h in m["eval_acc_hard"] if h is not None]
            if hard:
                axes[1].plot(steps, m["eval_acc_hard"], "s-", color="C3", label="eval acc (hard prune)")
        axes[1].set_ylabel("eval accuracy")
    elif "eval_coef_soft" in m:
        axes[1].plot(steps, m["eval_coef_soft"], "o-", color="C1", label="coef acc (soft)")
        if has_sparsity and "eval_coef_hard" in m:
            axes[1].plot(steps, m["eval_coef_hard"], "s-", color="C3", label="coef acc (hard)")
        axes[1].set_ylabel("per-coef accuracy")
    axes[1].axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="chance")
    axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_ylim(0, 1.05)

    # Panel 3: density + lambda (sparse runs only)
    if has_sparsity:
        axes[2].plot(steps, m["density_expected"], "-", color="C2", label="density (expected)")
        axes[2].plot(steps, m["density_hard"], "--", color="C4", label="density (hard)")
        axes[2].set_ylabel("density"); axes[2].set_yscale("log")
        axes[2].set_xlabel("step")
        ax_r = axes[2].twinx()
        ax_r.plot(steps, m["lambda"], ":", color="C5", alpha=0.7, label="λ")
        ax_r.set_ylabel("lambda", color="C5")
        axes[2].legend(loc="upper left"); axes[2].grid(alpha=0.3)
    else:
        axes[1].set_xlabel("step")

    plt.tight_layout()
    fig.savefig(out_path, dpi=110)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()

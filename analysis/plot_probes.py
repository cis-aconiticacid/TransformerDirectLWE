"""Plot linear-probe accuracy per residual-stream layer.

Usage:
    python3 analysis/plot_probes.py runs/R000f_lwe_sparse_n4/eval_report.json
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
    p.add_argument("report_json", type=str)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    report = Path(args.report_json)
    with open(report) as f:
        data = json.load(f)

    probes = data["probes_heldout"]
    layers = list(probes.keys())
    kinds = ["inner", "diff", "bit"]

    fig, ax = plt.subplots(figsize=(9, 5))
    for kind, c in zip(kinds, ["C0", "C2", "C3"]):
        vals = [probes[l][kind]["test"] for l in layers]
        ax.plot(range(len(layers)), vals, "o-", label=kind, color=c)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=30, ha="right")
    ax.set_ylabel("held-out probe accuracy")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, linestyle=":", color="gray", alpha=0.4, label="bit-chance")
    ax.axhline(1.0 / 7, linestyle=":", color="black", alpha=0.3, label="q-chance (1/7)")
    ax.grid(alpha=0.3)
    ax.set_title(f"Mechanistic probe progression — {report.parent.name}")
    ax.legend()
    plt.tight_layout()
    out = Path(args.out) if args.out else report.parent / "probe_progression.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

"""Training entry point for LWE Mini.

Usage:
    python -m lwe_mini.train --config configs/sanity.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# allow running as "python -m lwe_mini.train" or "python lwe_mini/train.py"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.l0 import (                                # noqa: E402
    L0Linear, l0_density, overall_density, hard_density, set_hard_eval
)
from common.transformer import TransformerEncoder      # noqa: E402
from lwe_mini.data import LWEMiniConfig, make_batch, make_fixed_eval  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--override", type=str, nargs="*", default=[],
                   help="key=value pairs to override config entries")
    return p.parse_args()


def load_config(path: str, overrides: list[str]) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for kv in overrides:
        k, v = kv.split("=", 1)
        try:
            v_parsed = yaml.safe_load(v)
        except yaml.YAMLError:
            v_parsed = v
        cfg[k] = v_parsed
    return cfg


def lambda_schedule(step: int, cfg: dict) -> float:
    """Piecewise-linear λ schedule for L0 penalty weight."""
    if not cfg.get("use_sparsity", False):
        return 0.0
    if cfg.get("constant_lambda", False):
        return float(cfg["lambda_max"])
    warmup = int(cfg["lambda_warmup_steps"])
    ramp_end = warmup + int(cfg["lambda_ramp_steps"])
    lam_max = float(cfg["lambda_max"])
    if step < warmup:
        return 0.0
    if step < ramp_end:
        frac = (step - warmup) / max(1, ramp_end - warmup)
        return lam_max * frac
    return lam_max


def cosine_lr(step: int, total: int, base: float, warmup: int = 500) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * t))


def _accuracy(model, tokens, labels, cls_pos, batch_size: int = 512) -> float:
    total = 0
    correct = 0
    with torch.no_grad():
        for start in range(0, tokens.size(0), batch_size):
            end = min(start + batch_size, tokens.size(0))
            logits = model(tokens[start:end], cls_pos=cls_pos)
            pred = logits.argmax(dim=-1)
            correct += (pred == labels[start:end]).sum().item()
            total += end - start
    return correct / total


def evaluate(model, tokens, labels, cls_pos, use_sparsity: bool,
             batch_size: int = 512) -> tuple[float, float]:
    """Return (soft_gate_acc, hard_prune_acc). Second value is None when sparsity is off."""
    model.eval()
    soft_acc = _accuracy(model, tokens, labels, cls_pos, batch_size)
    hard_acc = None
    if use_sparsity:
        set_hard_eval(model, True)
        hard_acc = _accuracy(model, tokens, labels, cls_pos, batch_size)
        set_hard_eval(model, False)
    model.train()
    return soft_acc, hard_acc


def main():
    args = parse_args()
    cfg = load_config(args.config, args.override)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    # persist resolved config
    with open(out_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    data_rng = np.random.default_rng(cfg["seed"])

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    lwe = LWEMiniConfig(
        n=cfg["n"], q=cfg["q"], hamming=cfg["hamming"],
        e_bound=cfg["e_bound"],
        mode=cfg.get("mode", "A"),
        fixed_s_seed=cfg.get("fixed_s_seed", 7),
    )
    model = TransformerEncoder(
        vocab_size=lwe.vocab_size,
        seq_len=lwe.seq_len,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        n_layers=cfg["n_layers"],
        use_mask=cfg.get("use_sparsity", False),
        out_dim=2,
        per_position_out=False,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=(0.9, 0.95),
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    # fixed eval set
    eval_tokens, eval_labels = make_fixed_eval(
        cfg["eval_size"], lwe, seed=cfg["seed"] + 999, device=device,
    )
    cls_pos = lwe.seq_len - 1

    total_steps = int(cfg["steps"])
    batch_size = int(cfg["batch_size"])
    log_every = int(cfg.get("log_every", 200))
    eval_every = int(cfg.get("eval_every", 1000))

    use_sparsity = cfg.get("use_sparsity", False)
    metrics = {"steps": [], "train_loss": [], "eval_acc_soft": [],
               "eval_acc_hard": [], "density_expected": [], "density_hard": [],
               "lambda": []}
    t0 = time.time()
    for step in range(total_steps):
        lr = cosine_lr(step, total_steps, cfg["lr"], warmup=cfg.get("warmup", 500))
        for g in opt.param_groups:
            g["lr"] = lr

        tokens, labels = make_batch(batch_size, lwe, data_rng, device=device)
        logits = model(tokens, cls_pos=cls_pos)
        ce = F.cross_entropy(logits, labels)
        lam = lambda_schedule(step, cfg)
        if lam > 0:
            # normalized density penalty ∈ [0,1] so lambda_max is CE-scale, param-count-independent
            loss = ce + lam * l0_density(model)
        else:
            loss = ce

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.get("clip", 1.0))
        opt.step()

        if step % log_every == 0 or step == total_steps - 1:
            d_e = overall_density(model) if use_sparsity else 1.0
            d_h = hard_density(model) if use_sparsity else 1.0
            print(
                f"step={step:6d}  loss={ce.item():.4f}  "
                f"lambda={lam:.5f}  density_exp={d_e:.5f}  density_hard={d_h:.5f}  "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

        if step % eval_every == 0 or step == total_steps - 1:
            soft_acc, hard_acc = evaluate(
                model, eval_tokens, eval_labels, cls_pos, use_sparsity
            )
            d_e = overall_density(model) if use_sparsity else 1.0
            d_h = hard_density(model) if use_sparsity else 1.0
            metrics["steps"].append(step)
            metrics["train_loss"].append(float(ce.item()))
            metrics["eval_acc_soft"].append(soft_acc)
            metrics["eval_acc_hard"].append(hard_acc)
            metrics["density_expected"].append(d_e)
            metrics["density_hard"].append(d_h)
            metrics["lambda"].append(lam)
            if use_sparsity:
                print(
                    f"  >> soft_acc={soft_acc:.4f}  hard_acc={hard_acc:.4f}  "
                    f"density_exp={d_e:.5f}  density_hard={d_h:.5f}",
                    flush=True,
                )
            else:
                print(f"  >> eval_acc={soft_acc:.4f}", flush=True)

    # final snapshot
    torch.save(
        {"model": model.state_dict(), "cfg": cfg},
        out_dir / "checkpoint.pt",
    )
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    if use_sparsity:
        print(
            f"done. soft_acc={metrics['eval_acc_soft'][-1]:.4f}  "
            f"hard_acc={metrics['eval_acc_hard'][-1]:.4f}  "
            f"density_hard={metrics['density_hard'][-1]:.5f}"
        )
    else:
        print(f"done. final eval_acc={metrics['eval_acc_soft'][-1]:.4f}")


if __name__ == "__main__":
    main()

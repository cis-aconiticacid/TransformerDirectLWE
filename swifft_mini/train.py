"""Training entry point for SWIFFT Mini.

Per-coefficient cross-entropy over the n CLS output positions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.l0 import (                                # noqa: E402
    l0_density, overall_density, hard_density, set_hard_eval
)
from common.transformer import TransformerEncoder                  # noqa: E402
from swifft_mini.data import SWIFFTMiniConfig, make_batch, make_fixed_eval  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--override", type=str, nargs="*", default=[])
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


def lambda_schedule(step, cfg):
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


def cosine_lr(step, total, base, warmup=500):
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * t))


def _coef_acc(model, tokens, targets, cls_start, cls_len, batch_size=256):
    total = 0
    correct_coef = 0
    correct_full = 0
    with torch.no_grad():
        for start in range(0, tokens.size(0), batch_size):
            end = min(start + batch_size, tokens.size(0))
            logits = model(tokens[start:end])
            out_logits = logits[:, cls_start:cls_start + cls_len]
            pred = out_logits.argmax(dim=-1)
            tgt = targets[start:end]
            correct_coef += (pred == tgt).sum().item()
            correct_full += ((pred == tgt).all(dim=-1)).sum().item()
            total += end - start
    return correct_coef / (total * cls_len), correct_full / total


def evaluate(model, tokens, targets, cls_start, cls_len, use_sparsity,
             batch_size: int = 256):
    """Return (soft_coef, soft_full, hard_coef, hard_full). Hard-* are None when sparsity is off."""
    model.eval()
    soft_coef, soft_full = _coef_acc(model, tokens, targets, cls_start, cls_len, batch_size)
    hard_coef = hard_full = None
    if use_sparsity:
        set_hard_eval(model, True)
        hard_coef, hard_full = _coef_acc(model, tokens, targets, cls_start, cls_len, batch_size)
        set_hard_eval(model, False)
    model.train()
    return soft_coef, soft_full, hard_coef, hard_full


def main():
    args = parse_args()
    cfg = load_config(args.config, args.override)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "resolved_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    data_rng = np.random.default_rng(cfg["seed"])
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    sw = SWIFFTMiniConfig(m=cfg["m"], n=cfg["n"], p=cfg["p"],
                          mult_seed=cfg.get("mult_seed", 17))
    model = TransformerEncoder(
        vocab_size=sw.vocab_size,
        seq_len=sw.seq_len,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        d_ff=cfg["d_ff"],
        n_layers=cfg["n_layers"],
        use_mask=cfg.get("use_sparsity", False),
        out_dim=sw.p,
        per_position_out=True,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95),
        weight_decay=cfg.get("weight_decay", 0.0),
    )

    eval_tokens, eval_targets = make_fixed_eval(
        cfg["eval_size"], sw, seed=cfg["seed"] + 999, device=device,
    )
    cls_start = sw.m * sw.n + 1        # first CLS position
    cls_len = sw.n

    total_steps = int(cfg["steps"])
    batch_size = int(cfg["batch_size"])
    log_every = int(cfg.get("log_every", 200))
    eval_every = int(cfg.get("eval_every", 1000))

    use_sparsity = cfg.get("use_sparsity", False)
    metrics = {"steps": [], "train_loss": [],
               "eval_coef_soft": [], "eval_full_soft": [],
               "eval_coef_hard": [], "eval_full_hard": [],
               "density_expected": [], "density_hard": [], "lambda": []}
    t0 = time.time()

    for step in range(total_steps):
        lr = cosine_lr(step, total_steps, cfg["lr"], warmup=cfg.get("warmup", 500))
        for g in opt.param_groups:
            g["lr"] = lr

        tokens, targets = make_batch(batch_size, sw, data_rng, device=device)
        logits = model(tokens)
        out_logits = logits[:, cls_start:cls_start + cls_len]
        ce = F.cross_entropy(out_logits.reshape(-1, sw.p), targets.reshape(-1))
        lam = lambda_schedule(step, cfg)
        # normalized density penalty ∈ [0,1]; lambda_max is on CE-scale
        loss = ce + lam * l0_density(model) if lam > 0 else ce

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
            sc, sf, hc, hf = evaluate(
                model, eval_tokens, eval_targets, cls_start, cls_len, use_sparsity
            )
            d_e = overall_density(model) if use_sparsity else 1.0
            d_h = hard_density(model) if use_sparsity else 1.0
            metrics["steps"].append(step)
            metrics["train_loss"].append(float(ce.item()))
            metrics["eval_coef_soft"].append(sc)
            metrics["eval_full_soft"].append(sf)
            metrics["eval_coef_hard"].append(hc)
            metrics["eval_full_hard"].append(hf)
            metrics["density_expected"].append(d_e)
            metrics["density_hard"].append(d_h)
            metrics["lambda"].append(lam)
            if use_sparsity:
                print(
                    f"  >> soft_coef={sc:.4f} soft_full={sf:.4f}  "
                    f"hard_coef={hc:.4f} hard_full={hf:.4f}  "
                    f"density_exp={d_e:.5f}  density_hard={d_h:.5f}",
                    flush=True,
                )
            else:
                print(f"  >> coef_acc={sc:.4f}  full_acc={sf:.4f}", flush=True)

    torch.save(
        {"model": model.state_dict(), "cfg": cfg,
         "multipliers": sw.multipliers.tolist()},
        out_dir / "checkpoint.pt",
    )
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    if use_sparsity:
        print(
            f"done. soft_coef={metrics['eval_coef_soft'][-1]:.4f}  "
            f"hard_coef={metrics['eval_coef_hard'][-1]:.4f}  "
            f"density_hard={metrics['density_hard'][-1]:.5f}"
        )
    else:
        print(
            f"done. coef_acc={metrics['eval_coef_soft'][-1]:.4f}  "
            f"full_acc={metrics['eval_full_soft'][-1]:.4f}"
        )


if __name__ == "__main__":
    main()

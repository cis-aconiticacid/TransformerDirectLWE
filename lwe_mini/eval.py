"""Post-training eval + interpretability probes for LWE Mini.

Computes:
- Final eval accuracy on held-out set, both soft-gate and hard-pruned
- Per-L0Linear expected and realized density
- Hard-prune drop: force every weight whose expected gate is zero to literally
  zero; report accuracy drop (NOT activation-level mean ablation — we use the
  stronger, cleaner hard-prune diagnostic for L0 sparsity verification)
- Linear probes for <a, s> mod q, (b - <a,s>) mod q, decision bit — evaluated
  on a held-out split not seen during probe training
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.l0 import (                                              # noqa: E402
    L0Linear, overall_density, hard_density, set_hard_eval,
)
from common.transformer import TransformerEncoder                   # noqa: E402
from lwe_mini.data import LWEMiniConfig, make_fixed_eval            # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n_eval", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    lwe = LWEMiniConfig(
        cfg["n"], cfg["q"], cfg["hamming"], cfg["e_bound"],
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
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, lwe, cfg


def eval_accuracy(model, lwe, n_eval: int, seed: int, device: str,
                  hard: bool = False) -> float:
    tokens, labels = make_fixed_eval(n_eval, lwe, seed=seed, device=device)
    if hard:
        set_hard_eval(model, True)
    with torch.no_grad():
        logits = model(tokens, cls_pos=lwe.seq_len - 1)
        pred = logits.argmax(dim=-1)
    if hard:
        set_hard_eval(model, False)
    return (pred == labels).float().mean().item()


def hard_prune_drop(model, lwe, n_eval: int, seed: int, device: str) -> dict:
    """Accuracy under (a) normal soft-gate eval vs. (b) hard-pruned eval.

    Non-trivial drop indicates the learned circuit still leans on weights
    whose hard mask is zero (i.e., the L0 solution has not converged cleanly).
    """
    soft = eval_accuracy(model, lwe, n_eval, seed, device, hard=False)
    hard = eval_accuracy(model, lwe, n_eval, seed, device, hard=True)
    return {"acc_soft_gate": soft, "acc_hard_prune": hard, "drop": soft - hard}


@torch.no_grad()
def collect_activations(model, tokens, cls_pos):
    """Forward pass returning residual-stream snapshots per layer."""
    acts = {}
    x = model.tok(tokens) + model.pos_embed[: tokens.size(1)]
    acts["layer0_in"] = x.detach().clone()
    for i, block in enumerate(model.blocks):
        x = x + block.attn(block.norm1(x))
        acts[f"layer{i}_postattn"] = x.detach().clone()
        x = x + block.mlp(block.norm2(x))
        acts[f"layer{i}_out"] = x.detach().clone()
    x = model.norm_f(x)
    acts["final"] = x.detach().clone()
    return acts


def linear_probe_accuracy(
    feats, targets, n_classes, device, train_frac: float = 0.5,
    steps: int = 500, lr: float = 1e-2,
) -> dict:
    """Fit a linear probe on `train_frac` of (feats, targets) and report
    held-out accuracy (and train accuracy, for sanity). Returns {"test", "train"}."""
    feats = feats.to(device).float()
    targets = targets.to(device).long()
    n = feats.size(0)
    # deterministic split so probe runs are reproducible across invocations
    gen = torch.Generator(device="cpu").manual_seed(0)
    perm = torch.randperm(n, generator=gen).to(device)
    n_train = int(train_frac * n)
    tr, te = perm[:n_train], perm[n_train:]
    clf = nn.Linear(feats.size(-1), n_classes).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        logits = clf(feats[tr])
        loss = F.cross_entropy(logits, targets[tr])
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_acc = (clf(feats[tr]).argmax(-1) == targets[tr]).float().mean().item()
        test_acc = (clf(feats[te]).argmax(-1) == targets[te]).float().mean().item()
    return {"train": train_acc, "test": test_acc}


def interp_probes(model, lwe, n_eval: int, seed: int, device: str) -> dict:
    rng = np.random.default_rng(seed)
    # resample so we know (a, s, b) ground truth explicitly. For Mode B the
    # model has a fixed secret baked into its weights, so we MUST use that
    # same secret when generating probe data — otherwise b is encrypted with
    # the wrong s and the model output is essentially random.
    from lwe_mini.data import sample_secret, encrypt_bit
    tokens_arr = np.empty((n_eval, lwe.seq_len), dtype=np.int64)
    inner = np.empty(n_eval, dtype=np.int64)
    diff = np.empty(n_eval, dtype=np.int64)
    bits = np.empty(n_eval, dtype=np.int64)
    for i in range(n_eval):
        if lwe.mode == "A":
            s = sample_secret(lwe.n, lwe.hamming, rng)
        else:
            s = lwe.fixed_s
        m = int(rng.integers(0, 2))
        a, b = encrypt_bit(s, m, lwe, rng)
        tokens_arr[i] = lwe.tokenize(s, a, b)
        inner[i] = int(np.dot(a, s) % lwe.q)
        diff[i] = int((b - np.dot(a, s)) % lwe.q)
        bits[i] = m
    tokens = torch.from_numpy(tokens_arr).to(device)
    acts = collect_activations(model, tokens, lwe.seq_len - 1)
    cls_acts = {k: v[:, -1, :] for k, v in acts.items()}    # CLS position

    probe_results = {}
    for k, feat in cls_acts.items():
        probe_results[k] = {
            "inner": linear_probe_accuracy(
                feat, torch.from_numpy(inner), lwe.q, device
            ),
            "diff": linear_probe_accuracy(
                feat, torch.from_numpy(diff), lwe.q, device
            ),
            "bit": linear_probe_accuracy(
                feat, torch.from_numpy(bits), 2, device
            ),
        }
    return probe_results


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, lwe, cfg = load_model(args.checkpoint, device)

    use_sparsity = cfg.get("use_sparsity", False)

    acc_soft = eval_accuracy(model, lwe, args.n_eval, args.seed, device, hard=False)
    density_expected = overall_density(model) if use_sparsity else 1.0
    density_realized = hard_density(model) if use_sparsity else 1.0

    if use_sparsity:
        prune_info = hard_prune_drop(model, lwe, args.n_eval, args.seed, device)
    else:
        prune_info = None

    probes = interp_probes(model, lwe, n_eval=min(args.n_eval, 2000),
                           seed=args.seed + 1, device=device)

    results = {
        "eval_accuracy_soft_gate": acc_soft,
        "density_expected": density_expected,
        "density_hard": density_realized,
        "hard_prune_check": prune_info,
        "probes_heldout": probes,
        "probe_note": ("Each probe value is a dict with train/test accuracy; "
                       "test is reported on a held-out 50% split."),
    }
    print(json.dumps(results, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

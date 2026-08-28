"""Hard-Concrete gate for L0-regularized weight sparsity.

Reference: Louizos, Welling, Kingma 2018 "Learning Sparse Neural Networks through L0
Regularization" (ICLR). Used by Gao et al. 2025 for weight-sparse transformers.

Each weight W is reparameterized W_eff = W * mask, where mask is drawn from a
hard-concrete distribution stretched onto (gamma, zeta) and clamped to [0,1].
At train time mask is stochastic; at eval time mask is deterministic (median).
L0 penalty = sum_ij P(mask_ij > 0) is differentiable w.r.t. log_alpha.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HardConcreteMask(nn.Module):
    """Element-wise hard-concrete mask over a weight of given shape."""

    def __init__(
        self,
        shape,
        temperature: float = 2.0 / 3.0,
        stretch_low: float = -0.1,
        stretch_high: float = 1.1,
        init_log_alpha: float = 2.0,
    ):
        super().__init__()
        self.shape = tuple(shape)
        self.beta = temperature
        self.gamma = stretch_low
        self.zeta = stretch_high
        self.log_alpha = nn.Parameter(
            torch.full(self.shape, float(init_log_alpha))
        )

    def _stretched_sigmoid(self, u: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(
            (torch.log(u.clamp(min=1e-8)) - torch.log((1.0 - u).clamp(min=1e-8))
             + self.log_alpha) / self.beta
        )
        s = s * (self.zeta - self.gamma) + self.gamma
        return s.clamp(0.0, 1.0)

    def sample_mask(self) -> torch.Tensor:
        if self.training:
            u = torch.rand_like(self.log_alpha)
            return self._stretched_sigmoid(u)
        else:
            s = torch.sigmoid(self.log_alpha) * (self.zeta - self.gamma) + self.gamma
            return s.clamp(0.0, 1.0)

    def l0_penalty(self) -> torch.Tensor:
        """Expected number of non-zero gates: sum P(mask > 0).

        P(mask > 0) = sigmoid(log_alpha - beta * log(-gamma / zeta))
        for gamma<0<zeta, this equals sigmoid(log_alpha - beta * log(-gamma/zeta)).
        """
        shift = self.beta * math.log(-self.gamma / self.zeta)
        return torch.sigmoid(self.log_alpha - shift).sum()

    def density(self) -> torch.Tensor:
        """Deterministic density P(mask > 0) as a scalar in [0, 1]."""
        with torch.no_grad():
            shift = self.beta * math.log(-self.gamma / self.zeta)
            return torch.sigmoid(self.log_alpha - shift).mean()

    def hard_mask(self) -> torch.Tensor:
        """Deterministic binary mask for post-hoc pruning/analysis."""
        with torch.no_grad():
            s = torch.sigmoid(self.log_alpha) * (self.zeta - self.gamma) + self.gamma
            return (s > 0.0).float()

    def hard_nonzero(self) -> int:
        """Realized non-zero count under hard pruning (not expectation)."""
        with torch.no_grad():
            return int(self.hard_mask().sum().item())


class L0Linear(nn.Module):
    """Linear layer whose weight is multiplied by a hard-concrete mask.

    Bias is NOT gated — only the weight matrix is sparsified, matching Gao 2025.
    When `use_mask=False` the layer reduces to a standard nn.Linear.

    Eval behavior depends on `force_hard_eval` (see `set_hard_eval`):
      - False (default): soft-gate eval (continuous stretched-sigmoid in [0,1])
      - True: hard-prune eval (binary mask = 1 iff expected gate > 0)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        use_mask: bool = True,
        **mask_kwargs,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
            bound = 1.0 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)
        self.use_mask = use_mask
        self.force_hard_eval = False
        if use_mask:
            self.mask = HardConcreteMask(self.weight.shape, **mask_kwargs)
        else:
            self.mask = None

    def effective_weight(self) -> torch.Tensor:
        if not self.use_mask:
            return self.weight
        if not self.training and self.force_hard_eval:
            return self.weight * self.mask.hard_mask()
        return self.weight * self.mask.sample_mask()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)

    def l0_penalty(self) -> torch.Tensor:
        if self.use_mask:
            return self.mask.l0_penalty()
        return torch.zeros((), device=self.weight.device)

    def density(self) -> torch.Tensor:
        if self.use_mask:
            return self.mask.density()
        return torch.ones((), device=self.weight.device)


def collect_l0_penalty(module: nn.Module) -> torch.Tensor:
    total = None
    for m in module.modules():
        if isinstance(m, L0Linear) and m.use_mask:
            pen = m.l0_penalty()
            total = pen if total is None else total + pen
    if total is None:
        return torch.zeros((), device=next(module.parameters()).device)
    return total


def total_nonzero_gates(module: nn.Module) -> tuple[float, float]:
    """Return (expected_nonzero, total_weights) across all L0Linear layers.

    NOTE: `expected_nonzero` is the differentiable P(z>0) sum, not realized
    hard-pruned count. Use `total_hard_nonzero` for the realized number that
    matches what `force_hard_eval=True` would actually drop.
    """
    nz = 0.0
    total = 0.0
    for m in module.modules():
        if isinstance(m, L0Linear) and m.use_mask:
            nz += float(m.l0_penalty().detach().item())
            total += float(m.weight.numel())
    return nz, total


def total_hard_nonzero(module: nn.Module) -> tuple[int, int]:
    """Realized (hard_nonzero, total_weights) under the deterministic
    hard pruning that `force_hard_eval=True` applies."""
    nz = 0
    total = 0
    for m in module.modules():
        if isinstance(m, L0Linear) and m.use_mask:
            nz += m.mask.hard_nonzero()
            total += int(m.weight.numel())
    return nz, total


def overall_density(module: nn.Module) -> float:
    """Expected density (differentiable surrogate). O(1) ∈ [0, 1]."""
    nz, total = total_nonzero_gates(module)
    return nz / max(total, 1.0)


def hard_density(module: nn.Module) -> float:
    """Realized density under hard pruning."""
    nz, total = total_hard_nonzero(module)
    return nz / max(total, 1.0)


def set_hard_eval(module: nn.Module, flag: bool) -> None:
    """Toggle hard-prune eval behavior on all L0Linear layers in-place."""
    for m in module.modules():
        if isinstance(m, L0Linear):
            m.force_hard_eval = flag


def l0_density(module: nn.Module) -> torch.Tensor:
    """Differentiable expected density = collect_l0_penalty / total_gates.

    Training uses this so that lambda_max is on the O(1) CE scale,
    independent of the model's total parameter count.
    """
    pen = None
    total = 0
    for m in module.modules():
        if isinstance(m, L0Linear) and m.use_mask:
            p = m.l0_penalty()
            pen = p if pen is None else pen + p
            total += int(m.weight.numel())
    if pen is None:
        return torch.zeros((), device=next(module.parameters()).device)
    return pen / max(total, 1)

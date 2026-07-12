"""Fused low-rank linear layer: x → V → U (σ absorbed into factors)."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class LowRankLinear(nn.Module):
    """Drop-in ``nn.Linear`` replacement with two GEMMs and fused singular values.

    Factorization::

        W ≈ U @ diag(S) @ Vh   (W shape: out × in)
        stored as:
            V_weight: (r, in)  = Vh[:r]
            U_weight: (out, r) = U[:, :r] @ diag(S[:r])

    Forward is exactly ``U(V(x))`` with optional bias on ``U``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if rank > min(in_features, out_features):
            raise ValueError(
                f"rank={rank} exceeds min(in,out)=({in_features},{out_features})"
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)

        factory = {"device": device, "dtype": dtype}
        self.v = nn.Linear(self.in_features, self.rank, bias=False, **factory)
        self.u = nn.Linear(self.rank, self.out_features, bias=bias, **factory)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int,
        *,
        energy: Optional[torch.Tensor] = None,
    ) -> "LowRankLinear":
        """Build a fused low-rank layer from a dense ``nn.Linear`` via truncated SVD.

        Args:
            linear: Source dense layer (weight shape ``[out, in]``).
            rank: Retained rank ``r``.
            energy: Optional precomputed singular values (unused; reserved).
        """
        _ = energy
        w = linear.weight.detach().float()
        out_features, in_features = w.shape
        rank = int(min(rank, out_features, in_features))

        # W = U @ diag(S) @ Vh
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        u_r = u[:, :rank]
        s_r = s[:rank]
        vh_r = vh[:rank, :]

        # Fuse σ into U: U_hat = U * S, V_hat = Vh
        u_hat = u_r * s_r.unsqueeze(0)  # (out, r)
        v_hat = vh_r  # (r, in)

        module = cls(
            in_features,
            out_features,
            rank,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
            device=linear.weight.device,
        )
        with torch.no_grad():
            module.v.weight.copy_(v_hat.to(dtype=module.v.weight.dtype))
            module.u.weight.copy_(u_hat.to(dtype=module.u.weight.dtype))
            if linear.bias is not None and module.u.bias is not None:
                module.u.bias.copy_(linear.bias.detach().to(dtype=module.u.bias.dtype))
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.u(self.v(x))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, bias={self.u.bias is not None}"
        )

    @property
    def param_count(self) -> int:
        """Number of parameters in the factorized layer (incl. bias)."""
        n = self.v.weight.numel() + self.u.weight.numel()
        if self.u.bias is not None:
            n += self.u.bias.numel()
        return int(n)

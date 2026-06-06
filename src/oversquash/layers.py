"""
QuotientWalkConv — a multi-hop, kQ/I-aware aggregation layer.

This is the *faithful* realization of the over-squashing claim in ACT_en.tex
(Remark on the quotient algebra, §4.4): identify **functionally equivalent
paths before aggregating**, so that the number of messages reaching node j from
node i at range g is the de-duplicated walk count

    M_g[i, j] = dim(e_i · (kQ/I)_g · e_j)   ≤   (A^g)[i, j] = n_g(i, j),

rather than the raw walk count (A^g)[i, j]. Over-squashing arises precisely
because (A^g)[i, j] messages are compressed into one fixed-width vector; the
quotient lowers that count whenever paths are redundant.

Why a multi-hop layer (not a standard 1-hop MessagePassing): the multiplicity
n_g(i, j) is a property of length-g *walks*, i.e. of the g-fold composition of
the adjacency — it is invisible to a single 1-hop layer. (A 1-hop merge of a
node's in-edges under mean aggregation is idempotent and changes nothing — see
the design note in the repo history.) So this layer aggregates directly over
the precomputed range-g walk operator:

    h_j^{(g)} = σ( Σ_i  W_g[i, j] · Θ h_i  +  Θ_root h_j ),

where W_g is the row-normalized effective-walk matrix M_g (a quotient model) or
the raw walk matrix A^g (the ablation baseline). Both are precomputed per graph
by transforms.AttachWalkOperators and passed in as sparse tensors, so the heavy
path-algebra work (via aiq-quivers) happens once at data-prep time, not in the
training loop.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class QuotientWalkConv(nn.Module):
    """One range-g walk-aggregation layer.

    Parameters
    ----------
    in_channels, out_channels : int
    depth : int
        The range g this layer aggregates over (1-indexed). Used only for
        bookkeeping / repr; the actual operator is supplied per forward call.

    forward(x, walk_op):
        x       : (N, in_channels) node features.
        walk_op : (N, N) sparse FloatTensor — the row-normalized range-g
                  operator for this graph/batch. Use the EFFECTIVE matrix
                  (kQ/I) for the quotient model, or A^g for the ablation.
                  If None, falls back to identity (skip-only), which makes the
                  layer a plain MLP — a clean null model.
    """

    def __init__(self, in_channels: int, out_channels: int, depth: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.theta = nn.Linear(in_channels, out_channels, bias=False)
        self.root = nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.theta.reset_parameters()
        self.root.reset_parameters()

    def forward(self, x: torch.Tensor,
                walk_op: Optional[torch.Tensor] = None) -> torch.Tensor:
        msg = self.theta(x)                       # (N, out)
        if walk_op is None:
            agg = torch.zeros_like(msg)
        else:
            # walk_op is (N, N) with walk_op[j, i] = normalized weight of the
            # range-g walks i -> j. Sparse mm aggregates neighbors into j.
            agg = torch.sparse.mm(walk_op, msg)   # (N, out)
        return agg + self.root(x)

    def __repr__(self):  # pragma: no cover
        return (f"{self.__class__.__name__}({self.in_channels}->"
                f"{self.out_channels}, range={self.depth})")

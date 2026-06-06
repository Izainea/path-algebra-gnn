"""
QuotientMessagePassing — a kQ/I-aware aggregation layer.

Standard message passing aggregates one message per incoming walk; when many
walks share a (source, target) pair, their messages are squashed into a single
fixed-width vector. Our layer first *merges messages that travel along
functionally equivalent paths* — paths identified by an admissible ideal I —
and only then aggregates. This realizes, at the level of the learnable
operator, the structural bound

    dim(e_i · (kQ/I)_g · e_j)  ≤  dim(e_i · kQ_g · e_j) = (A^g)_{ij}

from ACT_en.tex (Prop. prop:efecto_ideal). Fewer, de-duplicated messages mean
less is compressed into the node embedding — the algebraic route to relieving
over-squashing, as opposed to rewiring or pooling.

Design note: the ideal lives over *walks*, which are a global, multi-hop
object, whereas a PyG layer is local (1-hop). We bridge the two by passing a
precomputed `QuotientPlan` (see ideal_bridge) that tells the layer which
incoming 1-hop messages at each node belong to the same equivalence class for
the current depth. At depth g the layer consults `plan` for length-g classes;
when no plan is supplied it degrades gracefully to mean aggregation (i.e. a
vanilla GCN-style layer), which is the right null model for ablations.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class QuotientMessagePassing(MessagePassing):
    """One quotient-aware message-passing layer.

    Parameters
    ----------
    in_channels, out_channels : int
    depth : int
        Which FNS layer g this instance represents (1-indexed). Used to look up
        length-g equivalence classes in the QuotientPlan.
    aggr : str
        Base aggregation applied AFTER class-merging ('mean' by default).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 depth: int = 1, aggr: str = "mean"):
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.lin = nn.Linear(in_channels, out_channels)
        self.root = nn.Linear(in_channels, out_channels)  # self-loop / skip
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.root.reset_parameters()

    def forward(self, x, edge_index, edge_class: Optional[torch.Tensor] = None):
        """
        x          : (N, in_channels)
        edge_index : (2, E)
        edge_class : (E,) long tensor, optional. The equivalence-class id of
                     each edge under I at this depth, as produced by
                     ideal_bridge for the current graph. Edges sharing a class
                     and a target node have their messages averaged into one
                     before aggregation (redundant-path collapse). When None,
                     every edge is its own class => plain mean aggregation.
        """
        msg = self.lin(x)
        out = self.propagate(edge_index, x=msg, edge_class=edge_class,
                             size=(x.size(0), x.size(0)))
        return out + self.root(x)

    def message(self, x_j, edge_class, index, size_i):
        """Collapse messages within an equivalence class, then pass on.

        We pre-average x_j over (target node, class) groups so that a class of
        k redundant parallel paths contributes the weight of a single message
        rather than k. This is the discrete analogue of replacing
        (A^g)_{ij} by dim(e_i·(kQ/I)_g·e_j).
        """
        if edge_class is None:
            return x_j
        # group key = (target node, class id) -> mean of x_j within the group
        # Build a compact group id from (index, edge_class).
        max_class = int(edge_class.max().item()) + 1 if edge_class.numel() else 1
        group = index * max_class + edge_class
        uniq, inv, counts = torch.unique(
            group, return_inverse=True, return_counts=True
        )
        summed = torch.zeros(uniq.size(0), x_j.size(-1),
                            device=x_j.device, dtype=x_j.dtype)
        summed.index_add_(0, inv, x_j)
        group_mean = summed / counts.unsqueeze(-1).clamp(min=1)
        # each edge now carries its group's de-duplicated message
        return group_mean[inv]

    def __repr__(self):  # pragma: no cover
        return (f"{self.__class__.__name__}({self.in_channels}->"
                f"{self.out_channels}, depth={self.depth})")

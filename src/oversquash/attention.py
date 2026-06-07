"""
QuotientAttention — kQ/I as a multi-head attention *prior*, not a destructive collapse.

The earlier `QuotientWalkConv` operationalized kQ/I as "replace A^g by the
de-duplicated M_g". On a retrieval task that *destroys* signal (the task needs
to know how many sources reach the target), so it underperformed the raw
operator. Verified, robust over seeds. The lesson: kQ/I must *organize*
multiplicity, not delete it.

This layer reframes kQ/I the way ACT_en.tex Def 5.2 already suggests — multi-head
attention is parallel arrows in the quiver, and *head pruning is the ideal
quotient*. Concretely:

  - A standard multi-head GAT runs H independent attention maps and concatenates.
  - We instead tie heads by path-equivalence class under I: arrows (1-hop edges)
    in the same class SHARE attention parameters. Redundant/equivalent influence
    is handled by one shared head (right inductive bias, fewer parameters, no
    inter-head interference); genuinely distinct influence keeps separate heads.

So kQ/I becomes a *structural prior on the attention head assignment*. It does not
discard the multiplicity — attention weights still sum over all incoming edges —
it only constrains which edges share a head. Over-squashing is relieved because
each head aggregates one functionally-coherent bundle into its own slice of the
output, instead of one map spreading thin over all paths.

Where this should win over vanilla multi-head GAT: tasks where parallel paths in
a class carry *redundant* information (noise-redundancy), so weight-sharing is
correct and unshared heads waste capacity learning to ignore the duplication.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class WalkAttention(nn.Module):
    """Learned multi-hop attention whose SUPPORT is the walk-reachability mask.

    This is the synthesis the negative results point to (and the user's insight):
    the multi-hop walk operator that mitigates over-squashing is structurally an
    *attention* — like Transformer self-attention, but with the support given by
    the path algebra (which pairs are connected by length-g walks) instead of
    all-pairs, and crucially with **learned** weights instead of fixed
    multiplicity.

      - walkraw weights every walk-reachable source EQUALLY (by multiplicity): it
        gets the signal through the bottleneck but cannot *select* which source
        matters.
      - WalkAttention attends over the SAME range-g reachable pairs but learns
        query-key weights, so it can both mitigate over-squashing AND select.
      - kQ/I's role here is constructive/structural: it defines the reachability
        support (and could further split path-classes into separate heads). It is
        NOT a destructive collapse — that is what failed.

    The mask is sparse (e.g. ~2% of dense at the bottleneck's deepest range), so
    this is far cheaper than full self-attention while staying global-in-reach.

    forward(x, walk_mask):
        walk_mask : (N, N) sparse or dense 0/1 tensor, [j, i] = 1 if a length-g
                    walk i -> j exists. Provided per depth by the transform.
                    Attention is computed over the nonzero entries only.
    """

    def __init__(self, in_channels: int, out_channels: int, n_heads: int = 4,
                 concat: bool = True, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.concat = concat
        self.dropout = dropout
        self.q = nn.Linear(in_channels, n_heads * out_channels, bias=False)
        self.k = nn.Linear(in_channels, n_heads * out_channels, bias=False)
        self.v = nn.Linear(in_channels, n_heads * out_channels, bias=False)
        self.root = nn.Linear(in_channels,
                              (n_heads * out_channels) if concat else out_channels)
        self.scale = out_channels ** -0.5
        self.reset_parameters()

    def reset_parameters(self):
        for m in (self.q, self.k, self.v, self.root):
            m.reset_parameters()

    def forward(self, x, walk_mask):
        """walk_mask : sparse (N, N) with entry [j, i] != 0 iff a length-g walk
        i -> j exists. We attend ONLY over those nonzero (target j, source i)
        pairs — O(#reachable) not O(N^2) — via a segmented softmax over each
        target's incoming sources, exactly like a sparse graph-attention layer.
        Batched block-diagonal masks stay cheap because only the ~2% nonzero
        entries are ever materialized.
        """
        from torch_geometric.utils import softmax as pyg_softmax
        N, H, C = x.size(0), self.n_heads, self.out_channels
        q = self.q(x).view(N, H, C)
        k = self.k(x).view(N, H, C)
        v = self.v(x).view(N, H, C)

        if walk_mask is None or walk_mask._nnz() == 0:
            out = torch.zeros(N, (H * C) if self.concat else C,
                              device=x.device, dtype=x.dtype)
            return out + self.root(x)

        wm = walk_mask.coalesce()
        tgt, src = wm.indices()[0], wm.indices()[1]          # (E,), (E,)
        # per-pair attention logits: q_target . k_source  (E, H)
        logit = (q[tgt] * k[src]).sum(-1) * self.scale       # (E, H)
        alpha = pyg_softmax(logit, tgt, num_nodes=N)         # softmax over sources per target
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        msg = v[src] * alpha.unsqueeze(-1)                   # (E, H, C)
        agg = torch.zeros(N, H, C, device=x.device, dtype=x.dtype)
        agg.index_add_(0, tgt, msg)                          # scatter-add into targets
        out = agg.reshape(N, H * C) if self.concat else agg.mean(dim=1)
        return out + self.root(x)

    def __repr__(self):  # pragma: no cover
        return (f"{self.__class__.__name__}({self.in_channels}->"
                f"{self.out_channels}, heads={self.n_heads}, masked_by=walks)")


class QuotientAttention(MessagePassing):
    """Single-layer attention whose heads are tied by path-equivalence class.

    Parameters
    ----------
    in_channels, out_channels : int
    n_heads : int
        Number of *distinct* attention heads = number of edge-class buckets we
        map classes into (classes are hashed into [0, n_heads)). Edges in the
        same bucket share that head's attention parameters.
    concat : bool
        Concatenate heads (True) or average (False), as in GAT.
    dropout : float
        Attention dropout.

    forward(x, edge_index, edge_class):
        edge_class : (E,) long tensor, the kQ/I equivalence-class id of each edge
        (from ideal_bridge / the transform). Edges with the same class id are
        routed to the same head. If None, falls back to a single shared head
        (== plain single-head GAT) — a clean ablation.
    """

    def __init__(self, in_channels: int, out_channels: int, n_heads: int = 4,
                 concat: bool = True, dropout: float = 0.0):
        super().__init__(aggr="add", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.concat = concat
        self.dropout = dropout

        # one linear + attention vector PER head (heads are tied to classes)
        self.lin = nn.Linear(in_channels, n_heads * out_channels, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, n_heads, out_channels))
        self.att_dst = nn.Parameter(torch.empty(1, n_heads, out_channels))
        self.root = nn.Linear(in_channels,
                              (n_heads * out_channels) if concat else out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        self.root.reset_parameters()

    def forward(self, x, edge_index, edge_class: Optional[torch.Tensor] = None):
        H, C = self.n_heads, self.out_channels
        x_proj = self.lin(x).view(-1, H, C)              # (N, H, C)

        # per-edge head id: class mod H (so each class maps to exactly one head)
        if edge_class is None:
            head_of_edge = torch.zeros(edge_index.size(1), dtype=torch.long,
                                       device=x.device)
        else:
            head_of_edge = (edge_class % H).long()

        out = self.propagate(edge_index, x=x_proj, head_of_edge=head_of_edge)
        # out: (N, H, C); only the assigned head carries each edge's message
        if self.concat:
            out = out.reshape(-1, H * C)
        else:
            out = out.mean(dim=1)
        return out + self.root(x)

    def message(self, x_j, x_i, head_of_edge, index, size_i):
        # attention logits per head (GAT-style), but each edge only contributes
        # to ITS class's head — mask out the other heads for that edge.
        alpha = (x_j * self.att_src).sum(-1) + (x_i * self.att_dst).sum(-1)  # (E,H)
        alpha = F.leaky_relu(alpha, 0.2)

        # head mask: edge e active only in head head_of_edge[e]
        E, H = alpha.shape
        head_mask = F.one_hot(head_of_edge, num_classes=H).to(alpha.dtype)  # (E,H)
        alpha = alpha.masked_fill(head_mask == 0, float("-inf"))

        # softmax over incoming edges per (target node, head)
        alpha = softmax(alpha, index, num_nodes=size_i)                     # (E,H)
        alpha = torch.nan_to_num(alpha, nan=0.0)  # heads with no edges -> 0
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)                                    # (E,H,C)

    def __repr__(self):  # pragma: no cover
        return (f"{self.__class__.__name__}({self.in_channels}->"
                f"{self.out_channels}, heads={self.n_heads}, "
                f"tied_by=kQ/I)")

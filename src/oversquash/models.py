"""
Models for the over-squashing experiments.

Baselines (GCN, GAT, GIN) are thin wrappers over PyG layers so the comparison
is apples-to-apples (same width, depth, readout). QuotientNet stacks our
QuotientMessagePassing layers; at depth g it consumes the length-g edge-class
tensor from the QuotientPlan.

The factory `build_model(name, ...)` is what the train loop and notebooks call.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool

from .layers import QuotientWalkConv
from .attention import QuotientAttention


class _MLPReadout(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, in_dim)
        self.lin2 = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.lin2(F.relu(self.lin1(x)))


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, dropout=0.0):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.readout = _MLPReadout(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None, **kw):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.readout(x), x


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, heads=4, dropout=0.0):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim, heads=heads, concat=True))
        for _ in range(n_layers - 2):
            self.convs.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True)
            )
        if n_layers >= 2:
            self.convs.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
            )
        self.readout = _MLPReadout(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None, **kw):
        for conv in self.convs:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.readout(x), x


class GIN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, dropout=0.0):
        super().__init__()
        def mlp(i, o):
            return nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Linear(o, o))
        self.convs = nn.ModuleList()
        self.convs.append(GINConv(mlp(in_dim, hidden_dim)))
        for _ in range(n_layers - 1):
            self.convs.append(GINConv(mlp(hidden_dim, hidden_dim)))
        self.readout = _MLPReadout(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None, **kw):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.readout(x), x


class WalkNet(nn.Module):
    """A stack of multi-hop QuotientWalkConv layers.

    Each layer g aggregates over the precomputed range-g walk operator. The
    `operator` arg selects which family of operators the batch supplies:
      - operator='eff' : the EFFECTIVE kQ/I matrices M_g (our model).
      - operator='raw' : the RAW matrices A^g (the ablation that keeps the same
                         architecture but does NOT identify equivalent paths).
    The eff-vs-raw gap is exactly the kQ/I contribution, with everything else
    held fixed — the cleanest possible isolation of the claim.

    forward(x, edge_index, batch, walk_eff, walk_raw): walk_* are lists of
    n_layers sparse (N, N) tensors from transforms.collate_walk_operators.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, dropout=0.0,
                 operator="eff"):
        super().__init__()
        assert operator in ("eff", "raw")
        self.operator = operator
        self.layers = nn.ModuleList()
        self.layers.append(QuotientWalkConv(in_dim, hidden_dim, depth=1))
        for d in range(2, n_layers + 1):
            self.layers.append(QuotientWalkConv(hidden_dim, hidden_dim, depth=d))
        self.readout = _MLPReadout(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None,
                walk_eff=None, walk_raw=None, **kw):
        ops = walk_eff if self.operator == "eff" else walk_raw
        for i, layer in enumerate(self.layers):
            op = None if ops is None or i >= len(ops) else ops[i]
            x = F.relu(layer(x, op))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.readout(x), x


class QuotientAttentionNet(nn.Module):
    """Stack of QuotientAttention layers: multi-head attention whose heads are
    tied by kQ/I path-equivalence class. Reads `edge_class_per_depth` (n_layers,E)
    from the batch (attached by transforms.AttachEdgeClassMatrix). The fair
    baseline is a plain multi-head `GAT` with the same `n_heads`; the difference
    is solely whether heads are kQ/I-tied (this) or free (GAT).
    """

    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, n_heads=4,
                 dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            QuotientAttention(in_dim, hidden_dim, n_heads=n_heads, concat=True,
                              dropout=dropout))
        for _ in range(n_layers - 2):
            self.layers.append(
                QuotientAttention(hidden_dim * n_heads, hidden_dim,
                                  n_heads=n_heads, concat=True, dropout=dropout))
        if n_layers >= 2:
            self.layers.append(
                QuotientAttention(hidden_dim * n_heads, hidden_dim,
                                  n_heads=n_heads, concat=False, dropout=dropout))
        self.readout = _MLPReadout(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None, edge_class_per_depth=None, **kw):
        for i, layer in enumerate(self.layers):
            ec = None
            if edge_class_per_depth is not None and i < edge_class_per_depth.size(0):
                ec = edge_class_per_depth[i]
            x = F.elu(layer(x, edge_index, edge_class=ec))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.readout(x), x


def build_model(name: str, in_dim, hidden_dim, out_dim, n_layers, **kw):
    name = name.lower()
    dropout = kw.get("dropout", 0.0)
    if name == "gat":
        return GAT(in_dim, hidden_dim, out_dim, n_layers,
                   heads=kw.get("heads", 4), dropout=dropout)
    if name in ("gcn", "gin"):
        return {"gcn": GCN, "gin": GIN}[name](
            in_dim, hidden_dim, out_dim, n_layers, dropout=dropout)
    # 'quotient' = WalkNet over effective kQ/I operators (our model);
    # 'walkraw'  = same architecture over raw A^g operators (ablation baseline).
    if name in ("quotient", "walkeff"):
        return WalkNet(in_dim, hidden_dim, out_dim, n_layers,
                       dropout=dropout, operator="eff")
    if name == "walkraw":
        return WalkNet(in_dim, hidden_dim, out_dim, n_layers,
                       dropout=dropout, operator="raw")
    if name in ("qattn", "quotient_attention"):
        return QuotientAttentionNet(in_dim, hidden_dim, out_dim, n_layers,
                                    n_heads=kw.get("heads", 4), dropout=dropout)
    raise ValueError(
        f"unknown model {name!r}; choose from "
        "gcn, gat, gin, quotient, walkraw, qattn")

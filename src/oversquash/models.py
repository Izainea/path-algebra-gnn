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

from .layers import QuotientMessagePassing


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


class QuotientNet(nn.Module):
    """Our model: a stack of QuotientMessagePassing layers.

    forward() accepts `edge_classes`: a list (length n_layers) of (E,) class-id
    tensors, one per depth, from the QuotientPlan. If None, the layers fall back
    to mean aggregation, so QuotientNet(edge_classes=None) is a clean ablation
    against the same architecture without the kQ/I collapse.
    """

    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(QuotientMessagePassing(in_dim, hidden_dim, depth=1))
        for d in range(2, n_layers + 1):
            self.layers.append(
                QuotientMessagePassing(hidden_dim, hidden_dim, depth=d)
            )
        self.readout = _MLPReadout(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None, edge_classes=None, **kw):
        for i, layer in enumerate(self.layers):
            ec = None if edge_classes is None else edge_classes[i]
            x = F.relu(layer(x, edge_index, edge_class=ec))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.readout(x), x


def build_model(name: str, in_dim, hidden_dim, out_dim, n_layers, **kw):
    name = name.lower()
    table = {"gcn": GCN, "gat": GAT, "gin": GIN, "quotient": QuotientNet}
    if name not in table:
        raise ValueError(f"unknown model {name!r}; choose from {list(table)}")
    # GAT takes an extra `heads` kwarg; others ignore unknown kwargs gracefully.
    cls = table[name]
    if name == "gat":
        return cls(in_dim, hidden_dim, out_dim, n_layers,
                   heads=kw.get("heads", 4), dropout=kw.get("dropout", 0.0))
    return cls(in_dim, hidden_dim, out_dim, n_layers,
               dropout=kw.get("dropout", 0.0))

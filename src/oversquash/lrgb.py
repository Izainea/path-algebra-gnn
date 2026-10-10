"""
LRGB Peptides-func experiment: Walk Attention on a real long-range benchmark.

Two adaptations are needed beyond the synthetic bottleneck:

1. **Fast supports.** The path-algebra walk enumeration in ``ideal_bridge`` is
   exact but slow (Python loops over walks); on 10k molecules of ~150 nodes it
   would take hours. For Walk Attention we only need the *reachability support*
   (which pairs are connected by a length-g walk) and, for the raw baseline, the
   walk *counts* — both are sparse matrix powers of the adjacency, computed here
   with scipy in microseconds per graph. (The quotient is not used on LRGB; it
   was the negative result.)

2. **Real molecular I/O.** Atom features are integer category indices, so the
   models start with an embedding; and the task is graph-level multi-label, so
   they end with mean-pooling + a classifier. Metric is Average Precision (AP),
   the official LRGB metric for Peptides-func.

Supports are precomputed once per graph and cached on disk-free tensors attached
to each ``Data`` object; batching assembles them block-diagonally.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

from .attention import WalkAttention


# ----------------------------------------------------------------------
# Fast walk supports via sparse matrix powers
# ----------------------------------------------------------------------
def _masks_and_counts(edge_index: np.ndarray, n: int, max_len: int):
    """Return per-range sparse reachability masks (bool) and walk-count matrices
    A^g, as scipy CSR, computed by matrix powers. Cheap and exact for the support."""
    A = sp.csr_matrix((np.ones(edge_index.shape[1], dtype=np.float32),
                       (edge_index[0], edge_index[1])), shape=(n, n))
    masks, counts = [], []
    P = A.copy()
    for _ in range(max_len):
        counts.append(P.copy())
        masks.append((P > 0))
        P = P @ A
    return masks, counts


def _to_sparse_t(M_csr) -> torch.Tensor:
    """scipy CSR walk matrix M (M[i,j]=count i->j) -> torch sparse (target,source)
    boolean-ish mask used by WalkAttention (it only reads the index pattern)."""
    coo = M_csr.tocoo()
    # WalkAttention reads indices()[0]=target, [1]=source; our M is [i=src, j=tgt]
    idx = torch.tensor(np.vstack([coo.col, coo.row]), dtype=torch.long)
    val = torch.ones(idx.size(1), dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (M_csr.shape[0],) * 2).coalesce()


class AttachLRGBMasks:
    """Transform: attach `walk_masks` (list of n_layers sparse (N,N) tensors,
    target-source) to each molecule, via fast matrix powers."""

    def __init__(self, n_layers: int):
        self.n_layers = n_layers

    def __call__(self, data: Data) -> Data:
        ei = data.edge_index.cpu().numpy()
        n = int(data.num_nodes)
        masks, _ = _masks_and_counts(ei, n, self.n_layers)
        data.walk_masks = [_to_sparse_t(m) for m in masks]
        data.num_nodes_int = n
        return data


def _block_diag(ops, offsets, total):
    idx, val = [], []
    for op, off in zip(ops, offsets):
        op = op.coalesce()
        idx.append(op.indices() + off)
        val.append(op.values())
    if not idx:
        return torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long),
                                       torch.empty(0), (total, total))
    return torch.sparse_coo_tensor(torch.cat(idx, 1), torch.cat(val),
                                   (total, total)).coalesce()


def collate_lrgb(data_list):
    """DataLoader collate_fn: PyG batching + block-diagonal walk masks."""
    n_layers = len(data_list[0].walk_masks)
    sizes = [int(d.num_nodes_int) for d in data_list]
    offs = np.cumsum([0] + sizes[:-1]).tolist()
    total = int(sum(sizes))
    stash = [d.walk_masks for d in data_list]
    for d in data_list:
        d.walk_masks = None
    batch = Batch.from_data_list(data_list)
    for d, m in zip(data_list, stash):
        d.walk_masks = m
    batch.walk_masks = [_block_diag([s[g] for s in stash], offs, total)
                        for g in range(n_layers)]
    return batch


# ----------------------------------------------------------------------
# Models: atom embedding -> backbone -> mean pool -> multi-label head
# ----------------------------------------------------------------------
class _AtomEmbed(nn.Module):
    """Embed the 9 integer atom features and sum them into a hidden vector."""
    def __init__(self, hidden, n_feat=9, vocab=64):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(vocab, hidden) for _ in range(n_feat)])

    def forward(self, x):
        x = x.clamp(min=0, max=63)
        h = 0
        for i, emb in enumerate(self.embs):
            h = h + emb(x[:, i])
        return h


class LRGBNet(nn.Module):
    """Generic graph-level classifier with a pluggable backbone.

    backbone in {'gcn','gat','walkattn','walkraw'}. WalkAttention/walkraw read
    `walk_masks` from the batch; gcn/gat ignore them.
    """

    def __init__(self, backbone, hidden, out_dim, n_layers, n_heads=4, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.embed = _AtomEmbed(hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            if backbone == 'gcn':
                self.convs.append(GCNConv(hidden, hidden))
            elif backbone == 'gat':
                self.convs.append(GATConv(hidden, hidden // n_heads, heads=n_heads, concat=True))
            else:  # walkattn / walkraw both use WalkAttention layers (concat back to hidden)
                self.convs.append(WalkAttention(hidden, hidden // n_heads, n_heads=n_heads,
                                                concat=True, dropout=dropout))
            self.norms.append(nn.LayerNorm(hidden))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, out_dim))
        self.dropout = dropout

    def forward(self, batch):
        x = self.embed(batch.x)
        masks = getattr(batch, 'walk_masks', None)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            if self.backbone in ('gcn', 'gat'):
                h = conv(x, batch.edge_index)
            else:
                m = None if masks is None or i >= len(masks) else masks[i]
                h = conv(x, m)
            x = F.dropout(F.elu(norm(h)), p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch.batch)
        return self.head(g)


# ----------------------------------------------------------------------
# Average Precision (the official Peptides-func metric)
# ----------------------------------------------------------------------
@torch.no_grad()
def average_precision(model, loader, device='cpu'):
    from sklearn.metrics import average_precision_score
    model.eval()
    ys, ps = [], []
    for b in loader:
        b = b.to(device)
        ps.append(torch.sigmoid(model(b)).cpu().numpy())
        ys.append(b.y.cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    # macro AP over the label columns that have at least one positive
    aps = []
    for c in range(y.shape[1]):
        if y[:, c].sum() > 0:
            aps.append(average_precision_score(y[:, c], p[:, c]))
    return float(np.mean(aps))

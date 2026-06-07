"""
Attach per-depth walk operators to graphs and batch them block-diagonally.

The faithful kQ/I layer (layers.QuotientWalkConv) aggregates over a range-g
walk operator W_g of shape (N, N). We precompute, per graph and per range
g = 1..n_layers:

    - the EFFECTIVE operator  M_g[i, j] = dim(e_i·(kQ/I)_g·e_j)   (quotient model)
    - the RAW operator        A_g[i, j] = (A^g)[i, j]              (ablation)

each row-normalized (so each target aggregates a convex combination of sources)
and stored as a torch sparse COO tensor. The heavy path-algebra work runs once
here (cached by topology), never in the training loop.

Batching: a batch of graphs is one big block-diagonal graph. PyG offsets node
indices per graph, so we assemble the batched (N_total, N_total) operator by
shifting each graph's (row, col) indices by its node offset and concatenating —
`collate_walk_operators` does this and is passed to DataLoader as `collate_fn`.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data, Batch

from .ideal_bridge import walk_operators, edge_class_matrix


def _to_sparse_operator(M: np.ndarray, scale: float) -> torch.Tensor:
    """Turn a dense (N, N) walk-count matrix M (M[i, j] = walks i->j) into a
    sparse aggregation operator W with W[j, i] = M[i, j] / scale, transposed so
    `sparse.mm(W, H)` sends source features into targets (target = row j).

    CRITICAL — we do NOT row-normalize. Row (per-target) normalization makes the
    weights into a convex combination over sources, which **erases the
    multiplicity gap**: on a symmetric bottleneck, normalized-raw and
    normalized-effective both become uniform 1/K and are byte-identical, so the
    quotient layer would be a no-op (a bug we hit and verified). Instead we scale
    BOTH the raw and effective operators by a single shared constant `scale`
    (the max raw column-sum of the graph), so the eff/raw ratio survives: a
    target fed by M^d redundant raw walks gets a weight M^d larger than under the
    de-duplicated effective operator. That surviving amplification is exactly the
    over-squashing pressure the quotient relieves.
    """
    N = M.shape[0]
    W = M / (scale if scale > 0 else 1.0)
    Wt = W.T                                            # rows = targets
    idx = np.nonzero(Wt)
    if idx[0].size == 0:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty(0), (N, N)).coalesce()
    indices = torch.tensor(np.vstack(idx), dtype=torch.long)
    values = torch.tensor(Wt[idx], dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()


class AttachWalkOperators:
    """Transform: attach `walk_eff` and `walk_raw` — each a python list of
    n_layers sparse (N, N) tensors — to every Data. Cached by topology so
    identical graphs (all NeighborsMatch trees of one radius, repeated
    molecules) are analyzed once.
    """

    def __init__(self, n_layers: int, relation_strategy: str = "parallel_paths"):
        self.n_layers = n_layers
        self.relation_strategy = relation_strategy
        self._cache: dict = {}

    def __call__(self, data: Data) -> Data:
        ei = data.edge_index.cpu().numpy()
        num_nodes = int(data.num_nodes)
        key = (num_nodes, tuple(map(tuple, ei.T.tolist())))
        if key not in self._cache:
            raw, eff = walk_operators(ei, num_nodes, self.n_layers,
                                      self.relation_strategy)
            # One shared per-(graph, depth) scale for BOTH raw and eff so the
            # multiplicity ratio survives (see _to_sparse_operator). We use the
            # max raw column-sum at each depth (the worst squashing at that range)
            # to keep magnitudes ~O(1) while preserving eff/raw differences.
            eff_ops, raw_ops = [], []
            for Rg, Eg in zip(raw, eff):
                scale = float(Rg.sum(axis=0).max())
                eff_ops.append(_to_sparse_operator(Eg, scale))
                raw_ops.append(_to_sparse_operator(Rg, scale))
            self._cache[key] = (eff_ops, raw_ops)
        eff_ops, raw_ops = self._cache[key]
        # store as plain python lists; the custom collate assembles batches.
        data.walk_eff = eff_ops
        data.walk_raw = raw_ops
        data.num_nodes_int = num_nodes
        return data


def _block_diag_sparse(ops: list, offsets: list, total: int) -> torch.Tensor:
    """Assemble per-graph sparse (n_i, n_i) operators into one block-diagonal
    (total, total) sparse tensor, shifting each by its node offset."""
    all_idx, all_val = [], []
    for op, off in zip(ops, offsets):
        op = op.coalesce()
        idx = op.indices() + off          # shift both rows and cols by offset
        all_idx.append(idx)
        all_val.append(op.values())
    if not all_idx:
        return torch.sparse_coo_tensor(torch.empty((2, 0), dtype=torch.long),
                                       torch.empty(0), (total, total))
    indices = torch.cat(all_idx, dim=1)
    values = torch.cat(all_val)
    return torch.sparse_coo_tensor(indices, values, (total, total)).coalesce()


class AttachEdgeClassMatrix:
    """Transform for QuotientAttention: attach `edge_class_per_depth` (n_layers,E)
    long matrix and `num_edge_classes` to each Data. Cached by topology.

    The matrix is the kQ/I equivalence-class id of each edge at each depth. When
    batched by `collate_edge_classes`, ids are offset per graph so classes never
    collide across graphs (the attention layer hashes class -> head with class%H).
    """

    def __init__(self, n_layers: int, relation_strategy: str = "parallel_paths"):
        self.n_layers = n_layers
        self.relation_strategy = relation_strategy
        self._cache: dict = {}

    def __call__(self, data: Data) -> Data:
        ei = data.edge_index.cpu().numpy()
        num_nodes = int(data.num_nodes)
        key = (num_nodes, tuple(map(tuple, ei.T.tolist())))
        if key not in self._cache:
            self._cache[key] = edge_class_matrix(
                ei, num_nodes, self.n_layers, self.relation_strategy)
        mat = self._cache[key]
        data.edge_class_per_depth = torch.from_numpy(mat).long()
        data.num_edge_classes = int(mat.max()) + 1 if mat.size else 0
        data.num_nodes_int = num_nodes
        return data


def collate_edge_classes(data_list: list):
    """DataLoader collate_fn for QuotientAttention: PyG batching PLUS per-graph
    offset of `edge_class_per_depth` so class ids are globally unique."""
    from torch_geometric.data import Batch
    n_layers = data_list[0].edge_class_per_depth.size(0)
    offset = 0
    cols = []
    stash = []
    for d in data_list:
        stash.append(d.edge_class_per_depth)
        cols.append(d.edge_class_per_depth + offset)   # offset rows uniformly
        offset += int(d.num_edge_classes)
        d.edge_class_per_depth = None                  # hide from default batching
    batch = Batch.from_data_list(data_list)
    for d, m in zip(data_list, stash):
        d.edge_class_per_depth = m                     # restore
    batch.edge_class_per_depth = torch.cat(cols, dim=1)  # (n_layers, E_total)
    return batch


def collate_walk_operators(data_list: list):
    """DataLoader collate_fn: standard PyG batching PLUS block-diagonal assembly
    of the per-depth walk operators into `batch.walk_eff` / `batch.walk_raw`
    (each a list of n_layers sparse (N_total, N_total) tensors)."""
    n_layers = len(data_list[0].walk_eff)
    # node offsets per graph in the batch
    sizes = [int(d.num_nodes_int) for d in data_list]
    offsets = np.cumsum([0] + sizes[:-1]).tolist()
    total = int(sum(sizes))

    # strip the heavy attrs before PyG's default batching, re-attach after
    stash = []
    for d in data_list:
        stash.append((d.walk_eff, d.walk_raw))
        d.walk_eff = None
        d.walk_raw = None
    batch = Batch.from_data_list(data_list)
    for d, (e, r) in zip(data_list, stash):
        d.walk_eff, d.walk_raw = e, r   # restore (datasets may be reused)

    batch.walk_eff = [
        _block_diag_sparse([d.walk_eff[g] for d in data_list], offsets, total)
        for g in range(n_layers)
    ]
    batch.walk_raw = [
        _block_diag_sparse([d.walk_raw[g] for d in data_list], offsets, total)
        for g in range(n_layers)
    ]
    return batch

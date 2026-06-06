"""
Datasets for the over-squashing experiments.

1. NeighborsMatch (Alon & Yahav, 2021, "On the Bottleneck of Graph Neural
   Networks and its Practical Implications"): the canonical synthetic probe.
   A binary tree of depth r; the target (root) must output the label attached
   to the unique leaf whose degree-marker matches the root's. Solving it
   requires routing information from radius r through a single path — exactly
   the regime where over-squashing bites. Accuracy is plotted against r.

2. LRGB Peptides (Dwivedi et al., 2022, Long Range Graph Benchmark): a real
   molecular dataset where long-range dependencies matter. Loaded via PyG's
   `LRGBDataset`. Used to show the kQ/I layer helps beyond synthetic data,
   which is what an IAPR/CIARP reviewer will want.

The NeighborsMatch generator here is a compact, dependency-free reimplementation
sufficient for the radius sweep; it is intentionally small so a notebook can
run the full sweep on CPU.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch_geometric.data import Data


# --------------------------------------------------------------------------
# NeighborsMatch
# --------------------------------------------------------------------------
def make_neighborsmatch_tree(radius: int, generator: torch.Generator) -> Data:
    """Generate ONE NeighborsMatch instance as a tree of depth `radius`.

    Construction (following Alon & Yahav):
      - A perfect binary tree with `radius` levels below the root.
      - Each leaf carries a one-hot "key" (its index among leaves) and a "value".
      - The root carries one of the keys; the label is the value of the leaf
        whose key matches. Information must traverse `radius` hops to the root.

    Node features: [is_root, key_onehot(L), value_onehot(L)] where L = #leaves.
    Label: integer in [0, L) attached to the root node (others masked).
    """
    n_leaves = 2 ** radius
    # node ids: 0 = root, then a standard array-encoded binary tree
    n_nodes = 2 ** (radius + 1) - 1
    edges = []
    for node in range(1, n_nodes):
        parent = (node - 1) // 2
        edges.append((node, parent))   # leaf -> ... -> root (toward the root)
        edges.append((parent, node))   # keep it bidirectional
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    leaf_start = n_nodes - n_leaves
    # assign a random permutation of keys to leaves; values = identity
    perm = torch.randperm(n_leaves, generator=generator)
    feat_dim = 1 + n_leaves + n_leaves
    x = torch.zeros(n_nodes, feat_dim)
    for li in range(n_leaves):
        node = leaf_start + li
        key = perm[li].item()
        x[node, 1 + key] = 1.0            # key one-hot
        x[node, 1 + n_leaves + li] = 1.0  # value one-hot (= leaf position)
    # root: mark it and give it the query key
    query_key = int(torch.randint(n_leaves, (1,), generator=generator).item())
    x[0, 0] = 1.0
    x[0, 1 + query_key] = 1.0
    # label: the value (leaf position) whose key == query_key
    target_leaf = int((perm == query_key).nonzero(as_tuple=True)[0].item())

    y = torch.full((n_nodes,), -100, dtype=torch.long)  # -100 = ignore index
    y[0] = target_leaf

    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_classes = n_leaves
    data.radius = radius
    data.root_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.root_mask[0] = True
    return data


def neighborsmatch_dataset(radius: int, n_samples: int, seed: int = 0):
    """A list[Data] of `n_samples` NeighborsMatch instances at a fixed radius."""
    g = torch.Generator().manual_seed(seed + radius)  # decorrelate per radius
    return [make_neighborsmatch_tree(radius, g) for _ in range(n_samples)]


# --------------------------------------------------------------------------
# LRGB Peptides
# --------------------------------------------------------------------------
def load_lrgb_peptides(root: str = "data/lrgb", name: str = "Peptides-func",
                       split: str = "train"):
    """Load an LRGB Peptides split via PyG.

    name : 'Peptides-func' (multi-label classification, metric AP) or
           'Peptides-struct' (regression, metric MAE).
    Requires `torch_geometric>=2.5` (LRGBDataset) and rdkit for featurization.
    """
    from torch_geometric.datasets import LRGBDataset
    return LRGBDataset(root=root, name=name, split=split)

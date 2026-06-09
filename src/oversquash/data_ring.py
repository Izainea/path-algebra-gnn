"""
RingTransfer: the canonical over-squashing benchmark with *clear parallel paths*.

A cycle of ``n`` nodes. The source node carries a one-hot class label; the target
node, diametrically opposite, must recover it. Information can travel **two ways
around the ring** -- the two arcs are parallel paths of the same length ``n/2``.
This is the over-squashing probe of Di Giovanni et al. (2023): the distance is
``n/2`` (so ``n/2`` message-passing layers are needed), and the walk multiplicity
into the target grows like ``4^{n/2}`` with ring size, so the squashing is
severe and tunable -- exactly the regime where a multi-hop, parallel-path-aware
operator should help and where one-hop message passing collapses.

Unlike the bottleneck chain (which we designed), RingTransfer is a standard
benchmark, which makes the comparison more credible.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data


def make_ring_transfer(n: int, num_classes: int,
                       generator: torch.Generator) -> Data:
    """One RingTransfer instance on a cycle of ``n`` nodes.

    Features per node: [is_source, is_target, class_onehot(num_classes)].
    The source (node 0) holds a random class in its one-hot slot; the target
    (node n//2) is marked and must predict that class. All other nodes are
    neutral. Label is on the target node; chance is 1/num_classes.
    """
    assert n % 2 == 0 and n >= 4
    src, tgt = 0, n // 2
    # undirected cycle as a directed quiver (both orientations)
    edges = []
    for i in range(n):
        j = (i + 1) % n
        edges.append((i, j)); edges.append((j, i))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    feat_dim = 2 + num_classes
    x = torch.zeros(n, feat_dim)
    cls = int(torch.randint(num_classes, (1,), generator=generator).item())
    x[src, 0] = 1.0
    x[src, 2 + cls] = 1.0          # source carries the label
    x[tgt, 1] = 1.0               # target is marked (no label)

    y = torch.full((n,), -100, dtype=torch.long)
    y[tgt] = cls

    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_classes = num_classes
    data.ring_size = n
    data.root_mask = torch.zeros(n, dtype=torch.bool)
    data.root_mask[tgt] = True
    return data


def ring_dataset(ring_size: int, n_samples: int, seed: int = 0,
                 num_classes: int = 5):
    """list[Data] of RingTransfer instances at a fixed ring size.

    Signature mirrors the other generators (size, n_samples, seed) so it drops
    into the same training harness.
    """
    g = torch.Generator().manual_seed(seed + ring_size)
    return [make_ring_transfer(ring_size, num_classes, g) for _ in range(n_samples)]

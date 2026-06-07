"""
Synthetic graphs with *serious, tunable* over-squashing that the path algebra
can resolve — unlike trees (NeighborsMatch), where paths are unique and kQ/I has
nothing to collapse.

Construction — "bottleneck chain" (a chain of hourglasses):

    sources (K)              target
       o\                      /o
       o-+= L1 == L2 == ... == Ld =
       o/   (M)   (M)         (M) \o
       ...

  - K source nodes carry a one-hot KEY and a one-hot VALUE.
  - d "bottleneck layers" of M *interchangeable* nodes each. Consecutive layers
    are fully connected (every node in L_k -> every node in L_{k+1}); sources
    fully connect into L_1; L_d fully connects into a single target t.
  - Because each layer's nodes are interchangeable, the number of length-(d+1)
    walks source -> target explodes as ~K·M^d (vanilla must squash all of them
    into the target's fixed-width vector), while the de-duplicated count under
    kQ/I stays tiny (all those walks are functionally equivalent). So the
    raw/eff multiplicity gap grows with depth d — this is the over-squashing
    knob.

Task — "bottlenecked retrieval":
  The target must output the VALUE of the source whose KEY equals a query key
  (the query is placed on the target's own features). The required information
  sits d+1 hops away, behind the bottleneck, so a model that cannot preserve it
  through the compression fails. Accuracy is plotted against depth d.

This is intentionally a *graph-level* (single-target) classification, evaluated
on the target node only, exactly like NeighborsMatch but with heavy path
redundancy.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data


def _bottleneck_edges(K: int, M: int, depth: int):
    """Edges + node bookkeeping for a depth-`d` bottleneck chain.

    Node ids: [0..K)            sources
              then `depth` blocks of M nodes each
              then 1 target (last id).
    Edges are directed source->...->target (information flows to the target).
    """
    layers = []
    nid = K
    for _ in range(depth):
        layers.append(list(range(nid, nid + M)))
        nid += M
    target = nid
    n_nodes = nid + 1

    edges = []
    # sources -> L1
    for s in range(K):
        for v in layers[0]:
            edges.append((s, v))
    # L_k -> L_{k+1}
    for k in range(depth - 1):
        for u in layers[k]:
            for v in layers[k + 1]:
                edges.append((u, v))
    # L_d -> target
    for u in layers[-1]:
        edges.append((u, target))

    edge_index = torch.as_tensor(np.array(edges, dtype=np.int64).T,
                                 dtype=torch.long)
    return edge_index, n_nodes, target


def make_bottleneck_graph(K: int, M: int, depth: int,
                          generator: torch.Generator) -> Data:
    """One bottlenecked-retrieval instance.

    Features per node: [is_source, is_target, key_onehot(K), value_onehot(K)].
    The target carries the query key (in its key slot) and no value.
    Label (on the target): the value index of the source whose key == query.
    """
    edge_index, n_nodes, target = _bottleneck_edges(K, M, depth)
    feat_dim = 2 + K + K
    x = torch.zeros(n_nodes, feat_dim)

    # sources: random permutation of keys, value = source position
    perm = torch.randperm(K, generator=generator)
    for s in range(K):
        x[s, 0] = 1.0                 # is_source
        x[s, 2 + int(perm[s])] = 1.0  # key one-hot
        x[s, 2 + K + s] = 1.0         # value one-hot (= source position)

    # target: query key, marked
    query = int(torch.randint(K, (1,), generator=generator).item())
    x[target, 1] = 1.0                # is_target
    x[target, 2 + query] = 1.0        # query key one-hot
    target_value = int((perm == query).nonzero(as_tuple=True)[0].item())

    y = torch.full((n_nodes,), -100, dtype=torch.long)
    y[target] = target_value

    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_classes = K
    data.depth = depth
    data.root_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.root_mask[target] = True
    return data


def bottleneck_dataset(depth: int, n_samples: int, seed: int = 0,
                       K: int = 5, M: int = 4):
    """A list[Data] of bottlenecked-retrieval instances at a fixed depth.

    Signature mirrors data.neighborsmatch_dataset (depth, n_samples, seed) so it
    drops into run_sweep; K and M default to a regime with strong redundancy.

    NOTE (verified): on THIS retrieval task the kQ/I-as-collapse operator hurts,
    because the label depends on source multiplicity, which de-duplication
    destroys. The raw multi-hop walk operator (`walkraw`) is what mitigates the
    over-squashing here. Use `noise_redundancy_dataset` for the regime where
    de-duplication / head-tying is the right bias.
    """
    g = torch.Generator().manual_seed(seed + depth)
    return [make_bottleneck_graph(K, M, depth, g) for _ in range(n_samples)]


def make_noise_redundancy_graph(K: int, M: int, depth: int,
                                generator: torch.Generator,
                                noise_std: float = 1.0) -> Data:
    """Same bottleneck topology, but parallel paths carry REDUNDANT copies of the
    signal plus independent noise — the regime where collapsing/​tying equivalent
    paths is the *correct* inductive bias.

    Each of the K sources holds a clean one-hot key + a scalar 'payload'. The
    M interchangeable nodes in layer 1 each receive (and forward) a NOISY copy of
    the same source signal: copy = signal + N(0, noise_std). Because the M copies
    are functionally equivalent (same underlying signal, independent noise),
    averaging them once (kQ/I head-tying) denoises; a vanilla model that treats
    the M paths as distinct must squash M noisy copies into the target's vector
    and learn to average them, wasting capacity.

    Task: target outputs the key index of the source with the largest payload.
    The payload signal must survive the noisy, redundant bottleneck.
    """
    edge_index, n_nodes, target = _bottleneck_edges(K, M, depth)
    feat_dim = 2 + K + 1                      # is_source, is_target, key(K), payload
    x = torch.zeros(n_nodes, feat_dim)

    payloads = torch.randn(K, generator=generator)
    for s in range(K):
        x[s, 0] = 1.0
        x[s, 2 + s] = 1.0                     # key one-hot = identity
        x[s, 2 + K] = payloads[s]             # clean payload on the source

    # inject independent noise on the first bottleneck layer's copies
    layer1 = list(range(K, K + M))
    for v in layer1:
        x[v, 2 + K] = float(torch.randn(1, generator=generator) * noise_std)

    x[target, 1] = 1.0
    target_key = int(torch.argmax(payloads).item())

    y = torch.full((n_nodes,), -100, dtype=torch.long)
    y[target] = target_key

    data = Data(x=x, edge_index=edge_index, y=y)
    data.num_classes = K
    data.depth = depth
    data.root_mask = torch.zeros(n_nodes, dtype=torch.bool)
    data.root_mask[target] = True
    return data


def noise_redundancy_dataset(depth: int, n_samples: int, seed: int = 0,
                             K: int = 5, M: int = 4, noise_std: float = 1.0):
    """list[Data] of noise-redundancy instances; signature matches the sweep."""
    g = torch.Generator().manual_seed(seed + depth)
    return [make_noise_redundancy_graph(K, M, depth, g, noise_std)
            for _ in range(n_samples)]

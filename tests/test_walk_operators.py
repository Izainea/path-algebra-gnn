"""
Regression tests for the core kQ/I machinery the experiments rest on.

If any of these break, the over-squashing result is invalid — so they guard the
exact properties the paper claims:
  1. effective walk multiplicity <= raw, always (the structural bound);
  2. fully-interchangeable parallel paths collapse to a single representative;
  3. the bottleneck family produces the raw ~ K*M^d explosion with eff ~ K;
  4. the trainable quotient operator is NOT identical to the raw operator
     (i.e. kQ/I actually changes the function).

Run: pytest tests/ -v   (inside the `oversquash` conda env)
"""

import numpy as np
import torch
import pytest

from oversquash.ideal_bridge import walk_operators
from oversquash.data_bottleneck import make_bottleneck_graph, _bottleneck_edges


def _hourglass(K, M):
    t = K + M
    edges = []
    for s in range(K):
        for m in range(K, K + M):
            edges.append((s, m))
    for m in range(K, K + M):
        edges.append((m, t))
    return np.array(edges).T, K + M + 1, t


def test_effective_leq_raw():
    """eff[g] <= raw[g] elementwise for several graphs and ranges."""
    ei, N, _ = _hourglass(3, 5)
    raw, eff = walk_operators(ei, N, max_length=3)
    for R, E in zip(raw, eff):
        assert (E <= R + 1e-9).all()


@pytest.mark.parametrize("M", [2, 3, 5, 8])
def test_interchangeable_paths_collapse_to_one(M):
    """All M parallel length-2 paths source->target are equivalent => eff == 1."""
    ei, N, t = _hourglass(1, M)
    raw, eff = walk_operators(ei, N, max_length=2)
    assert raw[1][0, t] == M          # M raw walks
    assert eff[1][0, t] == 1          # collapse to a single representative


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_bottleneck_multiplicity_explodes(depth):
    """raw ~ K*M^d into the target while eff stays ~ K (the squashing knob)."""
    K, M = 5, 4
    g = torch.Generator().manual_seed(0)
    data = make_bottleneck_graph(K, M, depth, g)
    ei, N = data.edge_index.numpy(), data.x.size(0)
    t = int(data.root_mask.nonzero()[0])
    L = depth + 1
    raw, eff = walk_operators(ei, N, max_length=L)
    raw_sum = raw[L - 1][:, t].sum()
    eff_sum = eff[L - 1][:, t].sum()
    assert raw_sum == pytest.approx(K * (M ** depth))
    assert eff_sum == K               # one effective message per source
    assert eff_sum < raw_sum          # genuine compression for depth>=1 when M>1


def test_quotient_operator_differs_from_raw():
    """The effective operator the layer consumes must differ from the raw one,
    else the quotient layer is a no-op.

    Regression guard for a real bug: per-row normalization made raw==eff (both
    uniform 1/K) and silently nullified kQ/I. We now use a shared global scale so
    the multiplicity ratio survives — assert it does, via the actual transform.
    """
    from oversquash.transforms import _to_sparse_operator
    ei, N, t = _hourglass(1, 5)            # 5 redundant parallel paths -> target
    raw, eff = walk_operators(ei, N, max_length=2)
    scale = float(raw[1].sum(axis=0).max())
    Wr = _to_sparse_operator(raw[1], scale).to_dense()
    We = _to_sparse_operator(eff[1], scale).to_dense()
    # raw weight into target is 5x the effective (5 walks vs 1), so they differ.
    assert (Wr - We).abs().max().item() > 1e-6
    assert Wr[t].sum().item() > We[t].sum().item()  # raw amplifies, eff does not


def test_bottleneck_edges_shape():
    """Edge construction: source->layer->...->target wiring is well-formed."""
    K, M, d = 4, 3, 2
    ei, N, t = _bottleneck_edges(K, M, d)
    assert N == K + d * M + 1
    assert t == N - 1
    # every edge endpoint within range; directed toward the target side
    assert ei.min() >= 0 and ei.max() == t

"""
Diagnostic (fallback) claim: predict WHERE over-squashing happens from the
quotient algebra, and correlate it with empirical GNN failure.

This is the lightweight, near-certain-to-finish contribution. It reuses the
already-implemented `aiq.gnn.over_squashing_diagnostic`, which flags
(source, target, depth) triples where the walk-entropy H_g(i,j) exceeds
log2(hidden_dim) — i.e. where more path-diversity must be packed into the
embedding than the width can hold. We then check that nodes/pairs the algebra
marks as bottlenecked are exactly the ones a trained GNN gets wrong.

If the trainable QuotientMessagePassing layer does not beat baselines by the
day-7 checkpoint, the paper pivots to this section as its empirical core.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _import_aiq_gnn():
    try:
        from aiq.gnn import over_squashing_diagnostic
        from aiq.quiver import Quiver
        return over_squashing_diagnostic, Quiver
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "diagnostic requires aiq-quivers (aiq.gnn). "
            "pip install -e ../quivers_analysis"
        ) from e


def bottleneck_scores(edge_index: np.ndarray, num_nodes: int,
                      hidden_dim: int, max_depth: int) -> dict:
    """Return aiq's over-squashing diagnostic for the given graph.

    Output keys: 'bottlenecks' (list of (i, j, g, H_g)), 'max_entropy'.
    A pair is bottlenecked when H_g(i,j) > log2(hidden_dim).
    """
    over_squashing_diagnostic, Quiver = _import_aiq_gnn()
    vertices = list(range(num_nodes))
    # aiq.Quiver wants arrows as a list of (name, source, target) triples.
    arrows = [(f"a{a}", int(u), int(v))
              for a, (u, v) in enumerate(zip(edge_index[0].tolist(),
                                             edge_index[1].tolist()))]
    quiver = Quiver(vertices, arrows)
    return over_squashing_diagnostic(quiver, max_depth, hidden_dim)


def correlate_with_errors(bottleneck_pairs, error_pairs) -> dict:
    """Quantify agreement between predicted bottlenecks and empirical errors.

    Parameters
    ----------
    bottleneck_pairs : set of (i, j) flagged by the algebra.
    error_pairs : set of (i, j) the trained model gets wrong.

    Returns precision/recall/F1 of the algebraic predictor against errors, plus
    the raw confusion counts. This is the headline number for the fallback.
    """
    bp, ep = set(bottleneck_pairs), set(error_pairs)
    tp = len(bp & ep)
    fp = len(bp - ep)
    fn = len(ep - bp)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }

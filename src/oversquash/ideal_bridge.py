"""
Bridge between a PyG computation graph and the path-algebra quotient kQ/I
provided by `aiq-quivers`.

The CIARP claim is that identifying *functionally equivalent paths before
aggregating* mitigates over-squashing. Concretely, for two vertices i, j and a
walk length g, the path algebra distinguishes every walk i ⇝ j, so the number
of messages compressed into a fixed-width vector is

    n_g(i, j) = dim_k(e_i · kQ_g · e_j) = (A^g)_{ij}.

An admissible ideal I that identifies parallel walks (the relation
α₁β ≡ α₂β from ACT_en.tex, Example ex:efecto_ideal) collapses this to

    dim_k(e_i · (kQ/I)_g · e_j)  ≤  n_g(i, j),

so the quotient gives a *structural* upper bound on the multiplicity that has
to be squashed. This module computes, for a given graph, the per-(i, j, g)
collapse map that `layers.QuotientMessagePassing` uses to merge redundant
messages.

We depend only on the public `aiq-quivers` API:
    aiq.quiver.Quiver
    aiq.path_algebra.{PathAlgebra, Ideal, QuotientAlgebra, PathAlgebraElement}
    aiq.gnn.AttentionQuiver.from_edge_index / prune_by_ideal
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------
# aiq-quivers import. Kept lazy-friendly so `import oversquash` never hard-fails
# if the package layout changes; the experiment notebooks import this eagerly.
# --------------------------------------------------------------------------
def _import_aiq():
    try:
        from aiq.quiver import Quiver
        from aiq.path_algebra import (
            PathAlgebra,
            Ideal,
            QuotientAlgebra,
            PathAlgebraElement,
        )
        return Quiver, PathAlgebra, Ideal, QuotientAlgebra, PathAlgebraElement
    except ImportError as e:  # pragma: no cover - environment guard
        raise ImportError(
            "oversquash requires `aiq-quivers`. Install the local editable copy:\n"
            "    pip install -e ../quivers_analysis\n"
            f"(original error: {e})"
        ) from e


@dataclass
class QuotientPlan:
    """
    Precomputed collapse information for one computation graph.

    Attributes
    ----------
    num_nodes : int
    edge_index : np.ndarray            shape (2, E), the directed arrows
    effective_mult : dict[(i, j, g)] -> int
        dim(e_i · (kQ/I)_g · e_j): #messages AFTER identification.
    raw_mult : dict[(i, j, g)] -> int
        (A^g)_{ij}: #messages BEFORE identification.
    groups : dict[(i, j, g)] -> list[list[int]]
        Partition of the raw walks into equivalence classes under I.
        Each inner list holds indices into the enumerated walk list; messages
        within a class are merged (mean) before aggregation.
    max_length : int
    """

    num_nodes: int
    edge_index: np.ndarray
    effective_mult: dict = field(default_factory=dict)
    raw_mult: dict = field(default_factory=dict)
    groups: dict = field(default_factory=dict)
    max_length: int = 1

    def compression_ratio(self) -> float:
        """Mean effective/raw multiplicity over all (i, j, g) with raw > 0.

        1.0 = no compression; lower = more redundant paths collapsed.
        """
        ratios = [
            self.effective_mult[k] / self.raw_mult[k]
            for k in self.raw_mult
            if self.raw_mult[k] > 0
        ]
        return float(np.mean(ratios)) if ratios else 1.0


def build_quiver_from_edge_index(edge_index: np.ndarray, num_nodes: int):
    """Construct an aiq Quiver from a PyG-style edge_index (2, E) array.

    Parallel edges (multi-arrows) are preserved — they are exactly what the
    path algebra needs to see in order to count walk multiplicity.
    """
    Quiver, *_ = _import_aiq()
    vertices = list(range(num_nodes))
    # aiq.Quiver wants arrows as a list of (name, source, target) triples.
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    arrows = [(f"a{a}", int(u), int(v)) for a, (u, v) in enumerate(zip(src, dst))]
    return Quiver(vertices, arrows)


def parallel_path_ideal(quiver, max_length: int):
    """Build the admissible ideal that identifies same-(source,target,length)
    walks — the canonical 'functionally equivalent paths' relation.

    For every (i, j) and length 2..max_length with ≥2 walks, we add relations
    p_1 - p_k for each subsequent walk p_k, identifying them all with the first.
    Returns an `aiq.path_algebra.Ideal`.
    """
    _, PathAlgebra, Ideal, _, PathAlgebraElement = _import_aiq()
    algebra = PathAlgebra(quiver)
    generators = []
    n = len(quiver.Q0)
    for i in quiver.Q0:
        for j in quiver.Q0:
            for g in range(2, max_length + 1):
                walks = algebra.paths_from_to(i, j, g)
                if len(walks) < 2:
                    continue
                base = walks[0]
                for other in walks[1:]:
                    # relation: base - other  (∈ R_Q^2, hence admissible)
                    rel = PathAlgebraElement({base: 1.0, other: -1.0})
                    generators.append(rel)
    return Ideal(algebra, generators)


def build_quotient_plan(
    edge_index: np.ndarray,
    num_nodes: int,
    max_length: int,
    relation_strategy: str = "parallel_paths",
) -> QuotientPlan:
    """Top-level entry: graph -> QuotientPlan ready for the layer.

    Parameters
    ----------
    edge_index : (2, E) array of directed arrows.
    num_nodes  : |Q0|.
    max_length : longest walk length to consider (= #layers reaching the source).
    relation_strategy : currently only 'parallel_paths'.
    """
    Quiver, PathAlgebra, Ideal, QuotientAlgebra, PathAlgebraElement = _import_aiq()

    quiver = build_quiver_from_edge_index(edge_index, num_nodes)
    algebra = PathAlgebra(quiver)

    if relation_strategy == "parallel_paths":
        ideal = parallel_path_ideal(quiver, max_length)
    else:
        raise ValueError(f"unknown relation_strategy: {relation_strategy!r}")

    quotient = QuotientAlgebra(algebra, ideal)

    plan = QuotientPlan(
        num_nodes=num_nodes,
        edge_index=np.asarray(edge_index),
        max_length=max_length,
    )

    for i in quiver.Q0:
        for j in quiver.Q0:
            for g in range(1, max_length + 1):
                walks = algebra.paths_from_to(i, j, g)
                raw = len(walks)
                if raw == 0:
                    continue
                eff = quotient.dimension(i, j, g)
                plan.raw_mult[(i, j, g)] = raw
                plan.effective_mult[(i, j, g)] = eff
                plan.groups[(i, j, g)] = _equivalence_classes(
                    walks, ideal, PathAlgebraElement
                )
    return plan


def _equivalence_classes(walks, ideal, PathAlgebraElement) -> list[list[int]]:
    """Partition `walks` into classes that reduce to the same normal form mod I.

    Returns a list of index-lists into `walks`. Two walks are merged iff their
    reductions modulo I are equal (up to sign/scale of the leading term).
    """
    normal_forms = []
    for p in walks:
        reduced = ideal.reduce(PathAlgebraElement({p: 1.0}))
        # canonical signature: sorted (path-key, rounded-coeff) tuples
        sig = tuple(
            sorted((str(path), round(c, 9)) for path, c in reduced.terms.items())
        )
        normal_forms.append(sig)

    seen: dict = {}
    classes: list[list[int]] = []
    for idx, sig in enumerate(normal_forms):
        if sig in seen:
            classes[seen[sig]].append(idx)
        else:
            seen[sig] = len(classes)
            classes.append([idx])
    return classes

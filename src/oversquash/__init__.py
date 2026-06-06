"""
oversquash — Algebraic (kQ/I) mitigation of over-squashing in GNNs.

Companion code for the CIARP 2026 paper. The mathematical machinery
(quivers, path algebra kQ, admissible ideals I, quotient kQ/I, the impact
vector n_k = dim_k(e_i·kQ_k·e_j), and the over-squashing diagnostic) lives in
the `aiq-quivers` package; this package is the *learning* layer that turns the
quotient kQ/I into a trainable message-passing operator and benchmarks it.

Public surface:
  - layers.QuotientMessagePassing    the kQ/I-aware aggregation layer
  - models.{GCN,GAT,GIN,QuotientNet} baselines + our model
  - ideal_bridge.*                   build the ideal I from a computation graph
                                     via aiq-quivers and expose it to the layer
  - diagnostic.*                     the lightweight (fallback) claim: predict
                                     bottlenecks from dim(e_i·(kQ/I)^g·e_j)
"""

__version__ = "0.1.0"

__all__ = [
    "layers",
    "models",
    "ideal_bridge",
    "diagnostic",
    "data",
    "train",
]

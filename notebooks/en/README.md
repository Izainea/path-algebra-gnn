# Tutorial series (English)

A runnable, theory-first introduction to the idea behind this repo, for readers
with no background in graph neural networks or quiver algebra. **Each notebook
has at least four hand-drawn theoretical figures** that explain the concepts and
the worked examples, plus code cells that use `aiq-quivers` and `networkx` to
*build the graphs and check the claims*. Read in order.

| # | Notebook | What it explains |
|---|----------|------------------|
| **P0** | `P0_playground.ipynb` | Quivers, paths, and multiplicity (the anatomy) |
| **P1** | `P1_oversquashing.ipynb` | What over-squashing is (the bottleneck, K·M^d, the overflowing vector) |
| **P2** | `P2_quivers_and_paths.ipynb` | The path algebra `kQ`: counting paths via `(A^g)[i,j]` |
| **P3** | `P3_paths_are_messages.ipynb` | Each path is a message; the quotient `kQ/I`; signal vs noise |
| **P4** | `P4_walk_is_attention.ipynb` | The walk operator **is** attention (sparse, multi-hop support) |
| **P5** | `P5_putting_it_together.ipynb` | GAT vs walk operator vs Walk Attention — the proof |

The theoretical figures live in `../figs-theory/en/` (hand-drawn SVGs); the
Spanish series in `../es/` uses the same figures in `../figs-theory/es/`. Code
cells use `aiq-quivers` (path algebra) and `networkx` (to draw the networks) —
no data plots; the figures carry the explanations.

**One-sentence idea:** the path algebra of a graph tells you *which* nodes can
attend to each other across many hops; attention then learns *how much* each one
matters — beating both ordinary message passing and fixed path-counting.

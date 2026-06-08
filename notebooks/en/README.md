# Tutorial (English)

A single runnable, theory-first notebook that walks the whole idea behind this
repo from scratch — for readers with no background in graph neural networks or
quiver algebra.

**[`walk_attention_tutorial.ipynb`](walk_attention_tutorial.ipynb)** — six parts,
**24 hand-drawn theoretical figures** explaining the concepts and worked examples,
plus code cells that use `aiq-quivers` (path algebra) and `networkx` (to see the
networks) to define problems and check the claims. No data plots — the figures
carry the explanations.

| Part | Topic |
|------|-------|
| 0 | Quivers, paths, and multiplicity (the anatomy) |
| 1 | What over-squashing is (the bottleneck, K·M^d, the overflowing vector) |
| 2 | The path algebra `kQ`: counting paths via `(A^g)[i,j]` |
| 3 | Paths are messages; the quotient `kQ/I`; signal vs noise |
| 4 | The walk operator **is** attention (sparse, multi-hop support) |
| 5 | GAT vs walk operator vs Walk Attention — the proof |

Figures live in `../figs-theory/en/`. The Spanish version is in `../es/`.

**One-sentence idea:** the path algebra of a graph tells you *which* nodes can
attend to each other across many hops; attention then learns *how much* each one
matters — beating both ordinary message passing and fixed path-counting.

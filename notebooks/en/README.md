# Tutorial series (English)

A hands-on, runnable introduction to the idea behind this repo, for readers with
no background in graph neural networks or quiver algebra. **Every notebook has at
least three figures.** Read in order.

| # | Notebook | What you build |
|---|----------|----------------|
| **P0** | `P0_playground.ipynb` | Draw tiny graphs and count their paths (4 graphs) |
| **P1** | `P1_oversquashing.ipynb` | What over-squashing is — messages crammed into one vector |
| **P2** | `P2_quivers_and_paths.ipynb` | Quivers and counting paths: `(A^g)[i,j]`, the algebra `kQ` |
| **P3** | `P3_paths_are_messages.ipynb` | Each path is a message; redundancy = compression |
| **P4** | `P4_walk_is_attention.ipynb` | The walk operator **is** attention (sparse, multi-hop) |
| **P5** | `P5_putting_it_together.ipynb` | Train GAT vs walkraw vs WalkAttention — the proof |

**One-sentence idea:** the path algebra of a graph tells you *which* nodes can
attend to each other across many hops; attention then learns *how much* each one
matters — beating both ordinary message passing and fixed path-counting.

All plotting is in `oversquash.viz` (shared with the Spanish series in `../es/`),
so the figures are identical across languages. Setup: activate the `oversquash`
conda env (see the repo `README.md`), then open `P0_playground.ipynb`.

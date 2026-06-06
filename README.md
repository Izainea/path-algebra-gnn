# Algebraic Mitigation of Over-Squashing in GNNs

Experiments for the CIARP 2026 submission. We test whether the path-algebra
quotient **kQ/I** — identifying *functionally equivalent paths before
aggregating* — mitigates over-squashing in graph neural networks, as proposed
in the IAQ framework (thesis `ACT_en.tex`, §4.4 and Remark on over-squashing).

The mathematics (quivers, kQ, admissible ideals, the quotient kQ/I, the impact
vector `n_k = dim_k(e_i·kQ_k·e_j)`, and the over-squashing diagnostic) lives in
the **[`aiq-quivers`](../quivers_analysis)** package. This repo is the *learning*
layer: a trainable `QuotientMessagePassing` operator and the benchmarks.

## Claim

At the level of the learnable operator, the quotient gives a **structural upper
bound** on the multiplicity that must be squashed into a fixed-width embedding:

```
dim(e_i · (kQ/I)_g · e_j)  ≤  dim(e_i · kQ_g · e_j) = (A^g)_{ij}
```

Fewer, de-duplicated messages ⇒ less compression ⇒ relieved over-squashing —
the *algebraic* alternative to rewiring (SDRF), coarsening, or pooling.

- **Strong claim (primary):** `WalkNet` over effective kQ/I operators
  (`quotient`) sustains accuracy on NeighborsMatch where GCN/GAT/GIN and the
  architecture-matched raw-operator ablation (`walkraw`) collapse, and helps on
  LRGB-Peptides. The `quotient` vs `walkraw` gap isolates the kQ/I contribution.
- **Diagnostic claim (fallback):** `dim(e_i·(kQ/I)^g·e_j)` and walk-entropy
  `H_g(i,j)` predict *where* a GNN fails (see `src/oversquash/diagnostic.py`,
  which wraps the already-implemented `aiq.gnn.over_squashing_diagnostic`).
  Pivot here if the trainable layer doesn't beat baselines by the day-7 check.

## Setup

No conda/mamba is assumed present; install miniforge first if needed
(<https://github.com/conda-forge/miniforge>). Then:

```bash
conda env create -f environment.yml
conda activate oversquash

# the LOCAL aiq-quivers (has the 2026-06 fixes), not the PyPI release:
pip install -e ../quivers_analysis

# this package:
pip install -e .

# register the kernel for the notebooks:
python -m ipykernel install --user --name oversquash
```

> **Why conda and Python 3.11?** PyTorch Geometric's extension chain
> (`torch-scatter`/`torch-sparse`) resolves most reliably via the `pyg`
> conda channel, and the machine's default Python (3.14) has no PyTorch wheels
> yet. CPU-only by design — NeighborsMatch and a small Peptides run fit on CPU.

## Layout

```
mitigation_overquashing/
├── environment.yml          # conda env (PyG, torch, aiq deps)
├── pyproject.toml           # this package (oversquash)
├── configs/
│   └── neighborsmatch.yaml  # radius sweep + model/training hyperparameters
├── src/oversquash/
│   ├── ideal_bridge.py      # PyG graph -> aiq Quiver -> ideal I; raw A^g + effective kQ/I walk operators
│   ├── layers.py            # QuotientWalkConv (multi-hop, kQ/I-aware aggregation)
│   ├── transforms.py        # AttachWalkOperators + block-diagonal batch collate
│   ├── models.py            # GCN / GAT / GIN baselines + WalkNet (quotient / walkraw)
│   ├── data.py              # NeighborsMatch generator + LRGB Peptides loader
│   ├── diagnostic.py        # fallback: bottleneck prediction via kQ/I
│   └── train.py             # train/eval loop + radius sweep
├── notebooks/
│   ├── 00_env_check.ipynb       # verify torch/PyG/aiq import + GPU/CPU
│   ├── 01_neighborsmatch.ipynb  # main experiment: accuracy vs radius
│   ├── 02_lrgb_peptides.ipynb   # real-data experiment
│   └── 03_diagnostic.ipynb      # fallback claim
└── results/                 # figures + tables (gitignored; .gitkeep tracked)
```

## Design notes & limitations

- **Multi-hop, not 1-hop.** An earlier 1-hop `MessagePassing` design merged a
  node's in-edges by equivalence class, but under mean aggregation that merge is
  *idempotent* (mean of copies = the original mean) and modeled nothing. The
  multiplicity `n_g(i,j)` is a property of length-`g` **walks** — the `g`-fold
  adjacency composition — so the layer (`QuotientWalkConv`) aggregates directly
  over the precomputed range-`g` operator. This is the faithful reading of the
  paper's claim. Verified: on a radius-3 tree the effective operator satisfies
  `eff ≤ raw` everywhere and cuts total walk multiplicity ~30% (see notebook 00).
- **Batching is solved.** Walk operators are precomputed per graph (cached by
  topology) and assembled block-diagonally per batch by `collate_walk_operators`.
  ⚠️ Use **`torch.utils.data.DataLoader`** with that `collate_fn`, NOT
  `torch_geometric.loader.DataLoader` — the PyG loader overrides custom collate
  functions with its own `Collater` and silently mangles the operators.
- **Open empirical question.** The quotient *provably* lowers the multiplicity to
  squash; whether that yields an *accuracy* gap over `walkraw`/baselines at larger
  radius with proper training is what notebook 01 is for. Defaults in the config
  are untuned starting points. Pivot rule: if no gap by the day-7 checkpoint, lead
  with the diagnostic claim (notebook 03), which already works.
- **Walk-operator scaling.** `walk_operators` enumerates walks via the path
  algebra; for large/dense graphs (e.g. full LRGB molecules at high `max_length`)
  this can be costly. Cache by topology and cap `max_length` per dataset.
- LRGB requires `rdkit`; the `environment.yml` pulls `rdkit-pypi` via pip.

## Reproduce

Open `notebooks/00_env_check.ipynb` first; then `01_neighborsmatch.ipynb`.
Each notebook reads `configs/*.yaml` and writes figures/tables to `results/`.

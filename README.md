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

- **Strong claim (primary):** `QuotientMessagePassing` sustains accuracy on
  NeighborsMatch where GCN/GAT/GIN collapse, and helps on LRGB-Peptides.
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
│   ├── ideal_bridge.py      # PyG graph -> aiq Quiver -> ideal I -> QuotientPlan
│   ├── layers.py            # QuotientMessagePassing (the kQ/I-aware layer)
│   ├── models.py            # GCN / GAT / GIN baselines + QuotientNet
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

## Known limitations (work in progress — do not mistake for finished)

- **Batched edge classes.** `train.edge_classes_for_graph` builds the per-depth
  class tensor for a *single* graph. PyG's `DataLoader` remaps node indices when
  batching, so the quotient layer needs a collate that tiles + offsets the class
  tensor per batch. Until that lands, run the quotient model with `batch_size`
  = 1 graph topology per batch, or use the provided single-graph path. This is
  the first thing to fix before the day-3 training runs.
- **Walk→1-hop attribution.** The ideal is defined over full length-g walks; a
  PyG layer is 1-hop. `edge_classes_for_graph` attributes a walk's class to the
  last arrow into the target. Exact for the parallel-paths relation on trees;
  a documented heuristic on general graphs (paper §3 discusses this).
- LRGB requires `rdkit`; the `environment.yml` pulls `rdkit-pypi` via pip.

## Reproduce

Open `notebooks/00_env_check.ipynb` first; then `01_neighborsmatch.ipynb`.
Each notebook reads `configs/*.yaml` and writes figures/tables to `results/`.

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

- **Status (honest):** the original "kQ/I mitigates over-squashing" claim is
  **not supported** by the experiments — see "What the experiments actually show"
  below. What *does* hold: (1) the multi-hop **walk operator** mitigates
  over-squashing (the raw `A^g` variant, `walkraw`); (2) kQ/I is being re-tested
  as a **multi-head attention prior** (`QuotientAttention`); (3) the **diagnostic**
  use of kQ/I holds. The repo keeps the negative result rather than hiding it.
- **Diagnostic claim (fallback):** `dim(e_i·(kQ/I)^g·e_j)` and walk-entropy
  `H_g(i,j)` predict *where* a GNN fails (see `src/oversquash/diagnostic.py`,
  which wraps the already-implemented `aiq.gnn.over_squashing_diagnostic`).
  Pivot here if the trainable layer doesn't beat baselines by the day-7 check.

### What the experiments actually show (honest, verified)

Three separable findings — only the first is a "kQ/I mitigates over-squashing"
result, and it is **negative**:

**1. The multi-hop walk operator mitigates over-squashing; kQ/I de-duplication does NOT.**
The **bottleneck-chain** task (notebook 04, `data_bottleneck.py`): `K` sources →
`d` layers of `M` interchangeable nodes → one target, so raw walk multiplicity
into the target grows `~K·M^d` while the kQ/I-effective count stays `~K`. Letting
a node aggregate over its range-`g` walk operator (`walkraw`, raw `A^g`) crushes
the over-squashing that pins vanilla GCN/GAT/GIN to chance — but the **quotient
operator (kQ/I `M_g`) consistently *underperforms* `walkraw`**, robust over 3 seeds:

| depth | raw mult | GCN/GIN | GAT | quotient (kQ/I `M_g`) | **walkraw (raw `A^g`)** |
|------:|---------:|--------:|----:|----------------------:|------------------------:|
| 2 | 80  | 0.20 | 0.25 | 0.60 ± 0.16 | **0.79 ± 0.15** |
| 3 | 320 | 0.20 | 0.21 | 0.46 ± 0.05 | **0.69 ± 0.15** |

(chance = 1/K = 0.20, `hidden_dim=16`.) **Why:** this retrieval task needs the
source *multiplicity*, which the quotient discards — here redundant paths are
*signal*, not noise, so collapsing them removes information. An earlier
single-seed table that appeared to favor `quotient` was an artifact of per-row
operator normalization (it made `walk_eff == walk_raw`); fixed, with a
regression test.

**2. kQ/I as a multi-head attention prior** (`attention.py`, `QuotientAttention`):
heads tied by path-equivalence class, the way ACT_en.tex Def 5.2 frames it
("head pruning as ideal quotient"). Expected to help only where parallel paths
are *redundant noise* (`noise_redundancy_dataset`), so weight-sharing is the
right bias. **Under evaluation.**

**3. Diagnostic** (notebook 03): `dim(e_i·(kQ/I)^g·e_j)` and walk-entropy predict
*where* over-squashing happens. Independent of the trainable claims; holds up.

Trees (NeighborsMatch, notebook 01) have **unique** paths, so kQ/I has nothing to
act on and ties the baselines — consistent with finding 1.

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
│   ├── 00_env_check.ipynb           # verify torch/PyG/aiq import + walk operators
│   ├── 01_neighborsmatch.ipynb      # trees: kQ/I has little to collapse (ties)
│   ├── 02_lrgb_peptides.ipynb       # real-data experiment
│   ├── 03_diagnostic.ipynb          # fallback claim
│   └── 04_bottleneck_oversquashing.ipynb  # SERIOUS over-squashing kQ/I resolves
└── results/                 # figures + tables (gitignored; .gitkeep tracked)
```

## Experiment tracking (MLflow)

Sweeps log to a local SQLite-backed MLflow store (the file store is deprecated
in MLflow 3.x). Each sweep is one parent run with a nested run per (model, depth);
params, per-cell `val_acc`, the data's raw/eff multiplicity, and the summary
figure/CSV are logged. View:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`mlflow.db`, `mlruns/`, and `mlartifacts/` are gitignored (regenerate by running
the sweeps). The tidy result CSVs under `results/tables/` are the portable record.

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

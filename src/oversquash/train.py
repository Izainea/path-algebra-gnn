"""
Training / evaluation loop shared by the notebooks.

Kept deliberately small and framework-light: a single `run_experiment` that
takes a config dict, trains each model over the radius sweep (NeighborsMatch)
or a single dataset (LRGB), and returns a tidy results table (list of dicts)
that the notebooks turn into figures.

The quotient model needs, per graph, the length-g edge-class tensors derived
from its QuotientPlan. `edge_classes_for_graph` builds them once per distinct
tree topology (all NeighborsMatch trees of a given radius share a topology, so
this is computed once per radius, not per sample).
"""

from __future__ import annotations

from typing import Optional
import copy

import numpy as np
import torch
import torch.nn.functional as F
# NOTE: use torch's DataLoader, NOT torch_geometric.loader.DataLoader — the PyG
# loader overrides any custom collate_fn with its own Collater, which mangles
# our block-diagonal walk operators. torch's loader honors collate_fn.
from torch.utils.data import DataLoader

from .models import build_model
from .transforms import AttachWalkOperators, collate_walk_operators


# --------------------------------------------------------------------------
# Train / eval one model
# --------------------------------------------------------------------------
def _accuracy_rootmask(logits, data):
    mask = data.root_mask
    pred = logits[mask].argmax(dim=-1)
    return (pred == data.y[mask]).float().mean().item()


def _forward_batch(model, batch):
    """Run a model on a (possibly batched) Data, passing batched walk operators
    when present. Baselines accept **kw and ignore the extra args."""
    return model(batch.x, batch.edge_index, getattr(batch, "batch", None),
                 walk_eff=getattr(batch, "walk_eff", None),
                 walk_raw=getattr(batch, "walk_raw", None))


def train_model(model, loader_train, loader_val, cfg, device="cpu"):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg.get("weight_decay", 0.0))
    best_val, best_state, patience = -1.0, None, cfg.get("patience", 20)
    bad = 0
    for epoch in range(cfg["epochs"]):
        model.train()
        for batch in loader_train:
            batch = batch.to(device)
            opt.zero_grad()
            logits, _ = _forward_batch(model, batch)
            loss = F.cross_entropy(logits, batch.y, ignore_index=-100)
            loss.backward()
            opt.step()
        val_acc = evaluate(model, loader_val, device)
        if val_acc > best_val:
            best_val, best_state, bad = val_acc, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    accs = []
    for batch in loader:
        batch = batch.to(device)
        logits, _ = _forward_batch(model, batch)
        accs.append(_accuracy_rootmask(logits, batch))
    return float(np.mean(accs)) if accs else 0.0


# --------------------------------------------------------------------------
# NeighborsMatch radius sweep
# --------------------------------------------------------------------------
def run_neighborsmatch_sweep(cfg, dataset_fn):
    """Train every model in cfg['models'] across cfg['data']['radii'].

    dataset_fn(radius, n_samples, seed) -> list[Data]  (injected for testability)
    Returns a list of result dicts: {model, radius, val_acc}.
    """
    torch.manual_seed(cfg["seed"])
    results = []
    for radius in cfg["data"]["radii"]:
        data_list = dataset_fn(radius, cfg["data"]["n_samples"], cfg["seed"])
        in_dim = data_list[0].x.size(-1)
        out_dim = data_list[0].num_classes
        n_layers = cfg["model"]["n_layers"] or (radius + 1)

        # Attach per-graph, per-depth walk operators (raw A^g + effective kQ/I).
        # Cached by topology, so all trees of this radius are analyzed once.
        tf = AttachWalkOperators(n_layers=n_layers)
        data_list = [tf(d) for d in data_list]

        n_train = int(len(data_list) * cfg["data"]["train_fraction"])
        train_set, val_set = data_list[:n_train], data_list[n_train:]
        bs = cfg["train"]["batch_size"]
        # custom collate assembles block-diagonal walk operators per batch
        loader_train = DataLoader(train_set, batch_size=bs, shuffle=True,
                                  collate_fn=collate_walk_operators)
        loader_val = DataLoader(val_set, batch_size=bs,
                                collate_fn=collate_walk_operators)

        for name in cfg["models"]:
            model = build_model(name, in_dim, cfg["model"]["hidden_dim"],
                                out_dim, n_layers,
                                dropout=cfg["model"].get("dropout", 0.0))
            # quotient/walkraw read walk_eff/walk_raw from the batch; GCN/GAT/GIN
            # ignore them. Same data either way — no per-model branching needed.
            model, val_acc = train_model(
                model, loader_train, loader_val, cfg["train"],
                device=cfg.get("device", "cpu"),
            )
            results.append({"model": name, "radius": radius, "val_acc": val_acc})
            print(f"[r={radius}] {name:9s} val_acc={val_acc:.3f}")
    return results

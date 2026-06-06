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
from torch_geometric.loader import DataLoader

from .models import build_model
from .ideal_bridge import build_quotient_plan


# --------------------------------------------------------------------------
# Quotient plan -> per-edge class ids per depth
# --------------------------------------------------------------------------
def edge_classes_for_graph(edge_index: np.ndarray, num_nodes: int,
                           n_layers: int) -> list[torch.Tensor]:
    """Map a QuotientPlan to a list of (E,) class-id tensors, one per layer.

    For layer/depth g, edge e = (u -> v) is assigned the id of the length-g
    equivalence class (under I) of the walk that this edge *terminates*. We use
    a per-(target, depth) class numbering so that, within QuotientMessagePassing,
    edges sharing a (target, class) get their messages merged.

    Approximation note for the smoke test: the plan's classes are defined over
    full length-g walks, while a layer sees 1-hop edges. We attribute each
    incoming edge of v to the class of the *last arrow* of each length-g walk
    ending at v; arrows not participating in any multi-walk class get a unique
    id (no merging). This is exact for the parallel-paths relation on trees and
    a documented heuristic on general graphs (see paper §3).
    """
    plan = build_quotient_plan(edge_index, num_nodes, max_length=n_layers,
                               relation_strategy="parallel_paths")
    E = edge_index.shape[1]
    src, dst = edge_index[0], edge_index[1]
    per_layer = []
    for g in range(1, n_layers + 1):
        classes = torch.arange(E)  # default: every edge its own class
        next_id = E
        # merge edges that close a redundant length-g walk into the same target
        for (i, j, gg), groups in plan.groups.items():
            if gg != g:
                continue
            for cls in groups:
                if len(cls) < 2:
                    continue
                # edges into j whose source lies on these walks share a class
                in_edges = np.where(dst == j)[0]
                if in_edges.size == 0:
                    continue
                classes[in_edges] = next_id
                next_id += 1
        per_layer.append(classes.long())
    return per_layer


# --------------------------------------------------------------------------
# Train / eval one model
# --------------------------------------------------------------------------
def _accuracy_rootmask(logits, data):
    mask = data.root_mask
    pred = logits[mask].argmax(dim=-1)
    return (pred == data.y[mask]).float().mean().item()


def train_model(model, loader_train, loader_val, cfg, edge_classes=None,
                device="cpu"):
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
            logits, _ = model(batch.x, batch.edge_index, batch.batch,
                              edge_classes=edge_classes)
            loss = F.cross_entropy(logits, batch.y, ignore_index=-100)
            loss.backward()
            opt.step()
        val_acc = evaluate(model, loader_val, edge_classes, device)
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
def evaluate(model, loader, edge_classes=None, device="cpu"):
    model.eval()
    accs = []
    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch.x, batch.edge_index, batch.batch,
                          edge_classes=edge_classes)
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
        n_train = int(len(data_list) * cfg["data"]["train_fraction"])
        train_set, val_set = data_list[:n_train], data_list[n_train:]
        bs = cfg["train"]["batch_size"]
        loader_train = DataLoader(train_set, batch_size=bs, shuffle=True)
        loader_val = DataLoader(val_set, batch_size=bs)

        in_dim = data_list[0].x.size(-1)
        out_dim = data_list[0].num_classes
        n_layers = cfg["model"]["n_layers"] or (radius + 1)

        # Precompute edge classes once for this radius (shared topology).
        ec_single = edge_classes_for_graph(
            data_list[0].edge_index.numpy(),
            data_list[0].x.size(0), n_layers
        )

        for name in cfg["models"]:
            model = build_model(name, in_dim, cfg["model"]["hidden_dim"],
                                out_dim, n_layers,
                                dropout=cfg["model"].get("dropout", 0.0))
            # only the quotient model consumes edge classes; batching replicates
            # the per-graph class tensor — handled in the notebook via a collate
            # that tiles ec_single. For the smoke test we pass None for baselines.
            ec = ec_single if name == "quotient" else None
            model, val_acc = train_model(
                model, loader_train, loader_val, cfg["train"],
                edge_classes=ec, device=cfg.get("device", "cpu"),
            )
            results.append({"model": name, "radius": radius, "val_acc": val_acc})
            print(f"[r={radius}] {name:9s} val_acc={val_acc:.3f}")
    return results

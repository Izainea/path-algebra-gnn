"""
MLflow-tracked sweeps over a graph-depth/radius axis.

A single parent run per sweep, with one nested run per (model, depth) cell so the
MLflow UI shows each configuration separately and the parent aggregates them.
Logs: hyperparameters, per-cell val accuracy (+ accuracy-vs-depth as a stepped
metric per model), the raw/eff multiplicity gap of the data (the over-squashing
knob), and the summary figure + results CSV as artifacts.

Usage (see notebook 04):
    from oversquash.mlflow_runner import run_sweep_mlflow
    from oversquash.data_bottleneck import bottleneck_dataset
    df = run_sweep_mlflow(cfg, bottleneck_dataset, experiment="bottleneck-oversquash")

`cfg` is the same dict shape used by train.run_neighborsmatch_sweep, with the
depth axis under cfg['data']['radii'] (kept that name for drop-in compatibility).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import mlflow

from .models import build_model
from .transforms import AttachWalkOperators, collate_walk_operators
from .train import train_model
from .ideal_bridge import walk_operators


def _multiplicity_gap(data_list, n_layers):
    """Mean raw vs effective walk multiplicity into the target at range n_layers,
    over a few sample graphs — the data's over-squashing severity."""
    raws, effs = [], []
    for data in data_list[:8]:
        ei = data.edge_index.cpu().numpy()
        N = int(data.num_nodes)
        t = int(data.root_mask.nonzero()[0])
        raw, eff = walk_operators(ei, N, max_length=n_layers)
        raws.append(float(raw[n_layers - 1][:, t].sum()))
        effs.append(float(eff[n_layers - 1][:, t].sum()))
    return float(np.mean(raws)), float(np.mean(effs))


def run_sweep_mlflow(cfg, dataset_fn, experiment="oversquash",
                     tracking_uri="sqlite:///mlflow.db", run_name=None):
    """Train every model in cfg['models'] across cfg['data']['radii'] (= depths),
    logging everything to MLflow. Returns the tidy results DataFrame.

    tracking_uri defaults to a local SQLite DB (MLflow 3.x deprecated the file
    store). View the UI with:  mlflow ui --backend-store-uri sqlite:///mlflow.db
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    torch.manual_seed(cfg["seed"])
    results = []

    with mlflow.start_run(run_name=run_name) as parent:
        # log the flat hyperparameters once on the parent run
        mlflow.log_params({
            "seed": cfg["seed"],
            "hidden_dim": cfg["model"]["hidden_dim"],
            "n_layers_cfg": cfg["model"]["n_layers"],
            "dropout": cfg["model"].get("dropout", 0.0),
            "epochs": cfg["train"]["epochs"],
            "lr": cfg["train"]["lr"],
            "batch_size": cfg["train"]["batch_size"],
            "patience": cfg["train"].get("patience", 20),
            "n_samples": cfg["data"]["n_samples"],
            "train_fraction": cfg["data"]["train_fraction"],
            "models": ",".join(cfg["models"]),
            "depths": ",".join(map(str, cfg["data"]["radii"])),
        })

        for depth in cfg["data"]["radii"]:
            data_list = dataset_fn(depth, cfg["data"]["n_samples"], cfg["seed"])
            in_dim = data_list[0].x.size(-1)
            out_dim = data_list[0].num_classes
            n_layers = cfg["model"]["n_layers"] or (depth + 1)

            tf = AttachWalkOperators(n_layers=n_layers)
            data_list = [tf(d) for d in data_list]

            # over-squashing severity of this depth (logged as a data metric)
            raw_mult, eff_mult = _multiplicity_gap(data_list, n_layers)
            mlflow.log_metric("data_raw_mult", raw_mult, step=depth)
            mlflow.log_metric("data_eff_mult", eff_mult, step=depth)

            n_train = int(len(data_list) * cfg["data"]["train_fraction"])
            train_set, val_set = data_list[:n_train], data_list[n_train:]
            bs = cfg["train"]["batch_size"]
            loader_train = DataLoader(train_set, batch_size=bs, shuffle=True,
                                      collate_fn=collate_walk_operators)
            loader_val = DataLoader(val_set, batch_size=bs,
                                    collate_fn=collate_walk_operators)

            for name in cfg["models"]:
                with mlflow.start_run(run_name=f"{name}_d{depth}", nested=True):
                    mlflow.log_params({"model": name, "depth": depth,
                                       "n_layers": n_layers})
                    model = build_model(name, in_dim,
                                        cfg["model"]["hidden_dim"], out_dim,
                                        n_layers,
                                        dropout=cfg["model"].get("dropout", 0.0))
                    model, val_acc = train_model(
                        model, loader_train, loader_val, cfg["train"],
                        device=cfg.get("device", "cpu"))
                    mlflow.log_metric("val_acc", val_acc)
                # also log on the parent as a per-model curve over depth
                mlflow.log_metric(f"val_acc_{name}", val_acc, step=depth)
                results.append({"model": name, "depth": depth,
                                "val_acc": val_acc,
                                "raw_mult": raw_mult, "eff_mult": eff_mult})
                print(f"[d={depth}] {name:9s} val_acc={val_acc:.3f}  "
                      f"(raw={raw_mult:.0f} eff={eff_mult:.0f})")

        df = pd.DataFrame(results)

        # artifacts: results table + figure
        out = Path(cfg.get("results_dir", "results"))
        (out / "tables").mkdir(parents=True, exist_ok=True)
        (out / "figures").mkdir(parents=True, exist_ok=True)
        csv = out / "tables" / f"{experiment}.csv"
        df.to_csv(csv, index=False)
        mlflow.log_artifact(str(csv))

        fig_path = _plot(df, out / "figures" / f"{experiment}.png", experiment)
        mlflow.log_artifact(str(fig_path))

    return df


def _plot(df, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, g in df.groupby("model"):
        g = g.sort_values("depth")
        ax.plot(g["depth"], g["val_acc"], marker="o", label=name)
    ax.set_xlabel("bottleneck depth $d$ (over-squashing radius)")
    ax.set_ylabel("accuracy")
    ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.savefig(str(path).replace(".png", ".pdf"))
    plt.close(fig)
    return path

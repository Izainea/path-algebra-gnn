"""
Plotting helpers for the didactic notebook series (notebooks/en, notebooks/es).

Every figure used in the tutorials lives here, verified once, so the English and
Spanish notebooks call the *same* functions and stay in sync. Each function takes
an optional `labels` dict so the calling notebook can pass localized titles/axes
without duplicating any logic.

All functions return the Matplotlib figure (and show it). Pure visualization —
the maths is in quiver/ideal_bridge/walk_operators.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None


# ---------------------------------------------------------------- graph drawing
def draw_graph(edges, n_nodes, src=0, dst=None, pos=None, title="", ax=None):
    """Draw a small directed graph; source blue, target green, rest grey."""
    dst = n_nodes - 1 if dst is None else dst
    G = nx.DiGraph(); G.add_nodes_from(range(n_nodes)); G.add_edges_from(edges)
    pos = pos or nx.spring_layout(G, seed=1)
    cols = ['#34d399' if i == dst else '#60a5fa' if i == src else '#cbd5e1'
            for i in range(n_nodes)]
    own = ax is None
    if own:
        _, ax = plt.subplots(figsize=(5.5, 3))
    nx.draw(G, pos, ax=ax, node_color=cols, with_labels=True, node_size=620,
            arrowsize=14, edge_color='#94a3b8', font_size=10)
    ax.set_title(title)
    if own:
        plt.show()
    return ax


def draw_bottleneck(data, K, M, title=""):
    """Layered drawing of a bottleneck graph: sources | middle layers | target."""
    import torch  # local; only needed here
    G = nx.DiGraph(); G.add_edges_from(data.edge_index.t().tolist())
    N = data.num_nodes; t = int(data.root_mask.nonzero()[0])
    layer = {}
    for n in range(K):
        layer[n] = (0, K / 2 - n)
    mid = list(range(K, N - 1))
    for idx, n in enumerate(mid):
        layer[n] = (1 + idx // M, M / 2 - (idx % M))
    layer[t] = (1 + len(mid) // M + 1, 0)
    cols = ['#60a5fa'] * K + ['#fbbf24'] * (N - 1 - K) + ['#34d399']
    fig, ax = plt.subplots(figsize=(7, 4))
    nx.draw(G, layer, ax=ax, node_color=cols, with_labels=True, node_size=620,
            font_size=8, arrowsize=11, edge_color='#cbd5e1')
    ax.set_title(title)
    plt.show()
    return fig


# ------------------------------------------------------- over-squashing numbers
def plot_message_explosion(depths, msgs, title="", xlabel="", ylabel=""):
    """Semilog plot of messages-at-target vs depth (the explosion)."""
    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    ax.semilogy(depths, msgs, 'o-', color='#dc2626', lw=2)
    for d, m in zip(depths, msgs):
        ax.annotate(str(int(m)), (d, m), textcoords="offset points",
                    xytext=(0, 8), ha='center', fontsize=9)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
    return fig


def plot_multiplicity_heatmap(M, title="", xlabel="source i", ylabel="target j"):
    """Heatmap of a walk-count / weight matrix."""
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(M, cmap='Reds')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.show()
    return fig


# ------------------------------------------------------------- path / algebra
def plot_matrix_pair(A, A2, titles=("A", "A^2")):
    """Show adjacency A and A^2 side by side with integer annotations."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, Mx, ti in zip(axes, (A, A2), titles):
        ax.imshow(Mx, cmap='Blues', vmin=0, vmax=max(2, Mx.max()))
        for i in range(Mx.shape[0]):
            for j in range(Mx.shape[1]):
                ax.text(j, i, int(Mx[i, j]), ha='center', va='center',
                        fontsize=10, color='#1e293b')
        ax.set_title(ti); ax.set_xticks(range(Mx.shape[0])); ax.set_yticks(range(Mx.shape[0]))
    plt.tight_layout(); plt.show()
    return fig


def plot_paths_per_length(lengths, counts, title="", xlabel="", ylabel=""):
    """Bar chart of number of paths src->dst at each length."""
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar([str(l) for l in lengths], counts, color='#6366f1')
    for i, c in enumerate(counts):
        ax.text(i, c + 0.03, str(int(c)), ha='center', fontsize=10, fontweight='bold')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    plt.tight_layout(); plt.show()
    return fig


# -------------------------------------------------- raw vs effective (kQ/I)
def plot_raw_vs_eff(depths, raw_vals, eff_vals, title="", xlabel="",
                    ylabel="", legend=("raw  (A^g)", "effective  (kQ/I)")):
    """Grouped bars: raw vs de-duplicated message counts per depth."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(len(depths)); w = 0.38
    ax.bar(x - w/2, raw_vals, w, label=legend[0], color='#f59e0b')
    ax.bar(x + w/2, eff_vals, w, label=legend[1], color='#10b981')
    ax.set_xticks(x); ax.set_xticklabels([str(d) for d in depths])
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(); ax.set_yscale('log'); ax.grid(alpha=0.3, axis='y')
    plt.tight_layout(); plt.show()
    return fig


def plot_two_heatmaps(M1, M2, titles=("raw", "effective")):
    """Two matrices side by side (e.g. A^g vs M_g)."""
    vmax = max(M1.max(), M2.max(), 1)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for ax, Mx, ti in zip(axes, (M1, M2), titles):
        im = ax.imshow(Mx, cmap='Purples', vmin=0, vmax=vmax)
        ax.set_title(ti); ax.set_xlabel('source i'); ax.set_ylabel('target j')
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025)
    plt.show()
    return fig


# ---------------------------------------------------------------- attention
def plot_mask_grid(raw_ops, n_nodes, title="", subtitle_fmt="g={g}: {k}/{N2}"):
    """Show the walk-reachability mask (attention support) at each range g."""
    G = len(raw_ops)
    fig, axes = plt.subplots(1, G, figsize=(3.2 * G, 3.3))
    if G == 1:
        axes = [axes]
    for g, ax in enumerate(axes):
        mask = (raw_ops[g] > 0).astype(int)
        ax.imshow(mask, cmap='Greens', vmin=0, vmax=1)
        ax.set_title(subtitle_fmt.format(g=g+1, k=int(mask.sum()), N2=n_nodes**2),
                     fontsize=10)
        ax.set_xlabel('source'); ax.set_ylabel('target')
    fig.suptitle(title); plt.tight_layout(); plt.show()
    return fig


def plot_fixed_vs_learned(sources, fixed_w, learned_w, title="",
                          legend=("fixed (walkraw)", "learned (attention)")):
    """Grouped bars: fixed vs learned attention weights into the target."""
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(len(sources)); w = 0.38
    ax.bar(x - w/2, fixed_w, w, label=legend[0], color='#f59e0b')
    ax.bar(x + w/2, learned_w, w, label=legend[1], color='#10b981')
    ax.set_xticks(x); ax.set_xticklabels([f's{int(s)}' for s in sources])
    ax.set_ylabel('weight'); ax.set_title(title); ax.legend()
    plt.tight_layout(); plt.show()
    return fig


# ---------------------------------------------------------------- training
def plot_training_curves(curves: dict, title="", xlabel="epoch", ylabel="accuracy",
                         chance=None):
    """Plot one accuracy-vs-epoch curve per model."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, c in curves.items():
        ax.plot(range(1, len(c)+1), c, marker='.', label=name)
    if chance is not None:
        ax.axhline(chance, ls='--', color='grey', label='chance')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
    return fig


def plot_model_bars(names, accs, title="", ylabel="accuracy", chance=None,
                    colors=None):
    """Final-accuracy bar chart across models."""
    colors = colors or ['#fca5a5', '#fcd34d', '#6ee7b7', '#93c5fd', '#c4b5fd'][:len(names)]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(names, accs, color=colors, edgecolor='#334155')
    for i, v in enumerate(accs):
        ax.text(i, v + 0.02, f'{v:.2f}', ha='center', fontweight='bold')
    if chance is not None:
        ax.axhline(chance, ls='--', color='grey', label='chance'); ax.legend()
    ax.set_ylim(0, 1.12); ax.set_ylabel(ylabel); ax.set_title(title)
    plt.tight_layout(); plt.show()
    return fig


# ======================================================================
# DIDACTIC diagrams — these teach a concept, not just display data.
# ======================================================================
from matplotlib.patches import Rectangle, FancyArrowPatch  # noqa: E402


def explain_overflow(steps, capacity=8, title="", panel_fmt="depth {d}\n{m} messages",
                     lost_fmt="{n} lost!"):
    """A fixed-size vector (a box of height `capacity`) filling up and OVERFLOWING
    as the number of messages grows. The single clearest picture of over-squashing.
    `steps` = list of (label_d, n_messages)."""
    n = len(steps)
    fig, axes = plt.subplots(1, n, figsize=(2.7 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, (d, m) in zip(axes, steps):
        ax.add_patch(Rectangle((0, 0), 1, capacity, fill=False, lw=2, ec='#334155'))
        fill = min(m, capacity)
        over = m > capacity
        ax.add_patch(Rectangle((0, 0), 1, fill, fc=('#dc2626' if over else '#60a5fa'),
                               alpha=0.75))
        if over:
            ax.annotate('', xy=(0.5, capacity + 1.8), xytext=(0.5, capacity),
                        arrowprops=dict(arrowstyle='-|>', color='#dc2626', lw=2.2))
            ax.text(0.5, capacity + 2.4, lost_fmt.format(n=m - capacity), ha='center',
                    color='#dc2626', fontsize=9, fontweight='bold')
        ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, capacity + 4)
        ax.set_title(panel_fmt.format(d=d, m=m), fontsize=10); ax.axis('off')
    fig.suptitle(title, y=1.02, fontsize=12)
    plt.tight_layout(); plt.show()
    return fig


def explain_message_hops(edges, pos, path, n_nodes, src, dst, title="",
                         step_fmt="layer {s}"):
    """Show a message traveling one hop per layer: the highlighted path grows
    panel by panel. Makes 'g layers reach g hops' concrete."""
    G = nx.DiGraph(); G.add_nodes_from(range(n_nodes)); G.add_edges_from(edges)
    L = len(path)
    fig, axes = plt.subplots(1, L + 1, figsize=(3.2 * (L + 1), 3))
    for step, ax in enumerate(axes):
        active = set(path[:step])
        ec = ['#dc2626' if e in active else '#d1d5db' for e in G.edges()]
        ew = [3.2 if e in active else 1.0 for e in G.edges()]
        reached = src if step == 0 else path[step - 1][1]
        nc = ['#34d399' if i == dst else '#fb923c' if i == reached
              else '#60a5fa' if i == src else '#e5e7eb' for i in range(n_nodes)]
        nx.draw(G, pos, ax=ax, edge_color=ec, width=ew, node_color=nc,
                with_labels=True, node_size=520, arrowsize=14, font_size=9)
        ax.set_title(step_fmt.format(s=step) + f' · at node {reached}', fontsize=10)
    fig.suptitle(title, y=1.03, fontsize=12)
    plt.tight_layout(); plt.show()
    return fig


def explain_paths_highlighted(edges, pos, paths, n_nodes, src, dst, title="",
                              path_fmt="path {i}"):
    """Draw each src->dst path of a fixed length in its own panel, highlighted —
    so 'n_g = number of paths' is something you can literally count."""
    P = len(paths)
    fig, axes = plt.subplots(1, P, figsize=(3.0 * P, 3))
    if P == 1:
        axes = [axes]
    G = nx.DiGraph(); G.add_nodes_from(range(n_nodes)); G.add_edges_from(edges)
    palette = ['#dc2626', '#7c3aed', '#0891b2', '#ca8a04', '#db2777']
    for i, ax in enumerate(axes):
        pe = set(paths[i]); col = palette[i % len(palette)]
        ec = [col if e in pe else '#e5e7eb' for e in G.edges()]
        ew = [3.2 if e in pe else 1.0 for e in G.edges()]
        nc = ['#34d399' if n == dst else '#60a5fa' if n == src else '#e5e7eb'
              for n in range(n_nodes)]
        nx.draw(G, pos, ax=ax, edge_color=ec, width=ew, node_color=nc,
                with_labels=True, node_size=520, arrowsize=14, font_size=9)
        ax.set_title(path_fmt.format(i=i + 1), fontsize=10, color=col)
    fig.suptitle(title, y=1.03, fontsize=12)
    plt.tight_layout(); plt.show()
    return fig


def explain_attention_on_graph(edges, pos, weight_sets, n_nodes, src_nodes, dst,
                               titles, suptitle=""):
    """Draw attention as ARROW THICKNESS into a target, side by side: e.g. fixed
    (all equal) vs learned (focused). `weight_sets` = list of {edge: weight}."""
    G = nx.DiGraph(); G.add_nodes_from(range(n_nodes)); G.add_edges_from(edges)
    n = len(weight_sets)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.4))
    if n == 1:
        axes = [axes]
    for ax, ws, ti in zip(axes, weight_sets, titles):
        mx = max(ws.values()) or 1
        ew = [0.5 + 6.0 * ws.get(e, 0) / mx for e in G.edges()]
        nc = ['#34d399' if i == dst else '#60a5fa' if i in src_nodes else '#e5e7eb'
              for i in range(n_nodes)]
        nx.draw(G, pos, ax=ax, edge_color='#10b981', width=ew, node_color=nc,
                with_labels=True, node_size=560, arrowsize=12, font_size=9)
        ax.set_title(ti, fontsize=11)
    fig.suptitle(suptitle, y=1.03, fontsize=12)
    plt.tight_layout(); plt.show()
    return fig


def explain_architectures(specs, suptitle=""):
    """Draw N model 'reach' diagrams side by side. Each spec:
    {title, edges, pos, n_nodes, src, dst, reach_edges (highlighted), color}.
    Used in P5 to contrast GAT (1-hop) vs walkraw/WalkAttention (multi-hop)."""
    n = len(specs)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.3))
    if n == 1:
        axes = [axes]
    for ax, sp in zip(axes, specs):
        G = nx.DiGraph(); G.add_nodes_from(range(sp['n_nodes'])); G.add_edges_from(sp['edges'])
        reach = set(sp.get('reach_edges', []))
        col = sp.get('color', '#dc2626')
        ec = [col if e in reach else '#e5e7eb' for e in G.edges()]
        ew = [3.0 if e in reach else 1.0 for e in G.edges()]
        nc = ['#34d399' if i == sp['dst'] else '#60a5fa' if i == sp['src'] else '#e5e7eb'
              for i in range(sp['n_nodes'])]
        nx.draw(G, sp['pos'], ax=ax, edge_color=ec, width=ew, node_color=nc,
                with_labels=True, node_size=480, arrowsize=12, font_size=8)
        ax.set_title(sp['title'], fontsize=10)
    fig.suptitle(suptitle, y=1.03, fontsize=12)
    plt.tight_layout(); plt.show()
    return fig


def explain_message_stack(counts, labels, capacity=10, title="",
                          xlabel="", ylabel="messages in the target vector"):
    """Bar chart where each bar is a STACK of unit 'messages', with a capacity
    line — shows raw counts piling past what the vector can hold, and how kQ/I
    (or attention) keeps it manageable. `counts` = list, `labels` = x labels."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(len(counts))
    bars = ax.bar(x, counts, color=['#60a5fa' if c <= capacity else '#dc2626' for c in counts],
                  edgecolor='#334155')
    ax.axhline(capacity, ls='--', color='#334155', lw=1.5, label=f'vector capacity ({capacity})')
    for xi, c in zip(x, counts):
        ax.text(xi, c + 0.3, str(int(c)), ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.legend()
    plt.tight_layout(); plt.show()
    return fig

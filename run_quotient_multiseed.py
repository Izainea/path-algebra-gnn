"""Quotient ablation rerun: WalkNet over RAW A^g operators vs the EFFECTIVE
kQ/I operators, 5 seeds, per-seed rows (so the paper can report mean +- std).
The ONLY difference between the two models is the operator family; data,
architecture and optimiser recipe (AdamW + warmup + cosine + grad clip) are
shared. Run: python run_quotient_multiseed.py
Writes results/tables/quotient_ablation_5seeds.csv."""
import math, csv, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from oversquash.data_bottleneck import make_bottleneck_graph
from oversquash.transforms import AttachWalkOperators, collate_walk_operators
from oversquash.models import build_model

K, M = 5, 4
DEPTHS, SEEDS = [2, 3], [0, 1, 2, 3, 4]
N_GRAPHS, BS = 3000, 128
EPOCHS, LR, PATIENCE, WARMUP = 150, 2e-3, 30, 10
OUT = 'results/tables/quotient_ablation_5seeds.csv'

def ds(depth, n, seed):
    g = torch.Generator().manual_seed(seed + depth)
    return [make_bottleneck_graph(K, M, depth, g) for _ in range(n)]

def train_eval(m, tr, va, fwd):
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    def lr_lambda(ep):
        if ep < WARMUP: return (ep + 1) / WARMUP
        p = (ep - WARMUP) / max(1, EPOCHS - WARMUP)
        return 0.5 * (1 + math.cos(math.pi * p))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    def ev():
        m.eval(); acc = []
        with torch.no_grad():
            for b in va:
                lg, _ = fwd(b); mm = b.root_mask
                acc.append((lg[mm].argmax(-1) == b.y[mm]).float().mean().item())
        return float(np.mean(acc))
    best, bad = -1.0, 0
    for e in range(EPOCHS):
        m.train()
        for b in tr:
            opt.zero_grad(); lg, _ = fwd(b)
            F.cross_entropy(lg, b.y, ignore_index=-100).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step()
        v = ev()
        if v > best: best, bad = v, 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    return best

# Sanity check (memory gotcha): the eff and raw operators must differ at g>=2,
# otherwise the quotient model is a silent no-op and the comparison is invalid.
_g = torch.Generator().manual_seed(0)
_d = AttachWalkOperators(n_layers=4)(make_bottleneck_graph(K, M, 3, _g))
_diff = any((_d.walk_eff[i].to_dense() - _d.walk_raw[i].to_dense()).abs().max() > 1e-9
            for i in range(1, 4))
assert _diff, 'walk_eff == walk_raw at every range >= 2: invalid comparison'
print('sanity check passed: walk_eff != walk_raw at some range >= 2', flush=True)

rows, t00 = [], time.time()
for depth in DEPTHS:
    for seed in SEEDS:
        nl = depth + 1
        for name in ['walkraw', 'quotient']:
            torch.manual_seed(seed)
            data = ds(depth, N_GRAPHS, seed)
            tf = AttachWalkOperators(n_layers=nl)
            data = [tf(d) for d in data]
            tr = DataLoader(data[:2400], batch_size=BS, shuffle=True,
                            collate_fn=collate_walk_operators)
            va = DataLoader(data[2400:], batch_size=BS,
                            collate_fn=collate_walk_operators)
            fwd = lambda b: m(b.x, b.edge_index, getattr(b, 'batch', None),
                              walk_eff=b.walk_eff, walk_raw=b.walk_raw)
            in_dim, out_dim = data[0].x.size(-1), data[0].num_classes
            m = build_model(name, in_dim, 16, out_dim, nl)
            acc = train_eval(m, tr, va, fwd)
            rows.append({'depth': depth, 'seed': seed, 'model': name,
                         'val_acc': acc})
            print(f'[d={depth} s={seed}] {name:9s} acc={acc:.3f}  '
                  f'({(time.time()-t00)/60:.0f} min elapsed)', flush=True)

with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['depth', 'seed', 'model', 'val_acc'])
    w.writeheader()
    for r in rows: w.writerow(r)

print('\n=== SUMMARY (mean +- std over 5 seeds) ===')
import collections
agg = collections.defaultdict(list)
for r in rows: agg[(r['depth'], r['model'])].append(r['val_acc'])
for (d, mname), vs in sorted(agg.items()):
    print(f'  d={d} {mname:9s} {np.mean(vs):.3f} +- {np.std(vs, ddof=1):.3f}')
print('saved ' + OUT)

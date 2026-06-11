"""GCN / GIN on bottleneck retrieval, 5 seeds, same stabilized protocol as
run_bottleneck_multiseed.py. Empirical companion to the permutation-blindness
proposition (paper Sec. 'A formal account'): both models are PROVABLY at chance
(1/K = 0.20) for every depth/width, so these rows should read ~0.20 +- small.
Run: python run_bottleneck_gcn_gin.py
Writes results/tables/bottleneck_gcn_gin_5seeds.csv."""
import math, csv, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch, torch.nn.functional as F
from torch_geometric.loader import DataLoader as PyGLoader
from oversquash.data_bottleneck import make_bottleneck_graph
from oversquash.models import build_model

K, M = 5, 4
DEPTHS, SEEDS = [2, 3], [0, 1, 2, 3, 4]
N_GRAPHS, BS = 3000, 128
EPOCHS, LR, PATIENCE, WARMUP = 150, 2e-3, 30, 10
OUT = 'results/tables/bottleneck_gcn_gin_5seeds.csv'

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

rows, t00 = [], time.time()
for depth in DEPTHS:
    for seed in SEEDS:
        nl = depth + 1
        for name in ['gcn', 'gin']:
            torch.manual_seed(seed)
            data = ds(depth, N_GRAPHS, seed)
            tr = PyGLoader(data[:2400], batch_size=BS, shuffle=True)
            va = PyGLoader(data[2400:], batch_size=BS)
            fwd = lambda b: m(b.x, b.edge_index, getattr(b, 'batch', None))
            in_dim, out_dim = data[0].x.size(-1), data[0].num_classes
            m = build_model(name, in_dim, 16, out_dim, nl)
            acc = train_eval(m, tr, va, fwd)
            rows.append({'depth': depth, 'seed': seed, 'model': name,
                         'val_acc': acc})
            print(f'[d={depth} s={seed}] {name:4s} acc={acc:.3f}  '
                  f'({(time.time()-t00)/60:.0f} min elapsed)', flush=True)

with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['depth', 'seed', 'model', 'val_acc'])
    w.writeheader()
    for r in rows: w.writerow(r)

print('\n=== SUMMARY (mean +- std over 5 seeds; theory says 0.20) ===')
import collections
agg = collections.defaultdict(list)
for r in rows: agg[(r['depth'], r['model'])].append(r['val_acc'])
for (d, mname), vs in sorted(agg.items()):
    print(f'  d={d} {mname:4s} {np.mean(vs):.3f} +- {np.std(vs, ddof=1):.3f}')
print('saved ' + OUT)

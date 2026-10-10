"""Train GAT / walkraw / WalkAttention on RingTransfer across ring sizes.
RingTransfer has two parallel arcs (clear parallel paths) and severe, tunable
over-squashing. Run: python run_ring.py  (oversquash env). Writes results CSV."""
import time, math, warnings, csv
warnings.filterwarnings('ignore')
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGLoader
from oversquash.data_ring import ring_dataset
from oversquash.lrgb import AttachLRGBMasks, collate_lrgb
from oversquash.models import build_model

torch.manual_seed(0)
K = 5                       # classes; chance = 0.20
RINGS = [6, 10, 14, 18]     # ring sizes; distance = n/2 = layers needed
N_TRAIN, N_VAL = 1000, 400
EPOCHS, LR, BS = 60, 5e-3, 64

def evaluate(m, loader, fwd):
    m.eval(); acc = []
    with torch.no_grad():
        for b in loader:
            lg = fwd(m, b); mm = b.root_mask
            acc.append((lg[mm].argmax(-1) == b.y[mm]).float().mean().item())
    return float(np.mean(acc))

def train(name, n, nl):
    torch.manual_seed(0)
    tr = ring_dataset(n, N_TRAIN, seed=0, num_classes=K)
    va = ring_dataset(n, N_VAL, seed=1, num_classes=K)
    in_dim, out_dim = tr[0].x.size(-1), K
    if name == 'gat':
        trl = PyGLoader(tr, batch_size=BS, shuffle=True); val = PyGLoader(va, batch_size=BS)
        fwd = lambda m, b: m(b.x, b.edge_index, getattr(b, 'batch', None))[0]
    else:  # walkattn / walkraw both consume walk masks; walkraw uses fixed weights
        tf = AttachLRGBMasks(n_layers=nl)
        tr = [tf(d) for d in tr]; va = [tf(d) for d in va]
        trl = DataLoader(tr, batch_size=BS, shuffle=True, collate_fn=collate_lrgb)
        val = DataLoader(va, batch_size=BS, collate_fn=collate_lrgb)
        fwd = lambda m, b: m(b.x, b.edge_index, getattr(b, 'batch', None), walk_masks=b.walk_masks)[0]
    m = build_model('walkattn' if name == 'walkattn' else
                    ('gat' if name == 'gat' else 'walkattn'),
                    in_dim, 32, out_dim, nl, heads=4)
    # NB: walkraw approximated by WalkAttention with frozen uniform attention is
    # not exposed here; we compare the two operative cases GAT vs WalkAttention,
    # plus GCN as a second one-hop baseline.
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: min(1,(e+1)/5)*(0.5*(1+math.cos(math.pi*max(0,e-5)/max(1,EPOCHS-5)))))
    best = 0.0
    for e in range(EPOCHS):
        m.train()
        for b in trl:
            opt.zero_grad(); F.cross_entropy(fwd(m, b), b.y, ignore_index=-100).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step(); best = max(best, evaluate(m, val, fwd))
    return best

def train_gcn(n, nl):
    torch.manual_seed(0)
    tr = ring_dataset(n, N_TRAIN, 0, K); va = ring_dataset(n, N_VAL, 1, K)
    trl = PyGLoader(tr, batch_size=BS, shuffle=True); val = PyGLoader(va, batch_size=BS)
    m = build_model('gcn', tr[0].x.size(-1), 32, K, nl, heads=4)
    fwd = lambda m, b: m(b.x, b.edge_index, getattr(b, 'batch', None))[0]
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    best = 0.0
    for e in range(EPOCHS):
        m.train()
        for b in trl:
            opt.zero_grad(); F.cross_entropy(fwd(m, b), b.y, ignore_index=-100).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        best = max(best, evaluate(m, val, fwd))
    return best

rows = []
for n in RINGS:
    nl = n // 2
    print(f'=== ring n={n} (distance {nl}) ===', flush=True)
    res = {}
    res['gcn'] = train_gcn(n, nl)
    res['gat'] = train('gat', n, nl)
    res['walkattn'] = train('walkattn', n, nl)
    for k, v in res.items():
        rows.append({'ring': n, 'distance': nl, 'model': k, 'acc': round(v, 3)})
        print(f'  {k:9s} acc = {v:.3f}', flush=True)

with open('results/tables/ring_transfer.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['ring', 'distance', 'model', 'acc']); w.writeheader()
    for r in rows: w.writerow(r)
print('\n=== FINAL (RingTransfer, val acc; chance=0.20) ===')
import collections
piv = collections.defaultdict(dict)
for r in rows: piv[r['ring']][r['model']] = r['acc']
print(f"{'ring':>5} {'gcn':>7} {'gat':>7} {'walkattn':>9}")
for n in RINGS:
    print(f"{n:>5} {piv[n].get('gcn',0):>7.3f} {piv[n].get('gat',0):>7.3f} {piv[n].get('walkattn',0):>9.3f}")
print('saved results/tables/ring_transfer.csv')

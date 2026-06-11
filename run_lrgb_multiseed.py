"""LRGB Peptides-func rerun: GCN / GAT / WalkAttention, 4 seeds (the published
table was single-seed, so no variance could be reported). Same shared small
budget as before: 4 layers, hidden 80, no positional encodings.
Run: python run_lrgb_multiseed.py
Writes results/tables/lrgb_peptides_func_4seeds.csv (per-seed rows)."""
import time, math, warnings, csv
warnings.filterwarnings('ignore')
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.datasets import LRGBDataset
from oversquash.lrgb import AttachLRGBMasks, collate_lrgb, LRGBNet, average_precision

SEEDS = [0, 1, 2, 3]
N_LAYERS, HIDDEN, HEADS = 4, 80, 4
EPOCHS, LR, BS = 40, 3e-3, 64
DEV = 'cpu'
OUT = 'results/tables/lrgb_peptides_func_4seeds.csv'

print('loading + precomputing walk masks (matrix powers)...', flush=True)
t0 = time.time()
tf = AttachLRGBMasks(n_layers=N_LAYERS)
tr = [tf(d) for d in LRGBDataset(root='data/lrgb', name='Peptides-func', split='train')]
va = [tf(d) for d in LRGBDataset(root='data/lrgb', name='Peptides-func', split='val')]
te = [tf(d) for d in LRGBDataset(root='data/lrgb', name='Peptides-func', split='test')]
print(f'  {len(tr)}/{len(va)}/{len(te)} graphs, masks ready in {time.time()-t0:.0f}s',
      flush=True)

def loaders():
    return (DataLoader(tr, batch_size=BS, shuffle=True, collate_fn=collate_lrgb),
            DataLoader(va, batch_size=BS, collate_fn=collate_lrgb),
            DataLoader(te, batch_size=BS, collate_fn=collate_lrgb))

def train_one(backbone, seed):
    torch.manual_seed(seed)
    m = LRGBNet(backbone, HIDDEN, 10, N_LAYERS, n_heads=HEADS, dropout=0.1).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: min(1.0, (e + 1) / 5) *
        (0.5 * (1 + math.cos(math.pi * max(0, e - 5) / max(1, EPOCHS - 5)))))
    trl, val, tel = loaders()
    best_val, best_test = -1, -1
    for e in range(EPOCHS):
        m.train()
        for b in trl:
            b = b.to(DEV); opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(m(b), b.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step()
        v = average_precision(m, val, DEV)
        if v > best_val:
            best_val, best_test = v, average_precision(m, tel, DEV)
        if e % 10 == 0 or e == EPOCHS - 1:
            print(f'  [{backbone} s={seed}] epoch {e:2d}: val AP {v:.4f}  '
                  f'(best test {best_test:.4f})', flush=True)
    return best_val, best_test

rows = []
for seed in SEEDS:
    for bb in ['gcn', 'gat', 'walkattn']:
        print(f'=== {bb} seed {seed} ===', flush=True)
        t0 = time.time()
        bv, bt = train_one(bb, seed)
        rows.append({'seed': seed, 'model': bb, 'val_ap': round(bv, 4),
                     'test_ap': round(bt, 4),
                     'minutes': round((time.time() - t0) / 60, 1)})
        print(f'  -> {bb} s={seed}: val AP {bv:.4f}  test AP {bt:.4f}  '
              f'({(time.time()-t0)/60:.1f} min)', flush=True)

with open(OUT, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['seed', 'model', 'val_ap', 'test_ap', 'minutes'])
    w.writeheader()
    for r in rows: w.writerow(r)

print('\n=== SUMMARY (test AP, mean +- std over seeds) ===')
import collections
agg = collections.defaultdict(list)
for r in rows: agg[r['model']].append(r['test_ap'])
for mname, vs in sorted(agg.items()):
    print(f'  {mname:9s} {np.mean(vs):.4f} +- {np.std(vs, ddof=1):.4f}')
print('saved ' + OUT)

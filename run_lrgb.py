"""Train GCN / GAT / WalkAttention on LRGB Peptides-func and report test AP.
Run: python run_lrgb.py   (uses the oversquash conda env). Writes results CSV."""
import time, math, warnings, json
warnings.filterwarnings('ignore')
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.datasets import LRGBDataset
from oversquash.lrgb import AttachLRGBMasks, collate_lrgb, LRGBNet, average_precision

torch.manual_seed(0); np.random.seed(0)
N_LAYERS, HIDDEN, HEADS = 4, 80, 4
EPOCHS, LR, BS = 40, 3e-3, 64
DEV = 'cpu'

print('loading + precomputing walk masks (matrix powers)...', flush=True)
t0 = time.time()
tf = AttachLRGBMasks(n_layers=N_LAYERS)
tr = [tf(d) for d in LRGBDataset(root='data/lrgb', name='Peptides-func', split='train')]
va = [tf(d) for d in LRGBDataset(root='data/lrgb', name='Peptides-func', split='val')]
te = [tf(d) for d in LRGBDataset(root='data/lrgb', name='Peptides-func', split='test')]
print(f'  {len(tr)}/{len(va)}/{len(te)} graphs, masks ready in {time.time()-t0:.0f}s', flush=True)

def loaders():
    return (DataLoader(tr, batch_size=BS, shuffle=True, collate_fn=collate_lrgb),
            DataLoader(va, batch_size=BS, collate_fn=collate_lrgb),
            DataLoader(te, batch_size=BS, collate_fn=collate_lrgb))

def train_one(backbone):
    torch.manual_seed(0)
    m = LRGBNet(backbone, HIDDEN, 10, N_LAYERS, n_heads=HEADS, dropout=0.1).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: min(1.0,(e+1)/5) * (0.5*(1+math.cos(math.pi*max(0,e-5)/max(1,EPOCHS-5)))))
    trl, val, tel = loaders()
    best_val, best_test = -1, -1
    for e in range(EPOCHS):
        m.train()
        for b in trl:
            b = b.to(DEV); opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(m(b), b.y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step()
        v = average_precision(m, val, DEV)
        if v > best_val:
            best_val, best_test = v, average_precision(m, tel, DEV)
        if e % 5 == 0 or e == EPOCHS-1:
            print(f'  [{backbone}] epoch {e:2d}: val AP {v:.4f}  (best test {best_test:.4f})', flush=True)
    return best_val, best_test

rows = []
for bb in ['gcn', 'gat', 'walkattn']:
    print(f'=== {bb} ===', flush=True)
    t0 = time.time()
    bv, bt = train_one(bb)
    rows.append({'model': bb, 'val_ap': round(bv,4), 'test_ap': round(bt,4),
                 'minutes': round((time.time()-t0)/60,1)})
    print(f'  -> {bb}: val AP {bv:.4f}  test AP {bt:.4f}  ({(time.time()-t0)/60:.1f} min)', flush=True)

import csv
with open('results/tables/lrgb_peptides_func.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['model','val_ap','test_ap','minutes']); w.writeheader()
    for r in rows: w.writerow(r)
print('\n=== FINAL (Peptides-func, test AP) ===')
for r in rows: print(f"  {r['model']:9s}  test AP = {r['test_ap']:.4f}   val AP = {r['val_ap']:.4f}")
print('saved results/tables/lrgb_peptides_func.csv')

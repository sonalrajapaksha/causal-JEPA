"""Fixed-target tiny-batch overfit using training trajectories, no state input."""
import json, torch
from experiment3_temporal import Model, loader, OUT
torch.set_num_threads(2);torch.manual_seed(42)
x=next(iter(loader('train')))['video'][:2]
results={}
for kind in ('t0','t1','t2'):
 torch.manual_seed(42);m=Model(kind);opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.001)
 with torch.no_grad(): target=m.target(x[:,1:5]).detach()
 losses=[]
 for step in range(61):
  opt.zero_grad();p=m.predictor(m.relation(m.encoder(x[:,:4])));loss=(p-target).square().mean();loss.backward();opt.step()
  if step%10==0: losses.append({'step':step,'loss':loss.item()})
 results[kind]=losses
(OUT/'tiny_batch.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))

import sys,json
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ijepa.models import JEPA
from ijepa.data.pinball import CausalPinball
root=Path(__file__).resolve().parents[1]
ids=[]
for split in ['train','val','test']:
 d=CausalPinball(root/f'data/manifests/{split}.jsonl'); ids.append({r['simulation_id'] for r in d.records}); print(split,len(d))
assert all(not ids[i]&ids[j] for i in range(3) for j in range(i))
torch.set_num_threads(2)
for name in 'abc':
 m=JEPA(name,32); video=torch.rand(2,4,3,64,64); actions=torch.zeros(2,3,5); p,t=m(video,actions); assert p.shape==t.shape==(2,3,1 if name=='a' else 8,32); assert not t.requires_grad
 m.loss(video,actions).backward(); assert all(x.grad is None for x in m.target.parameters()); old=[p.detach().clone() for p in m.target.parameters()]
 with torch.no_grad():
  for x in m.encoder.parameters(): x.add_(1)
 m.update_target(.9)
 for before,on,target in zip(old,m.encoder.parameters(),m.target.parameters()): torch.testing.assert_close(target,before*.9+on*.1)
 print(name,'shapes, stop-gradient, EMA passed')
print('Transition alignment: see scripts/verify_replay.py; actions[t] advances frame t to t+1.')

"""Experiment 3 supervised audit. Never writes to earlier experiments."""
import json, sys, time, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ijepa.models import JEPA
from collapse_experiment import ROOT, labels, state_stats, penalties, fit, predict, scores
from ijepa.data.pinball import CausalPinball
OUT=ROOT/'outputs/experiment3'; OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2); torch.manual_seed(42); np.random.seed(42); random.seed(42)
device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
def loader(split,shuffle=False):
 return DataLoader(CausalPinball(ROOT/f'data/manifests/{split}.jsonl',context=4,prediction=1),batch_size=32,shuffle=shuffle)
mu,sd=state_stats(); mu=mu.to(device); sd=sd.to(device)
b=next(iter(loader('train'))); v=b['video'].to(device); y=labels(b).to(device)
m=JEPA('b',32).to(device); ck=torch.load(ROOT/'outputs/experiment2/b3_seed42/final.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);head=nn.Linear(32,4).to(device);head.load_state_dict(ck['head']);
p,t=m(v[:,:4],b['interventions'][:,:3].to(device));z=m.encoder(v[:,:3]); terms={'prediction':nn.functional.mse_loss(p,t),'regularization':penalties(z,'b3'),'supervision':nn.functional.mse_loss(head(z.mean(2)),(y[:,:3]-mu)/sd)}
grad={}
for name,loss in terms.items():
 m.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True);loss.backward(retain_graph=True)
 grad[name]={'loss':loss.item(),'encoder_grad_norm':sum(float(x.grad.square().sum()) for x in m.encoder.parameters() if x.grad is not None)**.5,'target_grad_present':any(x.grad is not None for x in m.target.parameters())}
result={'b3_terms':grad,'state_keys':[x[0] for x in b['state_keys']],'target_shape':list(y.shape),'slot_shape':list(z.shape),'head_shape':list(head(z.mean(2)).shape),'normalization_roundtrip_max_error':float((((y-mu)/sd)*sd+mu-y).abs().max()),'state_mean':mu.cpu().tolist(),'state_std':sd.cpu().tolist()}
# Compare train-fitted ridge probes at the exact same time points.
features={}
with torch.no_grad():
 for split in ('train','val'):
  a=[]; ys=[]
  for batch in loader(split):
   zz=m.encoder(batch['video'][:,:4].to(device)).cpu().numpy();a.append(zz);ys.append(labels(batch)[:,:4].numpy())
  features[split]=(np.concatenate(a),np.concatenate(ys))
tr,yt=features['train']; va,yv=features['val']; result['slot_probe_validation']={}
for name,x,xv in [('mean',tr.mean(2),va.mean(2)),('concatenated',tr.reshape(*tr.shape[:2],-1),va.reshape(*va.shape[:2],-1))]:
 result['slot_probe_validation'][name]=scores(yv,predict(xv,fit(x,yt)))
class CNN(nn.Module):
 def __init__(self,frames,outputs):
  super().__init__();self.skip=nn.Linear(frames*3*8*8,outputs);self.net=nn.Sequential(nn.Conv2d(3*frames,32,5,2,2),nn.ReLU(),nn.Conv2d(32,32,3,2,1),nn.ReLU(),nn.Conv2d(32,32,3,2,1),nn.ReLU(),nn.Flatten(),nn.Linear(32*8*8,128),nn.ReLU(),nn.Linear(128,outputs))
 def forward(self,x):
  x=x.flatten(1,2);return self.net(x)+self.skip(nn.functional.avg_pool2d(x,8).flatten(1))
def inputs(b,n): return b['video'][:,4-n:4].to(device),labels(b)[:,3].to(device)
# Fixed batch position overfit; no test set involved.
net=CNN(1,2).to(device);opt=torch.optim.Adam(net.parameters(),lr=.001); x,yy=inputs(b,1); target=(yy[:,:2]-mu[:2])/sd[:2];hist=[]
for step in range(301):
 opt.zero_grad(); loss=nn.functional.mse_loss(net(x),target);loss.backward();opt.step()
 if step%50==0: hist.append({'step':step,'mse':loss.item()})
result['single_frame_overfit']=hist
# Full train split, validation-selected, independent supervised models.
for n in (1,2):
 net=CNN(n,4).to(device)
 # Train-only ridge initialization of the pooled-pixel skip, with a zero residual CNN.
 xx=[]; yy=[]
 for batch in loader('train'):
  xb,yb=inputs(batch,n);xx.append(nn.functional.avg_pool2d(xb.flatten(1,2),8).flatten(1).cpu().numpy());yy.append(((yb-mu)/sd).cpu().numpy())
 xm,xs,ym,w=fit(np.concatenate(xx),np.concatenate(yy),alpha=1000)
 with torch.no_grad():
  net.skip.weight.copy_(torch.tensor((w/xs[:,None]).T,dtype=torch.float32,device=device));net.skip.bias.copy_(torch.tensor(ym-(xm/xs)@w,dtype=torch.float32,device=device));net.net[-1].weight.zero_();net.net[-1].bias.zero_()
 opt=torch.optim.Adam(net.parameters(),lr=.0001); best=float('inf'); history=[]; started=time.time()
 for epoch in range(8):
  net.train(); total=0
  for batch in loader('train',True):
   x,yy=inputs(batch,n); opt.zero_grad(); loss=nn.functional.mse_loss(net(x),(yy-mu)/sd);loss.backward();opt.step(); total+=loss.item()*len(x)
  net.eval(); err=0
  with torch.no_grad():
   for batch in loader('val'):
    x,yy=inputs(batch,n); err+=nn.functional.mse_loss(net(x),(yy-mu)/sd,reduction='sum').item()
  val=err/(300*4); history.append({'epoch':epoch+1,'train_mse':total/2400,'val_mse':val})
  if val<best:
   best=val;torch.save(net.state_dict(),OUT/f'supervised_{n}frame.pt'); bestepoch=epoch+1
  print('supervised',n,history[-1],flush=True)
 net.load_state_dict(torch.load(OUT/f'supervised_{n}frame.pt',map_location=device,weights_only=True)); net.eval(); pred=[]; truth=[]; masks=[]
 with torch.no_grad():
  for batch in loader('test'):
   x,yy=inputs(batch,n);pred.append((net(x)*sd+mu).cpu().numpy());truth.append(yy.cpu().numpy());masks.append(batch['interventions'][:,2].bool().any(-1).numpy())
 pred=np.concatenate(pred);truth=np.concatenate(truth);mask=np.concatenate(masks)
 result[f'supervised_{n}frame']={'history':history,'best_epoch':bestepoch,'seconds':time.time()-started,'test':{name:scores(truth[k],pred[k]) for name,k in [('all',np.ones(len(mask),bool)),('intervention',mask),('nonintervention',~mask)]}}
 (OUT/'supervised_audit.json').write_text(json.dumps(result,indent=2))
(OUT/'supervised_audit.json').write_text(json.dumps(result,indent=2))

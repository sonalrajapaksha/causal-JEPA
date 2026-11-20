"""Controlled single-seed experiment; no simulator state is used by JEPA training."""
import argparse, copy, csv, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from ijepa.data.pinball import CausalPinball
from ijepa.models import JEPA

ROOT=Path(__file__).resolve().parents[1]
def loader(split, n=None, batch=8, shuffle=False):
 d=CausalPinball(ROOT/f'data/manifests/{split}.jsonl',context=2,prediction=2)
 if n: d=Subset(d,range(min(n,len(d))))
 return DataLoader(d,batch_size=batch,shuffle=shuffle,num_workers=0)

def reg(z):
 # Variance across independent trajectories and frames, not across slot identities.
 z=z.mean(dim=2).reshape(-1,z.shape[-1])
 return torch.relu(0.1-torch.sqrt(z.var(dim=0,unbiased=False)+1e-8)).mean()

def cov_penalty(z):
 z=z.mean(dim=2).reshape(-1,z.shape[-1]); z=z-z.mean(0)
 cov=z.T@z/max(len(z)-1,1)
 off=cov-torch.diag_embed(torch.diagonal(cov))
 return off.square().sum()/z.shape[-1]

def run(a):
 random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(2)
 device=torch.device(a.device if a.device!='auto' else ('mps' if torch.backends.mps.is_available() else 'cpu'))
 variant='c' if a.model in ('c','b_info') else a.model
 model=JEPA(variant,dim=32).to(device); opt=torch.optim.AdamW(model.parameters(),lr=0.0003)
 train=loader('train',a.train_limit,a.batch,True); val=loader('val',a.val_limit,a.batch)
 out=ROOT/'outputs/experiments'/a.name; out.mkdir(parents=True,exist_ok=False)
 config=vars(a).copy(); config.update({'architecture_variant':variant,'variance_floor':0.1,'variance_weight':10.0,'covariance_weight':100.0,'momentum':0.99,'frames':4,'learning_rate':0.0003,'device_used':str(device),'train_examples':len(train.dataset),'validation_examples':len(val.dataset)})
 (out/'config.json').write_text(json.dumps(config,indent=2))
 history=[]; best=float('inf'); best_epoch=0; peak_mps=0
 for epoch in range(a.epochs):
  model.train(); total=0; n=0; started=time.time()
  for b in train:
   v=b['video'].to(device); act=b['interventions'].to(device); opt.zero_grad(set_to_none=True)
   pred,target=model(v,act); mse=torch.nn.functional.mse_loss(pred,target); features=model.encoder(v[:,:-1]); variance=reg(features); loss=mse+10*variance+100*cov_penalty(features)
   loss.backward(); opt.step(); model.update_target(); total+=loss.item()*len(v); n+=len(v)
   if device.type == 'mps': peak_mps=max(peak_mps,torch.mps.driver_allocated_memory())
  model.eval(); val_mse=0; count=0; zs=[]
  with torch.no_grad():
   for b in val:
    v=b['video'].to(device); act=b['interventions'].to(device); p,t=model(v,act); val_mse+=torch.nn.functional.mse_loss(p,t,reduction='sum').item(); count+=t.numel(); zs.append(model.encoder(v).mean(2).reshape(-1,32).cpu())
  z=torch.cat(zs); std=z.std(0,unbiased=False).mean().item(); vm=val_mse/count
  val_variance=torch.relu(0.1-torch.sqrt(z.var(0,unbiased=False)+1e-8)).mean().item(); score=vm+10*val_variance+100*cov_penalty(z.reshape(1,-1,1,32)).item()
  row={'epoch':epoch+1,'train_total':total/n,'val_mse':vm,'val_std':std,'val_total':score,'peak_mps_driver_bytes':peak_mps,'seconds':time.time()-started}; history.append(row)
  if score < best:
   best=score; best_epoch=epoch+1
   torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'config':config,'seed':a.seed},out/'best.pt')
  print(a.name,json.dumps(row),flush=True)
  (out/'history.json').write_text(json.dumps(history,indent=2))
  if epoch+1-best_epoch>=a.patience: break
 print('best_epoch',best_epoch,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--model',choices=['a','b','b_info','c'],required=True); p.add_argument('--name',required=True); p.add_argument('--epochs',type=int,default=3); p.add_argument('--patience',type=int,default=10); p.add_argument('--batch',type=int,default=8); p.add_argument('--train-limit',type=int); p.add_argument('--val-limit',type=int); p.add_argument('--seed',type=int,default=42); p.add_argument('--device',default='auto'); run(p.parse_args())

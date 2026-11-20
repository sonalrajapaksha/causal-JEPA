"""Matched Model B collapse study. B3 uses state labels and is supervised."""
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ijepa.models import JEPA
from ijepa.data.pinball import CausalPinball
from controlled_experiment import reg, cov_penalty

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/experiment2'
KEYS=['ball_x','ball_y','ball_x_vel','ball_y_vel']
def loader(split,batch=16,shuffle=False):
 return DataLoader(CausalPinball(ROOT/f'data/manifests/{split}.jsonl',context=2,prediction=2),batch_size=batch,shuffle=shuffle,num_workers=0)
def labels(b):
 keys=[x[0] for x in b['state_keys']]; return b['states'][:,:,[keys.index(k) for k in KEYS]]
def state_stats():
 y=torch.cat([labels(b).reshape(-1,4) for b in loader('train')]); return y.mean(0),y.std(0).clamp_min(1e-6)
def penalties(z,variant):
 if variant!='b2': return 10*reg(z)+100*cov_penalty(z)
 # Normalize each pooled vector across features; preserve gradients, avoid fixed scale inflation.
 x=F.layer_norm(z.mean(2),(z.shape[-1],)).reshape(-1,z.shape[-1]); x=x-x.mean(0)
 std=torch.sqrt(x.var(0,unbiased=False)+1e-4); cov=x.T@x/max(x.shape[0]-1,1); off=cov-torch.diag_embed(cov.diagonal())
 return 10*F.relu(0.5-std).mean()+off.square().sum()/x.shape[1]
def train(a):
 random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(2)
 device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); m=JEPA('b',32).to(device)
 # The head predicts global ball state from pooled slots: it does not assume a slot identity.
 rng=torch.random.get_rng_state()
 head=torch.nn.Linear(32,4).to(device) if a.variant=='b3' else None
 torch.random.set_rng_state(rng)
 opt=torch.optim.AdamW(list(m.parameters())+(list(head.parameters()) if head else []),lr=0.0003)
 mu,sd=state_stats(); mu=mu.to(device); sd=sd.to(device)
 out=OUT/f'{a.variant}_seed{a.seed}'; out.mkdir(parents=True,exist_ok=False)
 history=[]
 for epoch in range(5):
  start=time.time(); m.train(); total=0
  for b in loader('train',8,True):
   v=b['video'].to(device); act=b['interventions'].to(device); opt.zero_grad(set_to_none=True)
   p,t=m(v,act); z=m.encoder(v[:,:-1]); loss=F.mse_loss(p,t)+penalties(z,a.variant)
   if head: loss=loss+F.mse_loss(head(z.mean(2)),(labels(b)[:,:-1].to(device)-mu)/sd)
   loss.backward(); opt.step(); m.update_target(); total+=loss.item()*len(v)
  # Fixed final epoch checkpoint for all variants: no objective-scale selection bias.
  row={'epoch':epoch+1,'train_objective':total/2400,'seconds':time.time()-start}; history.append(row); print(a.variant,a.seed,json.dumps(row),flush=True)
  (out/'history.json').write_text(json.dumps(history,indent=2))
  torch.save({'model':m.state_dict(),'head':head.state_dict() if head else None,'seed':a.seed,'variant':a.variant,'epoch':epoch+1,'state_mean':mu.cpu(),'state_std':sd.cpu()},out/'final.pt')

def collect(m,split,device):
 Z=[]; Y=[]; E=[]; P=[]; I=[]; slot=[]
 with torch.no_grad():
  for b in loader(split):
   v=b['video'].to(device); a=b['interventions'].to(device); z=m.encoder(v); p,t=m(v,a); cur=m.target(v[:,:-1]);
   Z.append(z.mean(2).cpu().numpy()); Y.append(labels(b).numpy()); E.append((p-t).square().mean((-1,-2)).cpu().numpy()); P.append((cur-t).square().mean((-1,-2)).cpu().numpy()); I.append(a.bool().any(-1).cpu().numpy()); slot.append((z-z.mean(2,keepdim=True)).square().mean().item())
 return [np.concatenate(x) for x in (Z,Y,E,P,I)],float(np.mean(slot))
def fit(x,y,alpha=100):
 x=x.reshape(-1,x.shape[-1]).astype(np.float64); y=y.reshape(-1,4).astype(np.float64); mu=x.mean(0); sd=x.std(0)+1e-6; ym=y.mean(0); X=(x-mu)/sd
 return mu,sd,ym,np.linalg.solve(X.T@X+alpha*np.eye(X.shape[1]),X.T@(y-ym))
def predict(x,model):
 mu,sd,ym,w=model; return (x-mu)/sd@w+ym
def scores(y,p):
 y=y.reshape(-1,4); p=p.reshape(-1,4); return {'r2':(1-((p-y)**2).sum(0)/((y-y.mean(0))**2).sum(0)).tolist(),'rmse':np.sqrt(((p-y)**2).mean(0)).tolist()}
def evaluate(a):
 torch.set_num_threads(2); device='mps' if torch.backends.mps.is_available() else 'cpu'; results={}
 for variant in ['b1','b2','b3']:
  for seed in [42,43,44]:
   name=f'{variant}_seed{seed}'; path=OUT/name/'final.pt'
   if not path.exists(): continue
   ck=torch.load(path,map_location='cpu',weights_only=False); m=JEPA('b',32).to(device);m.load_state_dict(ck['model']);m.eval(); train,_=collect(m,'train',device); test,slot=collect(m,'test',device)
   z,y,e,p,i=test; eig=np.clip(np.linalg.eigvalsh(np.cov(z.reshape(-1,32),rowvar=False)),0,None); q=eig/(eig.sum()+1e-30); q=q[q>0]
   pr=predict(z,fit(train[0],train[1])); r={'std':float(z.reshape(-1,32).std(0).mean()),'effective_rank':float(np.exp(-(q*np.log(q)).sum())),'slot_dispersion':slot,'probe':scores(y,pr),'subsets':{}}
   for label,mask in [('all',np.ones_like(i,dtype=bool)),('intervention',i),('nonintervention',~i)]:
    r['subsets'][label]={'count':int(mask.sum()),'prediction_mse':float(e[mask].mean()),'persistence_mse':float(p[mask].mean()),'ratio':float(e[mask].mean()/p[mask].mean()),'probe_next_state':scores(y[:,1:][mask],pr[:,1:][mask])}
   results[name]=r; print('evaluated',name,flush=True); (OUT/'metrics.json').write_text(json.dumps(results,indent=2))

def baseline(a):
 # Train-only ridge on downsampled RGB, and on consecutive frame differences.
 import torch.nn.functional as fn
 data={}
 for split in ['train','test']:
  X=[]; Y=[]; I=[]
  for b in loader(split):
   v=b['video']; small=fn.avg_pool2d(v.reshape(-1,3,64,64),8).reshape(len(v),4,-1).numpy(); X.append(small);Y.append(labels(b).numpy());I.append(b['interventions'].bool().any(-1).numpy())
  data[split]=[np.concatenate(x) for x in (X,Y,I)]
 x,y,i=data['train']; xt,yt,it=data['test']; results={}
 for mode in ['single_frame','two_frame']:
  if mode=='single_frame': tr=x[:,1:]; te=xt[:,1:]
  else: tr=np.concatenate([x[:,1:],x[:,1:]-x[:,:-1]],-1);te=np.concatenate([xt[:,1:],xt[:,1:]-xt[:,:-1]],-1)
  pred=predict(te,fit(tr,y[:,1:],alpha=1000)); results[mode]={s:scores(yt[:,1:][mask],pred[mask]) for s,mask in [('all',np.ones_like(it,bool)),('intervention',it),('nonintervention',~it)]}
 # Simulator positions reveal whether sampled motion contains velocity information.
 pos_train=np.concatenate([y[:,1:,:2], y[:,1:,:2]-y[:,:-1,:2]],-1); pos_test=np.concatenate([yt[:,1:,:2], yt[:,1:,:2]-yt[:,:-1,:2]],-1)
 pred=predict(pos_test,fit(pos_train,y[:,1:],alpha=100)); results['oracle_two_position_velocity']={s:scores(yt[:,1:][mask],pred[mask]) for s,mask in [('all',np.ones_like(it,bool)),('intervention',it),('nonintervention',~it)]}
 # Supervised oracle: current state -> next state, trained only on train sequences.
 pred=predict(yt[:,:-1],fit(y[:,:-1],y[:,1:],alpha=100)); results['supervised_state_transition']={s:scores(yt[:,1:][mask],pred[mask]) for s,mask in [('all',np.ones_like(it,bool)),('intervention',it),('nonintervention',~it)]}
 results['state_persistence']={s:scores(yt[:,1:][mask],yt[:,:-1][mask]) for s,mask in [('all',np.ones_like(it,bool)),('intervention',it),('nonintervention',~it)]}
 (OUT/'baselines.json').write_text(json.dumps(results,indent=2)); print(json.dumps(results,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['train','evaluate','baseline']);p.add_argument('--variant',choices=['b1','b2','b3']);p.add_argument('--seed',type=int,default=42);a=p.parse_args(); OUT.mkdir(parents=True,exist_ok=True); {'train':train,'evaluate':evaluate,'baseline':baseline}[a.action](a)

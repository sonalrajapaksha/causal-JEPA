"""Four-frame context JEPA comparisons; train-only ridge and validation checkpoint selection."""
import argparse, copy, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ijepa.models.jepa import Encoder
from ijepa.data.pinball import CausalPinball
from collapse_experiment import ROOT, labels, fit, predict, scores
from controlled_experiment import reg, cov_penalty
OUT=ROOT/'outputs/experiment3'
torch.set_num_threads(2)
class Context(nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind;self.frame=Encoder(32,True)
  if kind=='t1': self.temporal=nn.Sequential(nn.Conv1d(32,32,3,padding=1),nn.GELU(),nn.Conv1d(32,32,3,padding=2,dilation=2))
  if kind=='t2':
   self.pos=nn.Parameter(torch.randn(1,4,32)*.02);self.temporal=nn.TransformerEncoderLayer(32,4,64,dropout=0,batch_first=True)
 def forward(self,x):
  z=self.frame(x)
  if self.kind=='t0': return z[:,-1]
  b,t,s,d=z.shape; h=z.permute(0,2,1,3).reshape(b*s,t,d)
  if self.kind=='t1': h=self.temporal(h.transpose(1,2)).transpose(1,2)+h
  else: h=self.temporal(h+self.pos)
  return h[:,-1].reshape(b,s,d)
class Model(nn.Module):
 def __init__(self,kind):
  super().__init__();self.encoder=Context(kind);self.target=copy.deepcopy(self.encoder);self.target.requires_grad_(False)
  self.relation=nn.TransformerEncoderLayer(32,4,64,dropout=0,batch_first=True);self.predictor=nn.Sequential(nn.Linear(32,32),nn.GELU(),nn.Linear(32,32))
 def forward(self,v):
  z=self.encoder(v[:,:4]); p=self.predictor(self.relation(z))
  with torch.no_grad(): t=self.target(v[:,1:5]);cur=self.target(v[:,:4])
  return z,p,t,cur
 @torch.no_grad()
 def update(self):
  for a,b in zip(self.encoder.parameters(),self.target.parameters()): b.lerp_(a,.01)
def loader(split,shuffle=False): return DataLoader(CausalPinball(ROOT/f'data/manifests/{split}.jsonl',context=4,prediction=1),batch_size=16,shuffle=shuffle)
def loss(z,p,t): return nn.functional.mse_loss(p,t)+10*reg(z[:,None])+100*cov_penalty(z[:,None])
def train(kind,seed,epochs,device):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);m=Model(kind).to(device); opt=torch.optim.AdamW(m.parameters(),lr=.0003);out=OUT/f'{kind}_seed{seed}';out.mkdir(exist_ok=False);history=[];best=float('inf');started=time.time()
 for epoch in range(epochs):
  m.train(); total=0; start=time.time()
  for b in loader('train',True):
   v=b['video'].to(device);opt.zero_grad(set_to_none=True);z,p,t,_=m(v);l=loss(z,p,t);l.backward();opt.step();m.update();total+=l.item()*len(v)
  m.eval();totalv=0
  with torch.no_grad():
   for b in loader('val'):
    v=b['video'].to(device);z,p,t,_=m(v);totalv+=loss(z,p,t).item()*len(v)
  score=totalv/300;row={'epoch':epoch+1,'train_loss':total/2400,'val_loss':score,'seconds':time.time()-start};history.append(row)
  if score<best:
   best=score;torch.save({'model':m.state_dict(),'kind':kind,'seed':seed,'epoch':epoch+1},out/'best.pt')
  (out/'history.json').write_text(json.dumps(history,indent=2));print(kind,seed,row,flush=True)
 return time.time()-started
def extract(m,split,device):
 arr={k:[] for k in ['z','y','p','persist','intervention','reverse','repeat','shuffle','z_next','y_next']};m.eval()
 with torch.no_grad():
  for b in loader(split):
   v=b['video'].to(device);z,p,t,c=m(v);arr['z'].append(z.mean(1).cpu().numpy());arr['y'].append(labels(b)[:,3].numpy());arr['p'].append((p-t).square().mean((1,2)).cpu().numpy());arr['persist'].append((c-t).square().mean((1,2)).cpu().numpy());arr['intervention'].append(b['interventions'][:,3].bool().any(-1).numpy());arr['z_next'].append(m.encoder(v[:,1:5]).mean(1).cpu().numpy());arr['y_next'].append(labels(b)[:,4].numpy())
   for name,idx in [('reverse',[3,2,1,0]),('repeat',[3,3,3,3]),('shuffle',[2,0,1,3])]: arr[name].append(m.encoder(v[:,idx]).mean(1).cpu().numpy())
 return {k:np.concatenate(v) for k,v in arr.items()}
def evaluate(kind,seed,device):
 path=OUT/f'{kind}_seed{seed}';ck=torch.load(path/'best.pt',map_location='cpu',weights_only=False);m=Model(kind).to(device);m.load_state_dict(ck['model']);tr=extract(m,'train',device);te=extract(m,'test',device);probe=fit(tr['z'],tr['y']);z=te['z'];y=te['y'];pred=predict(z,probe);eig=np.clip(np.linalg.eigvalsh(np.cov(z,rowvar=False)),0,None);q=eig/(eig.sum()+1e-30);q=q[q>0];res={'best_epoch':ck['epoch'],'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),'std':float(z.std(0).mean()),'effective_rank':float(np.exp(-(q*np.log(q)).sum())),'probe':scores(y,pred),'subsets':{},'order':{}}
 # latent error is for 3 -> 4; probe uses 3, so subset probe uses transition 2 -> 3.
 for name,mask in [('all',np.ones(len(z),bool)),('intervention',te['intervention']),('nonintervention',~te['intervention'])]: res['subsets'][name]={'count':int(mask.sum()),'prediction_to_persistence_ratio':float(te['p'][mask].mean()/te['persist'][mask].mean()),'probe_next_state':scores(te['y_next'][mask],predict(te['z_next'][mask],probe))}
 for name in ['reverse','repeat','shuffle']:
  zz=te[name];res['order'][name]={'representation_rms_change':float(np.sqrt(np.mean((zz-z)**2))),'representation_change_over_std':float(np.sqrt(np.mean((zz-z)**2))/(z.std(0).mean()+1e-12)),'probe_against_original_state':scores(y,predict(zz,probe))}
 (path/'metrics.json').write_text(json.dumps(res,indent=2));return res
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=5);parser.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);args=parser.parse_args();OUT.mkdir(exist_ok=True,parents=True);device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');results={}
 for seed in args.seeds:
  for kind in ['t0','t1','t2']:
   elapsed=train(kind,seed,args.epochs,device);results[f'{kind}_seed{seed}']=evaluate(kind,seed,device);results[f'{kind}_seed{seed}']['total_train_seconds']=elapsed;(OUT/'temporal_metrics.json').write_text(json.dumps(results,indent=2))

"""Frozen representation diagnostic. Run once; plot/report are reproducible from metrics.json."""
import sys, json, random, copy, hashlib, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ijepa.data.pinball import CausalPinball
from ijepa.models import JEPA
from experiment3_temporal import Model as Temporal
from collapse_experiment import labels, fit, predict
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/experiment4'; OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2); DEVICE=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
SPLITS=['train','val','test']; TARGETS=['ball_x','ball_y','ball_x_vel','ball_y_vel']; SEEDS=[11,22,33]
MODELS={**{f'full_{x}':('jepa',x,f'outputs/experiments/full_{x}/best.pt') for x in 'abc'}, **{x:('jepa','b',f'outputs/experiment2/{x}_seed42/final.pt') for x in ['b1','b2','b3']},**{x:('temporal',x,f'outputs/experiment3/{x}_seed42/best.pt') for x in ['t0','t1','t2']}}
CONFIG={'models':MODELS,'probe_seeds':SEEDS,'encoder_seed':42,'target_timestep':3,'context_frames':{'framewise':[3],'t0':[3],'t1':[0,1,2,3],'t2':[0,1,2,3]},'intervention_transition':'2 to 3','ridge_alpha':100,'pixel_ridge_alpha':1000,'mlp_hidden':[128,64],'activation':'GELU','optimizer':'AdamW','learning_rate':0.001,'weight_decay':0.01,'batch_size':256,'max_epochs':100,'patience':15,'min_delta':0.0001,'target_order':TARGETS,'device':str(DEVICE),'torch_version':torch.__version__,'selection':'fixed checkpoint and hyperparameters; early stop on mean validation standardized target MSE'}
(OUT/'config.json').write_text(json.dumps(CONFIG,indent=2))
def load_data():
 data={}; idsets=[]; seedsets=[]
 for split in SPLITS:
  ds=CausalPinball(ROOT/f'data/manifests/{split}.jsonl',context=4,prediction=1); ids=[r['id'] for r in ds.records]; sim=[r['simulation_id'] for r in ds.records]
  assert len(set(ids))==len(ids) and len(set(sim))==len(sim); idsets.append(set(sim))
  seeds=[]
  for rec in ds.records:
   with np.load(ROOT/rec['path']) as a: seeds.append(int(a['seed']))
  seedsets.append(set(seeds)); data[split]={'ids':ids,'simulation_ids':sim,'simulator_seeds':seeds}
 assert all(not idsets[i]&idsets[j] for i in range(3) for j in range(i))
 assert all(not seedsets[i]&seedsets[j] for i in range(3) for j in range(i)), 'duplicate simulator RNG seeds across splits'
 return data

def base_cache(data):
 path=OUT/'data.npz'
 if path.exists(): return dict(np.load(path,allow_pickle=False))
 out={}
 for split in SPLITS:
  arr={k:[] for k in ['y','intervention','single','temporal']}
  ds=CausalPinball(ROOT/f'data/manifests/{split}.jsonl',context=4,prediction=1)
  for b in DataLoader(ds,batch_size=32,shuffle=False):
   v=b['video']; p=nn.functional.avg_pool2d(v.reshape(-1,3,64,64),8).reshape(len(v),5,-1).numpy()
   arr['single'].append(p[:,3]); arr['temporal'].append(np.concatenate([p[:,3],p[:,3]-p[:,2]],1)); arr['y'].append(labels(b)[:,3].numpy()); arr['intervention'].append(b['interventions'][:,2].bool().any(-1).numpy())
  for k,vs in arr.items(): out[f'{split}_{k}']=np.concatenate(vs)
  for k in ['ids','simulation_ids','simulator_seeds']: out[f'{split}_{k}']=np.asarray(data[split][k])
 np.savez_compressed(path,**out); return out

def extract(name,info,data):
 path=OUT/f'{name}_representations.npz'
 if path.exists(): return dict(np.load(path))
 typ,kind,ckpath=info; ck=torch.load(ROOT/ckpath,map_location='cpu',weights_only=False)
 model=Temporal(kind) if typ=='temporal' else JEPA(kind,32); model.load_state_dict(ck['model']); enc=model.encoder.to(DEVICE).eval().requires_grad_(False)
 before=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in enc.parameters())).hexdigest(); out={}
 for split in SPLITS:
  ds=CausalPinball(ROOT/f'data/manifests/{split}.jsonl',context=4,prediction=1); pooled=[]; flat=[]; ids=[]
  with torch.inference_mode():
   for b in DataLoader(ds,batch_size=32,shuffle=False):
    v=b['video'].to(DEVICE); z=enc(v[:,:4]) if typ=='temporal' else enc(v[:,3:4])[:,0]
    pooled.append(z.mean(1).cpu().numpy()); flat.append(z.flatten(1).cpu().numpy()); ids.extend(b['id'])
  assert ids==data[split]['ids']; out[split+'_pooled']=np.concatenate(pooled)
  if kind!='a': out[split+'_flat']=np.concatenate(flat)
 after=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in enc.parameters())).hexdigest(); assert before==after and all(not p.requires_grad and p.grad is None for p in enc.parameters())
 out['checkpoint_sha256']=np.array(hashlib.sha256((ROOT/ckpath).read_bytes()).hexdigest()); out['encoder_sha256']=np.array(before); out['checkpoint_epoch']=np.array(ck['epoch']); np.savez_compressed(path,**out); return out

def metrics(y,p):
 err=p-y; den=((y-y.mean(0))**2).sum(0); return {'r2':(1-(err**2).sum(0)/den).tolist(),'mse':(err**2).mean(0).tolist(),'mae':np.abs(err).mean(0).tolist(),'prediction_std':p.std(0).tolist()}
def net(d): return nn.Sequential(nn.Linear(d,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,4))
def mlp(x,y,seed):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 xm=x['train'].mean(0); xs=x['train'].std(0)+1e-6; ym=y['train'].mean(0); ys=y['train'].std(0).clip(1e-6)
 xx={s:torch.tensor((x[s]-xm)/xs,dtype=torch.float32,device=DEVICE) for s in SPLITS}; yy={s:torch.tensor((y[s]-ym)/ys,dtype=torch.float32,device=DEVICE) for s in SPLITS}
 m=net(xm.size).to(DEVICE); opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01); best=float('inf'); best_state=None; best_epoch=0; hist=[]; gen=torch.Generator().manual_seed(seed)
 for epoch in range(100):
  m.train()
  for idx in torch.randperm(len(xx['train']),generator=gen).split(256):
   idx=idx.to(DEVICE); opt.zero_grad(set_to_none=True); loss=nn.functional.mse_loss(m(xx['train'][idx]),yy['train'][idx]); loss.backward(); opt.step()
  m.eval()
  with torch.no_grad(): tr=nn.functional.mse_loss(m(xx['train']),yy['train']).item(); val=nn.functional.mse_loss(m(xx['val']),yy['val']).item()
  hist.append([tr,val])
  if val<best-0.0001: best=val; best_epoch=epoch+1; best_state=copy.deepcopy(m.state_dict())
  if epoch+1-best_epoch>=15: break
 m.load_state_dict(best_state); m.eval()
 with torch.no_grad(): pred={s:m(xx[s]).cpu().numpy()*ys+ym for s in SPLITS}
 return pred,{'seed':seed,'best_epoch':best_epoch,'history':hist,'parameters':sum(p.numel() for p in m.parameters())}
def evaluate(pred,y,mask):
 return {s:metrics(y[s],pred[s]) for s in SPLITS} | {'test_intervention':metrics(y['test'][mask],pred['test'][mask]),'test_nonintervention':metrics(y['test'][~mask],pred['test'][~mask])}
def sanity():
 torch.manual_seed(123); x=torch.randn(32,32,device=DEVICE); y=torch.randn(32,4,device=DEVICE); m=net(32).to(DEVICE); opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=0); hist=[]
 for i in range(501):
  opt.zero_grad(); loss=nn.functional.mse_loss(m(x),y); loss.backward(); opt.step()
  if i in [0,500]: hist.append(loss.item())
 assert hist[-1]<.01, hist
 return {'initial_mse':hist[0],'final_mse':hist[-1],'examples':32,'steps':501}
def real_batch_sanity(base):
 a=np.load(OUT/'full_b_representations.npz'); xx=a['train_pooled'][:32]; yy=base['train_y'][:32]
 x=torch.tensor((xx-xx.mean(0))/(xx.std(0)+1e-6),dtype=torch.float32,device=DEVICE); y=torch.tensor((yy-yy.mean(0))/(yy.std(0)+1e-6),dtype=torch.float32,device=DEVICE)
 torch.manual_seed(123); m=net(x.shape[1]).to(DEVICE); opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=0); history=[]
 for i in range(1501):
  opt.zero_grad(); loss=nn.functional.mse_loss(m(x),y); loss.backward(); opt.step()
  if i in (0,1500): history.append(loss.item())
 assert history[-1]<.01,history
 return {'model':'full_b','examples':32,'steps':1501,'initial_mse':history[0],'final_mse':history[1]}
def main():
 data=load_data(); base=base_cache(data); y={s:base[s+'_y'] for s in SPLITS}; mask=base['test_intervention']; result={'config':CONFIG,'split_counts':{s:len(y[s]) for s in SPLITS},'subgroup_counts':{'intervention':int(mask.sum()),'nonintervention':int((~mask).sum())},'sanity':sanity(),'models':{},'baselines':{}}
 print('tiny batch',result['sanity'],flush=True)
 for mode in ['single','temporal']:
  x={s:base[s+'_'+mode] for s in SPLITS}; ridge=fit(x['train'],y['train'],alpha=1000); pred={s:predict(x[s],ridge) for s in SPLITS}; result['baselines']['pixel_'+mode]=evaluate(pred,y,mask)
 const={s:np.broadcast_to(y['train'].mean(0),y[s].shape) for s in SPLITS}; result['baselines']['constant']=evaluate(const,y,mask)
 for name,info in MODELS.items():
  print('extract',name,flush=True); a=extract(name,info,data); result['models'][name]={'checkpoint':info[2],'checkpoint_epoch':int(a['checkpoint_epoch']),'checkpoint_sha256':str(a['checkpoint_sha256']),'encoder_sha256':str(a['encoder_sha256']),'representations':{}}
  for mode in ['pooled']+(['flat'] if info[1]!='a' else []):
   x={s:a[s+'_'+mode] for s in SPLITS}; ridge=fit(x['train'],y['train'],alpha=100); lp={s:predict(x[s],ridge) for s in SPLITS}; entry={'linear':evaluate(lp,y,mask),'mlp':[]}
   for seed in SEEDS:
    p,meta=mlp(x,y,seed); meta['metrics']=evaluate(p,y,mask); entry['mlp'].append(meta)
    if seed==SEEDS[0]: np.savez_compressed(OUT/f'{name}_{mode}_predictions.npz',y=y['test'],linear=lp['test'],mlp=p['test'])
   result['models'][name]['representations'][mode]=entry
   print(name,mode,'linear',np.round(entry['linear']['test']['r2'],3),'mlp',np.round(np.mean([z['metrics']['test']['r2'] for z in entry['mlp']],0),3),flush=True)
   (OUT/'metrics.json').write_text(json.dumps(result,indent=2))
 result['sanity']['real_training_batch']=real_batch_sanity(base)
 (OUT/'metrics.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__': main()

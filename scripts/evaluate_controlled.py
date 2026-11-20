"""Evaluate fixed checkpoints and frozen ridge probes; no test-set fitting."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ijepa.data.pinball import CausalPinball
from ijepa.models import JEPA
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]

def get_data(m,split,device):
 ds=CausalPinball(ROOT/f'data/manifests/{split}.jsonl',context=2,prediction=2)
 loader=DataLoader(ds,batch_size=16,shuffle=False,num_workers=0)
 Z=[]; Y=[]; E=[]; P=[]; I=[]
 keys=['ball_x','ball_y','ball_x_vel','ball_y_vel']; idx=None
 with torch.no_grad():
  for b in loader:
   if idx is None: idx=[list(b['state_keys']).index(next(x for x in b['state_keys'] if x[0]==k)) for k in keys]
   v=b['video'].to(device); a=b['interventions'].to(device)
   z=m.encoder(v).mean(2); pred,tgt=m(v,a); cur=m.target(v[:,:-1])
   E.append(((pred-tgt)**2).mean((-1,-2)).cpu().numpy()); P.append(((cur-tgt)**2).mean((-1,-2)).cpu().numpy()); I.append(a.any(-1).cpu().numpy())
   Z.append(z.cpu().numpy()); Y.append(b['states'][:,:,idx].numpy())
 return [np.concatenate(x) for x in (Z,Y,E,P,I)]

def probe(train,test):
 z,y=train[:2]; x,truth=test[:2]; z=z.reshape(-1,z.shape[-1]); y=y.reshape(-1,4); x=x.reshape(-1,x.shape[-1]); truth=truth.reshape(-1,4)
 mu=z.mean(0); sd=z.std(0)+1e-6; Z=(z-mu)/sd; X=(x-mu)/sd; ym=y.mean(0); W=np.linalg.solve(Z.T@Z+100*np.eye(Z.shape[1]),Z.T@(y-ym)); pred=X@W+ym
 rmse=np.sqrt(np.mean((pred-truth)**2,axis=0)); r2=1-np.sum((pred-truth)**2,axis=0)/np.sum((truth-truth.mean(0))**2,axis=0)
 return {'rmse':rmse.tolist(),'r2':r2.tolist()},pred.reshape(-1,4,4),truth.reshape(-1,4,4)

def metrics(data):
 z,y,e,p,i=data; z=z.reshape(-1,z.shape[-1]); cov=np.cov(z,rowvar=False); eig=np.clip(np.linalg.eigvalsh(cov),0,None); probs=eig/(eig.sum()+1e-30); nz=probs[probs>0]; rank=float(np.exp(-(nz*np.log(nz)).sum())); std=float(z.std(0).mean())
 rng=np.random.default_rng(73); values=e.mean(1); boot=np.array([rng.choice(values,size=len(values),replace=True).mean() for _ in range(1000)])
 def masked(a,mask): return float(a[mask].mean()) if mask.any() else None
 return {'mean_embedding_std':std,'effective_covariance_rank':rank,'collapse_flag_std_below_0_01':bool(std<.01),'low_rank_flag_below_2':bool(rank<2),'potential_dimensional_collapse':bool(rank<z.shape[1]/4),'prediction_mse':float(e.mean()),'prediction_mse_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),'persistence_mse':float(p.mean()),'intervention_mse':masked(e,i),'nonintervention_mse':masked(e,~i),'intervention_count':int(i.sum()),'nonintervention_count':int((~i).sum())}

def main(args):
 torch.set_num_threads(2); device='mps' if torch.backends.mps.is_available() else 'cpu'; out=ROOT/'outputs/experiments'/args.group; out.mkdir(parents=True,exist_ok=True); results={}; histories={}; samples={}
 for name in args.names:
  ck=ROOT/'outputs/experiments'/name/'best.pt'; ckpt=torch.load(ck,map_location='cpu',weights_only=False); cfg=ckpt['config']; m=JEPA(cfg['architecture_variant'],32).to(device); m.load_state_dict(ckpt['model']); m.eval(); print('evaluating',name,flush=True)
  train=get_data(m,'train',device); test=get_data(m,args.split,device); result=metrics(test); result['probe'],pred,truth=probe(train,test); result.update({'checkpoint_epoch':ckpt['epoch'],'seed':cfg['seed'],'training_examples':cfg['train_examples']}); results[name]=result; samples[name]=(pred[0],truth[0]); histories[name]=json.loads((ck.parent/'history.json').read_text())
 (out/'metrics.json').write_text(json.dumps(results,indent=2));
 with (out/'metrics.csv').open('w',newline='') as f:
  cols=['model','prediction_mse','persistence_mse','intervention_mse','nonintervention_mse','mean_embedding_std','effective_covariance_rank']; w=csv.DictWriter(f,fieldnames=cols,lineterminator='\n');w.writeheader();w.writerows([dict(model=k,**{c:v[c] for c in cols[1:]}) for k,v in results.items()])
 figdir=ROOT/'outputs/figures'; figdir.mkdir(parents=True,exist_ok=True); plt.style.use('seaborn-v0_8-whitegrid'); names=list(results); labels=[n.replace('full_','') for n in names]; colors=['#435c82','#cb7049','#77a68f','#9d76a8']
 fig,ax=plt.subplots(1,2,figsize=(11,4));
 for n,c in zip(names,colors):
  h=histories[n]; ax[0].plot([r['epoch'] for r in h],[r['train_total'] for r in h],label=n,color=c); ax[1].plot([r['epoch'] for r in h],[r['val_mse'] for r in h],label=n,color=c)
 ax[0].set(title='Training objective (MSE + variance and covariance)',xlabel='Epoch',ylabel='Loss'); ax[1].set(title='Validation prediction MSE',xlabel='Epoch',ylabel='MSE'); ax[1].legend(); fig.tight_layout();fig.savefig(figdir/'training_curves.png',dpi=180);plt.close(fig)
 def bars(filename,title,series,ylabel):
  fig,ax=plt.subplots(figsize=(8,4.5)); x=np.arange(len(names)); width=.75/len(series)
  for j,(label,values) in enumerate(series): ax.bar(x+(j-(len(series)-1)/2)*width,values,width,label=label)
  ax.set_xticks(x,labels); ax.set(title=title,ylabel=ylabel); ax.legend(); fig.tight_layout();fig.savefig(figdir/filename,dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4.5)); x=np.arange(len(names)); means=np.array([results[n]['prediction_mse'] for n in names]); ci=np.array([results[n]['prediction_mse_bootstrap_95ci'] for n in names]); ax.bar(x-.18,means,.36,label='JEPA'); ax.errorbar(x-.18,means,yerr=np.maximum(0,np.stack((means-ci[:,0],ci[:,1]-means))),fmt='none',ecolor='black',capsize=3); ax.bar(x+.18,[results[n]['persistence_mse'] for n in names],.36,label='Persistence'); ax.set_xticks(x,labels); ax.set(title='One-step latent prediction (model-specific scales)',ylabel='MSE; 95% trajectory bootstrap interval'); ax.legend(); fig.tight_layout(); fig.savefig(figdir/'model_comparison.png',dpi=180); plt.close(fig)
 bars('collapse_diagnostics.png','Representation diagnostics', [('Mean dimension std',[results[n]['mean_embedding_std'] for n in names]),('Effective rank / 32',[results[n]['effective_covariance_rank']/32 for n in names])],'Statistic')
 bars('probe_results.png','Frozen linear probe: ball state', [(k,[results[n]['probe']['r2'][i] for n in names]) for i,k in enumerate(['x','y','x velocity','y velocity'])],'Held-out R²')
 bars('intervention_errors.png','One-step prediction by intervention label', [('Intervention',[results[n]['intervention_mse'] for n in names]),('No intervention',[results[n]['nonintervention_mse'] for n in names])],'Mean squared error')
 fig,axes=plt.subplots(2,4,figsize=(12,6)); ds=CausalPinball(ROOT/f'data/manifests/{args.split}.jsonl',context=2,prediction=2); sample=ds[0]
 for i in range(4): axes[0,i].imshow(sample['video'][i].permute(1,2,0));axes[0,i].set_title(f'Observed frame {i}');axes[0,i].axis('off')
 for i,(n,c) in enumerate(zip(names,colors)):
  pred,truth=samples[n]; axes[1,i].plot(range(4),truth[:,0],label='True ball x',color='black');axes[1,i].plot(range(4),pred[:,0],label='Frozen probe',color=c);axes[1,i].set_title(n);axes[1,i].set_xlabel('Observed frame');axes[1,i].legend(fontsize=7)
 fig.suptitle('Observed frames and decoded ball position (not generated futures)');fig.tight_layout();fig.savefig(figdir/'probe_example.png',dpi=180);plt.close(fig)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',required=True);p.add_argument('--group',default='controlled');p.add_argument('--split',choices=['val','test'],default='test');main(p.parse_args())

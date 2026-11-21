"""Plot completed experiment 3 runs, retaining per-seed points."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]/'outputs/experiment3'; data=json.loads((root/'temporal_metrics.json').read_text()); figs=root/'figures';figs.mkdir(exist_ok=True)
colors=['#576b98','#de854f','#488b73']; kinds=['t0','t1','t2']
def vals(kind,fn): return [fn(v) for k,v in data.items() if k.startswith(kind+'_')]
fig,axes=plt.subplots(1,2,figsize=(9,4))
for j,(title,ids) in enumerate([('Position R²',[0,1]),('Velocity R²',[2,3])]):
 for i,k in enumerate(kinds):
  v=vals(k,lambda x:np.mean(np.array(x['probe']['r2'])[ids]));axes[j].scatter([i]*len(v),v,color=colors[i]);axes[j].plot([i-.15,i+.15],[np.mean(v)]*2,color='black')
 axes[j].set_xticks(range(3),kinds);axes[j].set_title(title);axes[j].axhline(0,color='gray',lw=.5)
fig.tight_layout();fig.savefig(figs/'state_probes.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9,4))
for j,(name,label) in enumerate([('effective_rank','Effective rank'),('std','Mean embedding std')]):
 for i,k in enumerate(kinds): axes[j].scatter([i]*len(vals(k,lambda x:x[name])),vals(k,lambda x:x[name]),color=colors[i])
 axes[j].set_xticks(range(3),kinds);axes[j].set_title(label)
fig.tight_layout();fig.savefig(figs/'collapse.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4))
for i,k in enumerate(kinds):
 for j,condition in enumerate(['shuffle','reverse','repeat']):
  v=vals(k,lambda x:x['order'][condition]['representation_rms_change']/(x['std']+1e-12));axes[0].scatter([j+(i-1)*.18]*len(v),v,color=colors[i],label=k if j==0 else None)
 for j,condition in enumerate(['all','intervention','nonintervention']):
  v=vals(k,lambda x:x['subsets'][condition]['prediction_to_persistence_ratio']);axes[1].scatter([j+(i-1)*.18]*len(v),v,color=colors[i])
axes[0].set_xticks(range(3),['Shuffle','Reverse','Repeat']);axes[0].set_title('RMS change / mean embedding std');axes[0].legend();axes[1].set_xticks(range(3),['All','Intervention','No intervention']);axes[1].set_title('Prediction / own persistence');axes[1].axhline(1,color='gray',lw=.7);fig.tight_layout();fig.savefig(figs/'temporal_and_prediction.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,4))
for i,k in enumerate(kinds):
 for j,condition in enumerate(['correct','shuffle','reverse','repeat']):
  for ax,idx in zip(axes,([0,1],[2,3])):
   def score(v):
    value=v['probe']['r2'] if condition=='correct' else v['order'][condition]['probe_against_original_state']['r2']
    return float(np.mean(np.array(value)[idx]))
   v=vals(k,score);ax.scatter([j+(i-1)*.18]*len(v),v,color=colors[i],label=k if j==0 else None)
for ax,title in zip(axes,['Position probe R²','Velocity probe R²']):
 ax.set_xticks(range(4),['Correct','Shuffle','Reverse','Repeat']);ax.set_title(title);ax.axhline(0,color='gray',lw=.5)
axes[0].legend();fig.tight_layout();fig.savefig(figs/'order_probe_sensitivity.png',dpi=160);plt.close(fig)

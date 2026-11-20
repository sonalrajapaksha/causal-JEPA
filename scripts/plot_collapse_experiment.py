"""Plot experiment 2 results without altering experiment 1 figures."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]/'outputs/experiment2'
r=json.loads((root/'metrics.json').read_text()); root.joinpath('figures').mkdir(exist_ok=True)
fig,axs=plt.subplots(1,3,figsize=(13,4)); variants=['b1','b2','b3']; labels=['B1','B2','B3 supervised']
for i,v in enumerate(variants):
 rows=[x for name,x in r.items() if name.startswith(v+'_')]
 for row in rows:
  axs[0].scatter(i,row['effective_rank'],color='C'+str(i)); axs[1].scatter(i,np.mean(row['probe']['r2'][:2]),color='C'+str(i)); axs[2].scatter(i,np.mean(row['probe']['r2'][2:]),color='C'+str(i))
for ax,title in zip(axs,['Effective covariance rank','Position probe mean R²','Velocity probe mean R²']):
 ax.set_xticks(range(3),labels,rotation=15);ax.set_title(title);ax.grid(alpha=.2,axis='y')
fig.tight_layout();fig.savefig(root/'figures/rank_and_probes.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4))
for i,v in enumerate(variants):
 for row in [x for name,x in r.items() if name.startswith(v+'_')]:
  for j,s in enumerate(['intervention','nonintervention']): ax.scatter(i+(j-.5)*.22,row['subsets'][s]['ratio'],marker=['o','s'][j],color='C'+str(j))
ax.axhline(1,color='black',linestyle='--');ax.set_xticks(range(3),labels);ax.set_ylabel('Prediction MSE / own persistence MSE');ax.scatter([],[],label='Intervention',color='C0');ax.scatter([],[],label='No intervention',color='C1',marker='s');ax.legend();fig.tight_layout();fig.savefig(root/'figures/persistence_ratios.png',dpi=180)

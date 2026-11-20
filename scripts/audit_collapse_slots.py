"""Diagnostic: test whether same-index latent slots are distinct and temporally matched.
Uses validation sequences only; assignments are latent-space diagnostics, not object labels.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from collapse_experiment import OUT, loader
from ijepa.models import JEPA

torch.set_num_threads(2); device='mps' if torch.backends.mps.is_available() else 'cpu'; results={}
for variant in ['b1','b2','b3']:
 path=OUT/f'{variant}_seed42'/'final.pt'
 if not path.exists(): continue
 m=JEPA('b',32).to(device); m.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']); m.eval(); identity=[]; optimal=[]; match=[]; spread=[]; pred_identity=[]; pred_optimal=[]; pred_match=[]
 with torch.no_grad():
  for b in loader('val'):
   v=b['video'].to(device); z=m.target(v).cpu().numpy(); pred,tgt=m(v,b['interventions'].to(device)); pred=pred.cpu().numpy(); tgt=tgt.cpu().numpy()
   for pp,tt in zip(pred.reshape(-1,8,32),tgt.reshape(-1,8,32)):
    costs=((pp[:,None,:]-tt[None,:,:])**2).mean(-1); rows,cols=linear_sum_assignment(costs); pred_identity.append(np.diag(costs).mean()); pred_optimal.append(costs[rows,cols].mean()); pred_match.append(np.mean(rows==cols))
   for sample in z:
    for cur,nxt in zip(sample[:-1],sample[1:]):
     costs=((cur[:,None,:]-nxt[None,:,:])**2).mean(-1); rows,cols=linear_sum_assignment(costs)
     identity.append(np.diag(costs).mean()); optimal.append(costs[rows,cols].mean());match.append(np.mean(cols==rows));spread.append(((cur[:,None,:]-cur[None,:,:])**2).mean())
 results[variant]={'same_index_temporal_mse':float(np.mean(identity)),'optimal_assignment_temporal_mse':float(np.mean(optimal)),'optimal_same_index_fraction':float(np.mean(match)),'between_slot_mse':float(np.mean(spread)),'prediction_same_index_mse':float(np.mean(pred_identity)),'prediction_optimal_assignment_mse':float(np.mean(pred_optimal)),'prediction_optimal_same_index_fraction':float(np.mean(pred_match)),'transitions':len(identity)}
(OUT/'slot_audit_validation.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))

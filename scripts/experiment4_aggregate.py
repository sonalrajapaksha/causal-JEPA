"""Add derived seed summaries to the raw completed Experiment 4 metrics."""
import json
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parents[1]/'outputs/experiment4/metrics.json'; r=json.loads(p.read_text())
for model in r['models'].values():
 for e in model['representations'].values():
  a={s:np.array([z['metrics'][s]['r2'] for z in e['mlp']]) for s in ['train','val','test']}; lin=np.array(e['linear']['test']['r2'])
  e['aggregate']={s:{'r2_mean':v.mean(0).tolist(),'r2_std':v.std(0).tolist()} for s,v in a.items()}
  for key,v in [('test_nonlinear_minus_linear',a['test']-lin),('generalization_gap_train_minus_test',a['train']-a['test'])]: e['aggregate'][key]={'mean':v.mean(0).tolist(),'std':v.std(0).tolist()}
p.write_text(json.dumps(r,indent=2))

"""Matched four-frame pixel ridge reference; updates only Experiment 4 outputs."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import avg_pool2d
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ijepa.data.pinball import CausalPinball
from collapse_experiment import fit,predict
from experiment4 import OUT,ROOT,SPLITS,evaluate
p=OUT/'metrics.json'; result=json.loads(p.read_text()); data=np.load(OUT/'data.npz'); x={}; y={}
for s in SPLITS:
 ds=CausalPinball(ROOT/f'data/manifests/{s}.jsonl',context=4,prediction=1); chunks=[];ids=[]
 for b in DataLoader(ds,batch_size=32):
  v=b['video']; z=avg_pool2d(v[:,:4].reshape(-1,3,64,64),8).reshape(len(v),4,-1).numpy(); chunks.append(np.concatenate([z[:,3],z[:,3]-z[:,2],z[:,2]-z[:,1],z[:,1]-z[:,0]],axis=1));ids.extend(b['id'])
 assert ids==data[s+'_ids'].tolist();x[s]=np.concatenate(chunks);y[s]=data[s+'_y']
model=fit(x['train'],y['train'],alpha=1000); pred={s:predict(x[s],model) for s in SPLITS}; result['baselines']['pixel_four']=evaluate(pred,y,data['test_intervention']);result['config']['pixel_four_features']='8x8 RGB frame 3 and three consecutive differences (3-2, 2-1, 1-0)';p.write_text(json.dumps(result,indent=2)); print('four-frame pixel R²:',result['baselines']['pixel_four']['test']['r2'])

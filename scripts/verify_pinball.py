import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ijepa.data.pinball import CausalPinball, make_loader
from ijepa.models import JEPA

def main():
    sets={}; seeds=set(); intervention_count=0; total_bytes=0
    for split in ('train','val','test'):
        ds=CausalPinball(f'data/manifests/{split}.jsonl'); sets[split]=ds
        for rec in ds.records:
            path=ds.root/rec['path']; total_bytes+=path.stat().st_size
            with np.load(path,allow_pickle=False) as a:
                assert a['frames'].shape==(16,64,64,3) and a['frames'].dtype==np.uint8
                assert a['states'].shape[0]==16 and np.isfinite(a['states']).all()
                assert a['interventions'].shape==(15,5) and np.isin(a['interventions'],[0,1]).all()
                seed=int(a['seed']); assert seed not in seeds; seeds.add(seed)
                intervention_count+=int(a['interventions'].sum())
        print(split,len(ds))
    ids=[{x['simulation_id'] for x in ds.records} for ds in sets.values()]
    assert all(not ids[i]&ids[j] for i in range(3) for j in range(i))
    print('bytes',total_bytes,'interventions',intervention_count)
    batch=next(iter(make_loader('data/manifests/train.jsonl',batch_size=2)))
    assert batch['video'].shape==(2,16,3,64,64)
    torch.set_num_threads(2)
    for name in 'abc':
        torch.manual_seed(1); model=JEPA(name,dim=32); opt=torch.optim.Adam(model.parameters(),lr=.002)
        # Fixed target for an informative tiny-sample optimization check.
        losses=[]
        for _ in range(25):
            opt.zero_grad(); loss=model.loss(batch['video'][:,:4],batch['interventions'][:,:3]); losses.append(loss.item()); loss.backward(); opt.step()
        assert losses[-1] < losses[0], (name, losses[0], losses[-1])
        model.update_target(); ckpt=Path('outputs/checkpoints')/f'smoke_{name}.pt'; ckpt.parent.mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),ckpt)
        JEPA(name,dim=32).load_state_dict(torch.load(ckpt,weights_only=True))
        print(name,'overfit',losses[0],losses[-1],'checkpoint OK')
    fig, axes=plt.subplots(3,5,figsize=(13,8))
    for row in range(3):
        sample=sets['train'][row]
        for col,t in enumerate((0,3,6,9,12)):
            axes[row,col].imshow(sample['video'][t].permute(1,2,0)); axes[row,col].axis('off')
            names=sample['intervention_keys']; action=sample['interventions'][t-1] if t else torch.zeros(5)
            label=','.join(names[i] for i,v in enumerate(action) if v) or 'none'
            states=sample['states'][t]; keys=sample['state_keys']; x=states[keys.index('ball_x')]; y=states[keys.index('ball_y')]
            axes[row,col].set_title(f't={t} ball=({x:.1f},{y:.1f})\nincoming: {label}',fontsize=8)
    fig.tight_layout(); fig.savefig('outputs/figures/dataset_preview.png',dpi=130); plt.close(fig)
if __name__=='__main__': main()

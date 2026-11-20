import argparse, random, sys
from pathlib import Path
import numpy as np
import yaml
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ijepa.data.pinball import make_loader
from ijepa.models import JEPA

def main():
    pre=argparse.ArgumentParser(add_help=False); pre.add_argument('--config'); known,_=pre.parse_known_args()
    config=yaml.safe_load(Path(known.config).read_text()) if known.config else {}
    p=argparse.ArgumentParser(parents=[pre]); p.add_argument('--model', choices=['a','b','c'], default=config.get('model')); p.add_argument('--epochs', type=int, default=10); p.add_argument('--batch-size', type=int, default=8); p.add_argument('--lr', type=float, default=0.0003); p.add_argument('--seed', type=int, default=42); p.add_argument('--device', default='auto'); p.add_argument('--output', default=None); p.add_argument('--context', type=int, default=8); p.add_argument('--prediction', type=int, default=8)
    p.set_defaults(**{k: v for k,v in config.get('training',{}).items() if k in ('epochs','batch_size','seed')})
    if 'learning_rate' in config.get('training',{}): p.set_defaults(lr=config['training']['learning_rate'])
    p.set_defaults(**{k: v for k,v in config.get('data',{}).items() if k in ('context','prediction')})
    a=p.parse_args()
    if not a.model: p.error('--model or --config is required')
    out=Path(a.output or f'outputs/checkpoints/model_{a.model}.pt')
    if out.exists(): raise FileExistsError(f'{out} already exists; choose a new --output')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = ('mps' if torch.backends.mps.is_available() else 'cpu') if a.device == 'auto' else a.device
    model=JEPA(a.model).to(device); opt=torch.optim.AdamW(model.parameters(),lr=a.lr)
    train=make_loader(config.get('data',{}).get('train','data/manifests/train.jsonl'),batch_size=a.batch_size,shuffle=True, context=a.context, prediction=a.prediction)
    val=make_loader(config.get('data',{}).get('val','data/manifests/val.jsonl'),batch_size=a.batch_size, context=a.context, prediction=a.prediction)
    for epoch in range(a.epochs):
        model.train(); total=0
        for batch in train:
            opt.zero_grad(); loss=model.loss(batch['video'].to(device), batch['interventions'].to(device)); loss.backward(); opt.step(); model.update_target(); total+=loss.item()*len(batch['video'])
        model.eval(); v=0
        with torch.no_grad():
            for batch in val: v+=model.loss(batch['video'].to(device),batch['interventions'].to(device)).item()*len(batch['video'])
        print(f'epoch {epoch+1} train={total/len(train.dataset):.6f} val={v/len(val.dataset):.6f}',flush=True)
        out.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'args':vars(a)},out)
if __name__=='__main__': main()

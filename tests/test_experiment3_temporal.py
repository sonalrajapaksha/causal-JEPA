import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from experiment3_temporal import Model

def test_temporal_models_gradient_ema_and_context():
 torch.set_num_threads(2); torch.manual_seed(10);x=torch.randn(2,5,3,64,64)
 for kind in ('t0','t1','t2'):
  m=Model(kind);z,p,t,c=m(x);assert z.shape==p.shape==t.shape==c.shape==(2,8,32);assert not t.requires_grad
  (p-t).square().mean().backward(); assert any(v.grad is not None and v.grad.abs().sum()>0 for v in m.encoder.parameters());assert all(v.grad is None for v in m.target.parameters())
  before=next(m.target.parameters()).detach().clone()
  with torch.no_grad(): next(m.encoder.parameters()).add_(1)
  m.update();assert not torch.equal(before,next(m.target.parameters()))
  if kind!='t0':
   with torch.no_grad():
    assert not torch.allclose(m.encoder(x[:,:4]),m.encoder(x[:,[2,0,1,3]]))
    changed=x[:,:4].clone();changed[:,0]=torch.randn_like(changed[:,0]);assert not torch.allclose(m.encoder(x[:,:4]),m.encoder(changed))

def test_fixed_target_tiny_batch_loss_decreases():
 torch.set_num_threads(2);torch.manual_seed(3);x=torch.randn(2,5,3,64,64)
 for kind in ('t0','t1','t2'):
  m=Model(kind);opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.001)
  with torch.no_grad(): target=m.target(x[:,1:5]).detach()
  values=[]
  for _ in range(40):
   opt.zero_grad();p=m.predictor(m.relation(m.encoder(x[:,:4])));loss=(p-target).square().mean();values.append(loss.item());loss.backward();opt.step()
  assert values[-1]<values[0]*.3,(kind,values[0],values[-1])

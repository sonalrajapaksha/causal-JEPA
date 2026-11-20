import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from collapse_experiment import fit, predict, penalties, loader, labels

def test_labels_match_archived_state_keys():
 b=next(iter(loader('train',batch=2)))
 keys=[x[0] for x in b['state_keys']]
 for j,k in enumerate(['ball_x','ball_y','ball_x_vel','ball_y_vel']):
  torch.testing.assert_close(labels(b)[:,:,j],b['states'][:,:,keys.index(k)])

def test_probe_uses_fitted_training_statistics():
 rng=np.random.default_rng(1); x=rng.normal(size=(200,4)); y=2*x+3
 pred=predict(x,fit(x,y,alpha=.001))
 np.testing.assert_allclose(pred,y,atol=1e-3)

def test_penalties_have_finite_gradients_at_collapse():
 for variant in ('b1','b2'):
  z=torch.zeros(8,3,8,32,requires_grad=True); loss=penalties(z,variant);loss.backward()
  assert torch.isfinite(loss) and torch.isfinite(z.grad).all()

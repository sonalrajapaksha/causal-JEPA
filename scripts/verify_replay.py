"""Replay simulator transitions, including the RNG, against saved labels and states."""
import sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'external/CITRIS'), str(root/'external/CITRIS/data_generation')]
import data_generation_pinball as pinball
for split in ('train','val','test'):
    for path in sorted((root/'data/processed/causal_pinball_64'/split).glob('*.npz'))[:3]:
        with np.load(path) as a:
            settings=pinball.create_settings(42); np.random.seed(int(a['seed']))
            state=pinball.sample_random_point(settings); keys=a['state_keys'].tolist(); targets=a['intervention_keys'].tolist()
            for t in range(16):
                np.testing.assert_allclose([state[k] for k in keys],a['states'][t],rtol=1e-6,atol=1e-6)
                if t<15:
                    state, action=pinball.next_step(state,settings)
                    np.testing.assert_array_equal([action[k] for k in targets],a['interventions'][t])
        print(path.name,split,'replay OK')
# Saved simulator state, settings, and RNG state are sufficient to replay a transition.
from copy import deepcopy
settings=pinball.create_settings(42); np.random.seed(1234)
state=pinball.sample_random_point(settings)
rng=np.random.get_state(); saved_state=deepcopy(state); saved_settings=deepcopy(settings)
a, ia=pinball.next_step(state,settings)
np.random.set_state(rng)
b, ib=pinball.next_step(saved_state,saved_settings)
assert ia==ib and a==b
print('saved state and RNG replay OK')

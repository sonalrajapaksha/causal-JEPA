"""Streaming access to independently simulated Causal Pinball trajectories."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CausalPinball(Dataset):
    def __init__(self, manifest, context=8, prediction=8):
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent.parent.parent
        self.records = [json.loads(x) for x in self.manifest.read_text().splitlines() if x.strip()]
        if context < 1 or prediction < 1 or context + prediction > 16:
            raise ValueError('context and prediction must be positive and sum to at most 16')
        self.context, self.prediction = context, prediction
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        rec = self.records[index]
        with np.load(self.root / rec['path'], allow_pickle=False) as data:
            frames = torch.from_numpy(data['frames'].copy()).permute(0, 3, 1, 2).float().div_(255)
            states = torch.from_numpy(data['states'].copy())
            actions = torch.from_numpy(data['interventions'].copy()).float()
            state_keys = data['state_keys'].tolist()
            action_keys = data['intervention_keys'].tolist()
        n = self.context + self.prediction
        return {'video': frames[:n], 'context': frames[:self.context], 'future': frames[self.context:n],
                'interventions': actions[:n-1], 'states': states[:n], 'id': rec['id'],
                'state_keys': state_keys, 'intervention_keys': action_keys}

def make_loader(manifest, batch_size=8, shuffle=False, num_workers=0, **kwargs):
    return DataLoader(CausalPinball(manifest, **kwargs), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

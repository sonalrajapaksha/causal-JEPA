"""Generate independent CITRIS pinball trajectories using upstream dynamics and renderer."""
import argparse, json, os, sys, tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'external/CITRIS'))
sys.path.insert(0, str(ROOT / 'external/CITRIS/data_generation'))
os.environ.setdefault('MPLBACKEND', 'Agg')
import data_generation_pinball as pinball


def make_one(args):
    split, index, seed = args
    dest = ROOT / 'data/processed/causal_pinball_64' / split / f'{index:05d}.npz'
    if dest.exists():
        return
    settings = pinball.create_settings(42)
    np.random.seed(seed)
    state = pinball.sample_random_point(settings)
    keys = sorted(state)
    target_keys = settings['keys_targets']
    states, targets, frames = [], [], []
    with tempfile.TemporaryDirectory() as tmp:
        image_path = str(Path(tmp) / 'frame.png')
        for t in range(16):
            states.append([state[k] for k in keys])
            pinball.plot_matplotlib_figure(state, settings, image_path, ignore_projection=True)
            with Image.open(image_path) as img:
                frames.append(np.asarray(img.convert('RGB')).copy())
            if t < 15:
                state, intervention = pinball.next_step(state, settings)
                targets.append([intervention[k] for k in target_keys])
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix('.tmp.npz')
    np.savez_compressed(tmp_path, frames=np.stack(frames), states=np.array(states, dtype=np.float32),
                        interventions=np.array(targets, dtype=np.uint8), state_keys=np.array(keys),
                        intervention_keys=np.array(target_keys), seed=np.int64(seed))
    os.replace(tmp_path, dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--train', type=int, default=2400)
    parser.add_argument('--val', type=int, default=300)
    parser.add_argument('--test', type=int, default=300)
    args = parser.parse_args()
    tasks = [(s, i, base+i) for s, n, base in [('train', args.train, 100000), ('val', args.val, 200000), ('test', args.test, 300000)] for i in range(n)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(make_one, tasks, chunksize=4), 1):
            if i % 50 == 0 or i == len(tasks): print(f'{i}/{len(tasks)}', flush=True)
    manifest_dir = ROOT / 'data/manifests'
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for s, n in [('train', args.train), ('val', args.val), ('test', args.test)]:
        path = manifest_dir / f'{s}.jsonl'
        with path.open('w') as out:
            for i in range(n):
                record = {'id': f'{s}-{i:05d}', 'path': f'data/processed/causal_pinball_64/{s}/{i:05d}.npz', 'simulation_id': f'{s}-{i:05d}'}
                out.write(json.dumps(record)+'\n')

if __name__ == '__main__': main()

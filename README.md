# Interventional JEPA: Causal Pinball

This project uses [CITRIS/iCITRIS Causal Pinball](https://github.com/phlippe/CITRIS) by Lippe et al. The simulator and Matplotlib RGB renderer are upstream code, revision `95f6c90b9ff769ef0250d3a5434b9352853f4302`. Cite the CITRIS and iCITRIS papers described in that repository. Upstream is cloned at `external/CITRIS`; it is not modified.

## Setup and generation

Python 3.11 and `uv` were used on Apple Silicon; `requirements-pinball.txt` records the installed versions. From the repository root:

```sh
git submodule update --init --recursive
uv venv .venv
uv pip install --python .venv/bin/python -r requirements-pinball.txt
.venv/bin/python scripts/prepare_pinball.py --workers 3
.venv/bin/python scripts/verify_replay.py
.venv/bin/python scripts/verify_pinball.py
```

The submodule pins the upstream checkout to the revision above. The generator invokes upstream Python functions rather than its full CLI, avoiding independent and triplet datasets. Its default CLI accepts `--output_folder`, `--dataset_size`, `--num_processes`, and `--seed`; it creates a single long sequence per split and auxiliary datasets. The wrapper initializes each 16-frame trajectory separately, with fixed settings seed 42 and distinct trajectory seeds 100000+, 200000+, and 300000+ for train, validation, and test. It calls the official renderer with its velocity projection disabled: only the first three RGB channels are used. Existing per-sequence files are skipped; generate in a clean directory for a fresh reproduction. The wrapper writes manifests after all requested trajectories finish.

`data/processed/causal_pinball_64/{train,val,test}/*.npz` contains compressed uint8 `frames` (16,64,64,3), float32 `states`, binary `interventions` (15,5), their key arrays, and the simulation seed. `data/manifests/*.jsonl` indexes samples and unique simulation IDs. The raw directory includes the small official CLI smoke output. Data are streamed one sequence at a time; tensors are float32 in [0,1], shaped `[B,T,C,H,W]`. The loader supports context and prediction lengths, with deterministic slicing. States are returned only for evaluation and are not passed to the models.

An intervention row at index t applies to the transition from frame t to t+1. Its five indicators denote upstream `ball`, `cyl`, `paddle_left`, `paddle_right`, and `score` interventions. These indicate which stochastic mechanisms were replaced; they are not signed magnitudes. Simultaneous interventions can occur. State changes can also occur without intervention due to physics. No factual/counterfactual pairs are supplied: upstream triplets mix latent variables from separate samples and are not matched rollouts. A saved initial state and RNG state replay identically, but intervention branches consume random draws differently; controlled common-noise counterfactuals have not been established.

## Models and training

All variants use the same sequences, adjacent-frame loss, optimizer settings, and train/validation splits. Model A predicts a global visual representation. B uses eight learned attention slots over image features and a relational transformer. C has the same visual machinery as B and adds an embedding of the actual transition intervention labels to its predictions. Its objective tests whether labelled interventions improve conditional next-frame prediction; it does not measure a paired causal effect. Slots are learned from pixels, not simulator states or masks; query indices are consistent model parameters, but object assignment and identity tracking are not guaranteed. The momentum target encoder is updated after each optimizer step. These are lightweight research baselines; representation collapse and the causal usefulness of the slots require evaluation.

The default window is 8 context plus 8 prediction frames. Training currently predicts every adjacent frame in that window, including transitions within the context; it is not an eight-step rollout. Configurations in `configs/` select matched defaults. Run from the repository root:

```sh
.venv/bin/python scripts/train.py --config configs/a_video_jepa.yaml --output outputs/checkpoints/model_a_run1.pt
.venv/bin/python scripts/train.py --config configs/b_object_jepa.yaml --output outputs/checkpoints/model_b_run1.pt
.venv/bin/python scripts/train.py --config configs/c_interventional_jepa.yaml --output outputs/checkpoints/model_c_run1.pt
```

The script selects MPS when available, otherwise CPU. Specify `--device cpu` if needed; use `--epochs`, `--batch-size`, `--context`, or `--prediction` to adjust the run. Checkpoints are not overwritten by a new run. Validation is used for monitoring; the test split is never loaded by training. Dataset checks, small-batch gradient/overfit checks, and checkpoint round trips are in `scripts/verify_pinball.py`; `outputs/figures/dataset_preview.png` shows examples with ball states and incoming interventions.

The generated dataset has 2,400 train, 300 validation, and 300 test sequences (48,000 frames). The sequence archives total 17,807,170 bytes; their allocated disk usage is approximately 23 MB, plus approximately 44 KB for the upstream smoke output and 392 KB for manifests. `verify_pinball.py` checks every archive and uses a two-sequence CPU batch; on the generation machine its peak resident set was about 373 MB. MPS forward and backward smoke checks passed for all three models. The fixed-target tiny-batch optimization test checks that loss falls, not that the models learn useful or non-collapsed representations.

## Controlled experiment audit

The original `scripts/train.py` objective has no variance or covariance safeguard and can collapse. For the controlled pilots use `scripts/controlled_experiment.py`; it records seed, settings, validation history, MPS driver allocation, and validation-selected checkpoints under `outputs/experiments/`. It adds variance and covariance penalties to the latent prediction loss. The `b_info` ablation and `c` are intentionally identical architectures and objectives: both condition on intervention labels. They cannot establish an added causal objective. `scripts/audit_experiment.py` checks split identity, shapes, target stop-gradient, and EMA updates. `scripts/evaluate_controlled.py` computes frozen ridge probes and diagnostics; its `--split` option distinguishes validation from held-out test results.

## Representation-collapse follow-up

`scripts/collapse_experiment.py` compares the original controlled Model B objective (B1), normalized variance/covariance regularization (B2), and a simulator-state supervised reference (B3). It uses the existing splits and writes only to `outputs/experiment2/`. The B3 head and state-transition/position-oracle baselines use labels and are not self-supervised. The detailed methods and results are in `outputs/experiment2/methods.md` and `outputs/research_summary.md`. To reproduce in a fresh output directory:

```sh
for variant in b1 b2 b3; do
  for seed in 42 43 44; do
    .venv/bin/python scripts/collapse_experiment.py train --variant "$variant" --seed "$seed"
  done
done
.venv/bin/python scripts/collapse_experiment.py baseline
.venv/bin/python scripts/collapse_experiment.py evaluate
.venv/bin/python scripts/audit_collapse_slots.py
.venv/bin/python scripts/plot_collapse_experiment.py
```

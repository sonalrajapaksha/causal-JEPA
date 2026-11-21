# Experiment 3: temporal representation learning

## Supervised audit

The B3 checkpoint seed 42 has prediction MSE 0.0088, weighted regularization 0.3114, and unit-weight standardized state MSE 1.1250 on the sampled training batch. Corresponding online encoder gradient norms are 0.0227, 0.3970, and 0.3423. These are separate term gradients, not evidence of their combined direction. No target gradients were present. The sampled shapes were video [32,5,3,64,64], state [32,5,4], online slots [32,3,8,32], and head output [32,3,4]. The head is linear, 32 to 4, applied to the mean of eight slots; the target order is ball_x, ball_y, ball_x_vel, ball_y_vel. The training-only mean and standard deviation are stored in the audit JSON. The standardize/inverse round-trip maximum error is 9.5e-07.

States are saved in sorted key order and explicitly indexed by name. There is one ball with a persistent simulator key, not a list of ball instances. Slot indices are learned queries; nothing assigns them to objects. Previous validation slot audit found B3 slots nearly identical (between-slot MSE 7.46e-5). On validation, mean and concatenated-slot probes gave the following R²:

- Mean: -0.008, 0.001, -0.003, 0.006
- Concatenated: -0.007, 0.003, -0.003, 0.006

Concatenation does not recover a meaningful signal here; this cannot prove pooling never discards object information. B3 is not a pure supervised baseline: competing regularization and prediction terms, a linear head and almost identical slots limit interpretation. No target-order or normalization bug was found.

A 32-example fixed-batch position CNN reduced normalized MSE from 1.1076 to 0.0022. A first full-data CNN run trained from scratch converged near constant predictions; its log is retained. The final supervised CNNs use a train-fitted pooled-pixel ridge skip, zero-initialized CNN residual, and Adam at 1e-4. The two-frame CNN concatenates consecutive RGB frames as channels. These are hybrid references, not independently successful CNNs trained from scratch. Eight epochs and validation-selected checkpoints were used. Test R² (x, y, vx, vy):

- One frame: 0.934, 0.947, -0.040, -0.050.
- Two frames: 0.942, 0.956, 0.102, -0.020.
- Existing experiment 2 single-frame pixel ridge: 0.952, 0.979, -0.017, -0.015; two-frame pixel ridge: 0.962, 0.977, 0.205, 0.129. Those use frames 1–3 rather than only frame 3; comparisons are indicative, not identical test examples.

## Design and protocol

All runs use the existing train/validation/test manifests (2400/300/300 independent sequences), first five frames, batch 16, five epochs, AdamW 3e-4, EMA 0.99, seeds 42/43/44. No state or intervention labels enter self-supervised training. The objective is next-context latent MSE + 10×variance-floor penalty + 100×covariance penalty, as in experiment 2 B1. The checkpoint minimizes the same objective on validation batches; training-fit ridge (alpha 100) maps pooled 32-dimensional representations to ball state. Test probes evaluate frame 3; subgroup probes evaluate frame 4 and use the intervention on transition 3→4. For each model, prediction is divided by persistence error in its own EMA latent space. No raw latent MSE is compared across models.

T0 uses the existing framewise eight-query encoder and last frame only, with the existing slot relation/predictor. Unlike experiment 2, all variants here train on one transition per sequence (3→4) rather than all three transitions in the first four frames; the resulting T0 runs are a new matched baseline, not the old checkpoint. T1 and T2 share the same frame CNN and slot encoder. T1 adds two temporal convolutions (the second dilated) per slot with a residual connection; T2 adds a temporal Transformer and learned positions per slot. The final temporal token feeds the same relation and predictor. Target encoding uses the overlapping window shifted by one frame and is stop-gradient. This is one-step prediction between contexts, not a multi-step rollout. Pooling is permutation invariant for the global ball probe; the self-supervised slot loss still assumes that same learned query indices correspond across time. It does not establish object identity.

A fixed-target two-training-sequence overfit check gave initial → final losses: t0 0.03602 → 1.972e-05; t1 0.03342 → 1.008e-05; t2 1.024 → 0.00109. Synthetic tests also verify EMA updates, gradients, shapes and dependence on the first context frame.

## Held-out results

| Model / seed | Position R² x,y | Velocity R² x,y | Effective rank | Std | Prediction / persistence | Params | Train seconds |
|---|---|---|---:|---:|---:|---:|---:|
| t0_seed42 | 0.003, 0.015 | -0.002, -0.030 | 2.07 | 0.1044 | 2.223 | 38112 | 19.8 |
| t1_seed42 | 0.006, 0.009 | -0.004, -0.035 | 4.50 | 0.1120 | 1.099 | 44320 | 19.4 |
| t2_seed42 | 0.008, 0.012 | -0.005, -0.035 | 4.38 | 0.1076 | 1.110 | 46784 | 20.1 |
| t0_seed43 | -0.000, 0.012 | -0.004, -0.032 | 1.01 | 0.0898 | 0.863 | 38112 | 19.9 |
| t1_seed43 | 0.000, 0.012 | -0.005, -0.035 | 4.68 | 0.1103 | 1.178 | 44320 | 19.2 |
| t2_seed43 | 0.003, 0.023 | -0.003, -0.038 | 5.01 | 0.1024 | 1.275 | 46784 | 18.5 |
| t0_seed44 | -0.000, 0.013 | -0.004, -0.033 | 2.03 | 0.0979 | 1.931 | 38112 | 20.2 |
| t1_seed44 | 0.001, 0.008 | -0.004, -0.034 | 4.96 | 0.1108 | 1.183 | 44320 | 18.8 |
| t2_seed44 | 0.004, 0.010 | -0.005, -0.037 | 5.01 | 0.1077 | 1.091 | 46784 | 18.7 |

In this five-epoch comparison, temporal modelling did not meaningfully improve physical-state information: position R² remains near zero and velocity R² is negative across all seeds. T1 and T2 have higher effective rank than T0, but this does not imply a useful state representation. The temporal models also have prediction/persistence ratios above one in all three seeds, so they did not outperform their own persistence baselines.

The wall time includes validation each epoch and checkpoint I/O, but excludes probe evaluation; hardware is the local Apple MPS device (macOS 26.2 arm64, Python 3.11.15, PyTorch 2.15.0). Parameter counts exclude the frozen EMA copy. Per-seed intervention/no-intervention ratios and state probes are in temporal_metrics.json. Effective rank is the exponential entropy of the pooled covariance spectrum.

## Order sensitivity

Each fitted probe is held fixed while its input is reversed, endpoint-preserving shuffled ([2,0,1,3]), or replaced by four copies of the last frame. R² is always scored against the original final-frame state. Reversal changes the final image and therefore confounds temporal direction with position; the endpoint-preserving shuffle and repeat are more specific tests of context dependence, although neither proves a physical motion representation. Normalized changes below are RMS embedding change divided by mean feature standard deviation.

| Model / seed | Shuffle / std | Reverse / std | Repeat / std | Repeated-frame velocity R² x,y |
|---|---:|---:|---:|---|
| t0_seed42 | 0.000 | 1.594 | 0.000 | -0.002, -0.030 |
| t1_seed42 | 0.925 | 1.275 | 1.047 | -0.008, -0.031 |
| t2_seed42 | 0.001 | 1.346 | 0.896 | -0.006, -0.033 |
| t0_seed43 | 0.000 | 1.501 | 0.000 | -0.004, -0.032 |
| t1_seed43 | 0.911 | 1.320 | 1.010 | -0.009, -0.030 |
| t2_seed43 | 0.001 | 1.412 | 0.855 | -0.007, -0.036 |
| t0_seed44 | 0.000 | 1.279 | 0.000 | -0.004, -0.033 |
| t1_seed44 | 0.980 | 1.339 | 1.058 | -0.008, -0.032 |
| t2_seed44 | 0.002 | 1.411 | 0.885 | -0.008, -0.036 |

The T1 representation changes under an endpoint-preserving shuffle, whereas T2 is nearly invariant to that shuffle in these runs despite responding to repeated frames. Neither has useful velocity probes; the responses do not establish directional motion. Interpret the sensitivity values with the corresponding correct-order probe scores: a change in a weak probe does not establish useful directional motion. These short runs are an exploratory comparison, not evidence of causal understanding or long-horizon rollout capability. The restricted four-frame window, one scalar objective, pooled linear probes, same-query slot loss, single validation objective, and three seeds constrain the conclusion.

## Reproduction

From the repository root, with the existing dataset and environment:

```sh
.venv/bin/python scripts/experiment3_supervised.py
.venv/bin/python scripts/experiment3_tiny.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/experiment3_temporal.py --epochs 5 --seeds 42 43 44
.venv/bin/python scripts/plot_experiment3.py
.venv/bin/python scripts/report_experiment3.py
```

The training script refuses to overwrite existing run directories; move or archive experiment3 before reproducing. The interrupted initial T0 attempt is retained and excluded from analysis. Experiment 1 and 2 artifacts and datasets were not modified.

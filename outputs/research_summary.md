# Causal Pinball: controlled JEPA experiment

## Hypothesis

Intervention labels may improve one-step latent prediction, and visually learned object slots may encode physical state better than a global video embedding. These are predictive hypotheses, not tests of causal identification.

## Architecture and data

The dataset has 2,400 training, 300 validation and 300 test independently seeded sequences, each 16 RGB frames at 64×64. This experiment uses the first four frames and three adjacent transitions per sequence. A is a global CNN JEPA. B replaces global pooling with eight learned attention slots and a relational transformer. C adds an embedding of the five intervention indicators for the corresponding transition. B+I is the exact same implementation and objective as C: there is no intervention-specific loss and no valid separate causal-objective ablation. Slots are not guaranteed to correspond to physical objects or stable identities. Simulator states are used only for frozen probes.

## Training methodology

All runs used seed 42, batch size 8, 5 epochs (1,500 optimizer steps), AdamW at 0.0003, the same training split and MPS. Checkpoints were selected by validation MSE plus variance and covariance penalties; no test data were used for selection. The target encoder is stop-gradient and updated with EMA 0.99. A variance floor and covariance penalty discourage, but cannot guarantee avoidance of, collapse. Earlier pilots had near-zero standard deviation or covariance rank near 1. Larger pilots achieved nonzero variance but remained low-rank. Variance is measured across frames and sequences, pooling slots.

The frozen ridge probe is fit on representations and ball x/y position and velocity labels from the training split only, with fixed regularization, and evaluated on the test split. R² may be negative. Persistence uses the target encoder representation of the current frame to predict the next. Interventions are binary indicators for t→t+1; no controlled counterfactual pairs exist here. Confidence intervals bootstrap sequences, not individual frames.

## Results: test split

| Model | Epoch | JEPA MSE [95% CI] | Persistence MSE | Intervention MSE | No intervention MSE | Std | Effective rank / 32 | Probe R²: x, y, vx, vy |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| full_a | 5 | 0.01610 [0.01549, 0.01667] | 0.01489 | 0.01945 | 0.01269 | 0.1086 | 4.76 | 0.025, 0.078, -0.001, -0.011 |
| full_b | 5 | 0.01141 [0.01090, 0.01194] | 0.01668 | 0.01191 | 0.01089 | 0.0996 | 2.53 | 0.026, 0.047, 0.001, -0.007 |
| full_b_info | 4 | 0.00840 [0.00806, 0.00876] | 0.01635 | 0.00855 | 0.00824 | 0.1014 | 2.17 | 0.047, 0.051, 0.001, -0.007 |
| full_c | 4 | 0.00840 [0.00806, 0.00876] | 0.01635 | 0.00855 | 0.00824 | 0.1014 | 2.17 | 0.047, 0.051, 0.001, -0.007 |

MSE values have model-specific latent scales and should not be ranked directly across models. Compare each prediction MSE to its own persistence baseline. Intervened and non-intervened transitions are observational subsets, not matched treatment effects. B+I and C use the same model and seed, so their equality is expected, not evidence of an effect.

## Failure cases and limitations

A single seed and short run cannot establish statistical robustness. Low effective rank indicates substantial dimensional concentration even where per-dimension variance exceeds the heuristic threshold. Learned attention slots may collapse onto similar content; no object-discovery or identity metric was established. Moving target embeddings change MSE scales through training, making loss-curve comparisons difficult. No autoregressive latent transition or visual decoder is implemented, so no multi-step rollout or generated future frame is claimed. Training steps are matched but architecture-dependent compute is not identical. Device allocation logs cover MPS driver allocation during batches, not whole-system unified memory.

## Suggested Substack narrative

Lead with the finding that an apparently excellent predictive loss can hide collapsed representations. Show how variance and effective rank change that interpretation, then compare each model against persistence and inspect frozen physical-state probes. Explain that intervention conditioning supplies information, while a causal claim would require a separately validated objective and suitable intervention or counterfactual evaluation.

## Reproduce

From the repository root, see `scripts/controlled_experiment.py`, `scripts/audit_experiment.py`, and `scripts/evaluate_controlled.py`. Results and exact run configurations are in `outputs/experiments/`; figures are in `outputs/figures/`. To reproduce the four full-split runs with fresh output names:

```sh
for m in a b b_info c; do .venv/bin/python scripts/controlled_experiment.py --model "$m" --name "replicate_$m" --epochs 5 --patience 10; done
.venv/bin/python scripts/evaluate_controlled.py --names replicate_a replicate_b replicate_b_info replicate_c --group replicate --split test
```


# Experiment 2: does anti-collapse help physical representations?

Nine full-split runs completed: B1, B2 and the **supervised reference** B3, each at seeds 42, 43 and 44, five epochs. The existing splits and data were reused. The final epoch was specified in advance. See `outputs/experiment2/methods.md` for objectives, training, probe and baseline definitions. B2's normalized variance floor and covariance penalty were fixed before testing; B3 adds a four-output ball-state head to B1 and uses simulator labels. It is not self-supervised.

The held-out results do **not** support improved physical understanding from B2. Its effective rank is somewhat higher, but raw embedding variance collapses to 0.0032–0.0045 mean standard deviation, and position and velocity probes remain poor. Normalizing the regularized vectors leaves their overall magnitude free to shrink; the raw JEPA MSE may reward this. B3 also failed to learn useful state features under these settings. This failure of the supervised reference limits any attribution of weak JEPA probes to collapse alone: architecture, optimization, loss weighting and framewise input are plausible limitations. No causal claim follows.

Frozen probe R² below is for ball x, y, vx, vy, fitted on training representations and evaluated on all four test frames. Ratios are latent prediction MSE divided by that model's own target-space persistence MSE, separately for each transition subset; below 1 beats persistence. Prediction MSE values from different models have different scales and are not directly comparable.

| Run | Mean std | Effective rank / 32 | Probe R² x, y, vx, vy | Pred / persistence: all | Intervention | No intervention |
|---|---:|---:|---|---:|---:|---:|
| b1_seed42 | 0.0902 | 2.38 | 0.005, 0.110, -0.000, -0.008 | 1.65 | 1.45 | 1.86 |
| b1_seed43 | 0.0970 | 2.07 | -0.000, 0.031, -0.000, -0.009 | 0.54 | 0.54 | 0.53 |
| b1_seed44 | 0.1022 | 2.99 | 0.001, 0.032, 0.001, -0.007 | 1.00 | 0.96 | 1.05 |
| b2_seed42 | 0.0033 | 2.98 | -0.001, 0.016, 0.001, -0.007 | 3.01 | 2.85 | 3.20 |
| b2_seed43 | 0.0045 | 3.05 | -0.001, 0.020, -0.001, -0.008 | 1.35 | 1.42 | 1.28 |
| b2_seed44 | 0.0032 | 3.80 | 0.000, 0.019, 0.001, -0.008 | 0.96 | 1.05 | 0.87 |
| b3_seed42 | 0.0854 | 1.04 | -0.001, 0.016, 0.000, -0.009 | 0.59 | 0.62 | 0.57 |
| b3_seed43 | 0.0961 | 1.01 | 0.001, 0.021, -0.000, -0.007 | 0.61 | 0.68 | 0.53 |
| b3_seed44 | 0.0872 | 1.03 | 0.000, 0.024, 0.000, -0.008 | 0.61 | 0.71 | 0.51 |

There are 454 intervention and 446 non-intervention test transitions. Individual within-model prediction and persistence MSE, subset next-frame probe R²/RMSE, and all seeds are in `outputs/experiment2/metrics.json`. This is a descriptive three-seed comparison, not an uncertainty interval. Unlike the original validation-selected B1 checkpoint, this comparison fixes the final epoch; B1 seed 42 does not beat persistence here.

## Temporal information and baselines

The first four frames contain motion information, but neither the framewise encoder nor its per-frame probes use it. The train-fitted, held-out linear baseline on 8×8 RGB pixels from a single frame obtains position R² (x,y) (0.952, 0.979) and velocity R² (−0.017, −0.015). Adding the preceding-frame difference obtains velocity R² (0.205, 0.129) overall, (0.122, 0.034) for intervention transitions and (0.274, 0.206) for non-intervention transitions. It demonstrates some recoverable velocity information, not that the window determines exact velocity. A supervised oracle using two simulator positions obtains velocity R² (0.162, 0.098) overall, (0.002, −0.022) intervention and (0.296, 0.195) non-intervention; resets, collisions and unobserved within-step dynamics limit inference from displacement. A train-fitted supervised current-state-to-next-state linear reference obtains next velocity R² (0.327, 0.317) overall, (0.326, 0.286) intervention and (0.327, 0.341) non-intervention. Its inputs are ground-truth simulator states, not pixels. All baseline metrics including state persistence and RMSE are in `outputs/experiment2/baselines.json`.

## Alignment and limitations

The generator saves the state and image from the same simulator step, then records the intervention used to advance to the next frame. Sampled replay checks passed for each split. Ball labels are aligned by state-key name, and the B3 head and probes pool slots instead of assuming a ball slot. Same-index training is still an unverified object-identity assumption: learned queries do not provide ground-truth object correspondence. On validation seed 42, same-index assignment was optimal for all 900 target-to-target comparisons for each variant, but B2 slots are nearly identical (mean pairwise slot MSE about 7.2e−11), making assignments uninformative. For prediction-to-target assignment B2 matched the same indices only 16.8% of the time, with essentially indistinguishable assignment costs; B1 and B3 were 100% and 99.8%. These diagnostics do not prove object tracking. See `outputs/experiment2/slot_audit_validation.json`.

Short training, a single fixed regularization choice, only three seeds, no multi-frame encoder, no object masks or verified identities, and the failed supervised reference prevent a general conclusion that anti-collapse cannot help. The high position score of the simple pixel probe makes the encoder's failure especially clear. Effective rank alone is not evidence of physical understanding; even a stronger rank can coexist with shrinking scale and poor state probes. Results and new figures are isolated in `outputs/experiment2/`; experiment-one artifacts are retained.

# Experiment 4: nonlinear frozen probes

A matched frame-3 diagnostic evaluated preselected seed-42 A, B, C, B1–B3 and T0–T2 checkpoints with train-fitted ridge and three modest MLP probe seeds. The MLP recovered more position information from some representations: pooled B1 y-position test R² rose from 0.054 to 0.210; concatenated-slot B1 rose from 0.251 to 0.390. B and C also showed smaller position gains. These remain well below the matched pixel references, and temporal JEPA velocity scores stayed near or below zero. A paired sequence bootstrap for the first fixed probe gave positive, unadjusted intervals for the B and C position gains and B1 y gain; previous test-set inspection and the single encoder seed make these exploratory results.

The MLP did not show wholesale training-set memorization, though some generalization gaps and near-constant predictions occur. These findings qualify the earlier linear-probe conclusion: physical position can be nonlinearly accessible in some frozen encoders, while velocity remains hard to recover under the tested protocols. Probe failure does not prove information is absent. B3 was state-supervised, learned slots are not verified object identities, and the experiment tests neither causal identification nor generated future frames. Full methods, metrics, limitations and figures are in `outputs/experiment4/nonlinear_probe_summary.md`.

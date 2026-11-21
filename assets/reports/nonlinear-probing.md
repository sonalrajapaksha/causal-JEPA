# Experiment 4: nonlinear probes of frozen JEPA representations

## Question and hypothesis

Can modest nonlinear probes recover physical state when linear probes cannot? The hypothesis is that state information may be nonlinearly encoded and therefore more accessible to an MLP.

## Protocol

Preselected seed-42 checkpoints: full A, B and C (validation objective); B1, B2 and B3 (fixed final epoch); and T0, T1 and T2 (validation objective). B3 used simulator-state supervision in encoder training; all others were self-supervised. full B+I duplicates C and is excluded. Earlier investigations have already inspected these models on test data, so this is a follow-up on previously examined checkpoints, not an entirely pristine confirmatory study.

Each split has 2,400/300/300 independent sequences. Simulator RNG seeds and sequence IDs are disjoint across splits. Each sample is frame 3 and the matching same-step ball state; intervention groups refer to transition 2→3. The encoder receives only pixels, is in evaluation mode, and is frozen with no gradients; parameter hashes are unchanged. A and framewise B/C/B1/B2/B3/T0 use frame 3; T1/T2 use frames 0–3, matching training context. The simulator has one ball, selected by named state keys. No learned slot is identified as a ball. Mean slots is permutation invariant; flattened slots preserve learned query order and are permutation sensitive.

Ridge uses the prior train-standardized input procedure and alpha 100. The two-hidden-layer 128/64 GELU MLP uses AdamW, training-only feature/target means and standard deviations, and standardized target MSE. Validation loss selects its epoch; the test set is used only after selection. Three probe seeds are reported as mean ± population SD. Pixel ridge uses alpha 1000 and pooled 8×8 RGB inputs; single-frame and four-frame history references use the same sequences and targets. See config.json for exact settings. Tiny-batch MSE: 1.12 → 5.2e-15.

A separate fixed batch of 32 real training representations and targets reduced standardized MSE from 1.02 to 2.39e-05.

## Results

R² is computed as 1 − SSE / target-centered SST per split; MSE and MAE are in simulator physical units in metrics.json. Negative R² is worse than the split-mean predictor.

| Model | Ridge x, y, vx, vy | MLP mean ± SD x, y, vx, vy |
|---|---|---|
| full_a | 0.011, 0.036, -0.005, -0.030 | 0.011 ± 0.006, 0.034 ± 0.001, -0.009 ± 0.008, -0.035 ± 0.003 |
| full_b | 0.003, 0.030, -0.002, -0.031 | 0.073 ± 0.004, 0.082 ± 0.007, -0.020 ± 0.003, -0.027 ± 0.005 |
| full_c | 0.016, 0.020, -0.002, -0.031 | 0.084 ± 0.004, 0.108 ± 0.002, -0.015 ± 0.002, -0.030 ± 0.000 |
| b1 | -0.001, 0.054, -0.000, -0.027 | 0.002 ± 0.005, 0.210 ± 0.027, -0.013 ± 0.001, -0.033 ± 0.006 |
| b2 | 0.001, 0.007, -0.004, -0.034 | 0.000 ± 0.001, 0.015 ± 0.003, -0.008 ± 0.001, -0.032 ± 0.001 |
| b3 | 0.000, 0.010, -0.004, -0.032 | -0.001 ± 0.000, 0.011 ± 0.006, -0.008 ± 0.001, -0.036 ± 0.001 |
| t0 | 0.003, 0.015, -0.002, -0.030 | 0.006 ± 0.002, 0.037 ± 0.002, -0.005 ± 0.002, -0.034 ± 0.004 |
| t1 | 0.006, 0.009, -0.004, -0.035 | -0.002 ± 0.007, -0.002 ± 0.003, -0.009 ± 0.005, -0.032 ± 0.005 |
| t2 | 0.008, 0.012, -0.005, -0.035 | 0.001 ± 0.001, 0.005 ± 0.008, -0.006 ± 0.002, -0.041 ± 0.002 |
| constant | -0.000, -0.000, -0.004, -0.032 | — |
| pixel_single | 0.933, 0.947, -0.038, -0.050 | — |
| pixel_four | 0.941, 0.947, 0.140, 0.052 | — |

The pooled MLP improves position for B and C and especially B1 y: B1 y rises from 0.054 to 0.210 R². Its flattened slots reach 0.390 versus 0.251 for flattened ridge. Four-frame pixel ridge scores 0.941 / 0.947 for position and 0.140 / 0.052 for velocity; the framewise single-pixel reference scores 0.933 / 0.947 for position. T1 and T2 pooled MLP velocity scores remain negative. This supports outcome D: some position is recoverable nonlinearly in certain encoders, with little generalizable velocity under these probes.

For seed 11, paired test-sequence bootstrap intervals for the MLP-minus-ridge gain exclude zero for both positions in B and C and for B1 y, conditional on the fixed probes; these are unadjusted exploratory intervals after prior test inspection. They do not establish a population-wide or multiple-comparison-corrected effect. The three probe seeds measure optimization variability, not independent model or dataset replications.

The pooled B MLP averages train/test R² of 0.119/0.073 for x and 0.105/0.082 for y: some gap, but no evidence of wholesale memorization. Several other models have near-zero training scores, suggesting underfitting or weak accessible signal rather than a memorization-dominated outcome. Predicted standard deviations for B2 are much smaller than target variation (see metrics.json), consistent with output collapse. Training loss curves, validation R², and train–test gaps provide the corresponding diagnostics.

Train, validation, test, intervention and non-intervention metrics, all per-seed curves, output standard deviations and the full concatenated-slot results are stored in metrics.json. Test subset counts: {'intervention': 157, 'nonintervention': 143}. The bootstrap.json intervals are paired sequence resamples for the first fixed probe seed, conditional on the trained models; they are descriptive and not adjusted for multiple comparisons. Probe-seed standard deviations are not confidence intervals over encoder training or datasets.

## Interpretation and limitations

These are short-trained, small models with one selected encoder seed per family. Compare train and test scores and output standard deviations before interpreting low scores: an MLP may collapse toward its training mean or fit training samples without generalizing. A low frozen-probe score establishes only that the tested readouts, given these data and training settings, could not reliably recover the target; it does not prove the information is absent. The supervised B3 encoder is not evidence about purely self-supervised learning. The fixed-query slot flattening can exploit query ordering but does not establish persistent object identity. Intervention subsets are observational, not causal effects.

Earlier Experiment 2 scores pooled all four frames and Experiment 3 scores frame 3; therefore the new matched single-frame scores are not directly interchangeable with all older scores. The earlier pixel reference pooled frames 1–3; this experiment uses exactly frame 3. The two-frame pixel reference and four-frame pixel reference use their indicated histories, and temporal JEPA comparisons should use the latter. Low velocity recoverability can also reflect limited information in the observed history, collisions, and interventions.

For the article, distinguish predictive success from a demonstration of physical understanding, and state exactly which probes, data, and models support the claim. Neither frozen-probe failure nor a nonlinear gain alone establishes the absence or presence of a general causal world model.

## Reproduction

Run `.venv/bin/python scripts/experiment4.py` and `.venv/bin/python scripts/experiment4_pixel_four.py`, then `.venv/bin/python scripts/experiment4_aggregate.py` and `.venv/bin/python scripts/plot_experiment4.py`. Results and cached representations remain in outputs/experiment4; checkpoint and encoder hashes are recorded.

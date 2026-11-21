# The Predictive Representation Fallacy

Run from the repository root with `.venv/bin/python outputs/substack/generate.py`. The script requires NumPy, Matplotlib, Pillow and FFmpeg with libx264. It reads experiment outputs and existing simulator clips and writes exclusively to this directory. It verifies all nine Experiment 4 checkpoint SHA-256 digests and cross-checks cached encoder hashes and checkpoint epochs against saved metrics.

| Asset | Data and method |
| --- | --- |
| `state_recovery.mp4` and `.gif` | Experiment 4 B1 final epoch 5, encoder seed 42, concatenated slots, MLP seed 11. Saved lead-in frames 0–2 play for 0.5 s each. Frame 3 holds for 2.2 s with its simulator ball center and independently decoded position. Three clips, 11.1 s total at an editorial 10 fps. |
| `embedding_spectrum.png` and `.svg` | Sample covariance across 300 cached pooled online embeddings at observed frame 3. Uses A as a global reference and B1/B2 as the regularization comparison, all seed 42. Sorted eigenvalues are shown as percent of trace and in absolute squared latent units on a log axis. This pooling and timestep differ from the slot-level collapse audits. |
| `linear_vs_nonlinear.png` and `.svg` | All nine preselected Experiment 4 encoder checkpoints, with epoch shown on each row. Fixed train-only ridge and MLPs evaluated on the same pooled features at frame 3. Scores average component-wise R² separately for position and velocity. The MLP diamond averages probe seeds 11, 22 and 33, with each seed shown as a small dot. |
| `prediction_vs_recoverability.png` and `.svg` | All nine final epoch-5 Experiment 2 runs. Each x value is 100 × (1 − latent prediction MSE / same-run latent persistence MSE), on all 900 test transitions. Each y value is a component mean of train-fitted ridge R² on observed state across the four test frames. Negative gain means persistence performed better. |
| `temporal_order.png` and `.svg` | First test-manifest clip illustrates Experiment 3 permutations [0,1,2,3], [3,2,1,0] and [2,0,1,3]. Charts use the saved aggregate responses for all 300 test clips per validation-selected T0/T1/T2 checkpoint, encoder seeds 42–44. Embedding response is RMS change from the correct order divided by mean correct-order embedding standard deviation. The same train-fitted ridge and original t=3 target are used in every condition. Reversal changes which frame appears last, so it also affects the framewise T0 control. |

Set `FFMPEG_BINARY` to an alternate FFmpeg executable when the system binary is unavailable.


## Alignment and example selection

The test manifest and Experiment 4 cache agree on all row IDs, simulation IDs and simulator seeds. The script checks named state columns at zero-based timestep 3 against all cached targets and the prediction archive, then recomputes the archived MLP test R². The generator records state and renders its image before advancing the simulator. The upstream renderer draws the ball center at `ball_x, ball_y` on an equal-aspect 0–32 square with y increasing upward. The animation maps the saved 64×64 RGB raster onto that same extent using an upper raster origin. Position errors are Euclidean distances in simulator units.

The recovery example uses the first archived probe seed for the B1 flat-slot readout discussed in the nonlinear probe report. The three examples come from the 10th, 50th and 90th percentiles of position error across all 300 held-out clips for that readout. The script sorts by error, breaks ties by manifest ID, and selects zero-based ranks `round(q × (N − 1))`. Exact IDs, coordinates and errors are in `state_recovery_selection.json`. The temporal filmstrip uses the first manifest row regardless of response.

Saved nonlinear predictions cover the ball at frame 3 only. The simulator provides the footage for both assets. Continuous nonlinear tracking, other object positions, generated future footage and clip-specific temporal-response scores are outside the available evidence used for these assets. The temporal response panels report test-set aggregates.

## Temporal checkpoint epochs

| Variant | Seed 42 | Seed 43 | Seed 44 |
| --- | ---: | ---: | ---: |
| T0 | 5 | 3 | 3 |
| T1 | 5 | 4 | 4 |
| T2 | 4 | 5 | 5 |

Paths follow `outputs/experiment3/<variant>_seed<seed>/best.pt`. Experiment 2 uses `outputs/experiment2/<variant>_seed<seed>/final.pt`. The Experiment 4 metric entries record the exact checkpoint path for each plotted row.

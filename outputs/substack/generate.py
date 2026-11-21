"""Render article assets using saved test artifacts only."""
import hashlib
import json
from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
E4 = OUT.parent / "experiment4"
BG, INK, MUTED = "#faf8f3", "#18292f", "#53666d"
COLORS = ["#007d86", "#bd4e27", "#6655a3"]
TRUTH, PROBE = "#00e4f0", "#ff9a42"
plt.rcParams.update({
    "animation.ffmpeg_path": os.environ.get("FFMPEG_BINARY", "ffmpeg"),
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": "#bdc6c4",
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": BG,
    "axes.facecolor": BG, "grid.color": "#e2e5de", "svg.fonttype": "none",
})

def read(path):
    return json.loads(path.read_text())

METRICS = read(E4 / "metrics.json")
MODELS = METRICS["models"]
CONFIG = METRICS["config"]
ORDER = read(OUT.parent / "experiment3/temporal_metrics.json")
RUNS = read(OUT.parent / "experiment2/metrics.json")
NAMES = {"full_a": "A", "full_b": "B", "full_c": "C", **{
    k: k.upper() for k in ("b1", "b2", "b3", "t0", "t1", "t2")
}}

def header(fig, title, subtitle):
    fig.text(.085, .955, title, fontsize=22, fontweight="bold", va="top")
    fig.text(.085, .888, subtitle, color=MUTED, fontsize=10, va="top")

def export(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=240)
    plt.close(fig)

def verify_checkpoints():
    for name, entry in MODELS.items():
        checkpoint = ROOT / entry["checkpoint"]
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == entry["checkpoint_sha256"]
        with np.load(E4 / f"{name}_representations.npz") as cache:
            for key in ("checkpoint_sha256", "encoder_sha256", "checkpoint_epoch"):
                assert cache[key].item() == entry[key]

def spectrum():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    header(fig, "How much of the embedding varies?",
           "Experiment 4 cache · Held-out frame 3 · Pooled online embeddings · Encoder seed 42")
    for name, color in zip(("full_a", "b1", "b2"), COLORS):
        with np.load(E4 / f"{name}_representations.npz") as cache:
            z = cache["test_pooled"]
        eig = np.linalg.eigvalsh(np.cov(z, rowvar=False))[::-1].clip(0)
        label = f"{NAMES[name]} · epoch {MODELS[name]['checkpoint_epoch']}"
        rank = np.arange(1, len(eig) + 1)
        axes[0].plot(rank, 100 * eig / eig.sum(), color=color, lw=2, label=label)
        positive = eig > 0
        axes[1].plot(rank[positive], eig[positive], color=color, lw=2)
    for ax in axes:
        ax.set(xlabel="Eigenvalue rank", xlim=(1, 32), xticks=[1, 8, 16, 24, 32])
        ax.grid(axis="y")
    axes[0].set_ylabel("Share of total variance (%)")
    axes[1].set(ylabel="Covariance eigenvalue (latent units²)", yscale="log")
    axes[0].legend(frameon=False)
    fig.subplots_adjust(left=.085, right=.97, bottom=.15, top=.79, wspace=.32)
    export(fig, "embedding_spectrum")

def comparison():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    header(fig, "What can a frozen probe recover?",
           "Experiment 4 · Frame 3 · One selected encoder checkpoint per variant (seed 42)")
    for ax, dims, title in zip(axes, (slice(0, 2), slice(2, 4)), ("Position", "Velocity")):
        for row, (name, entry) in enumerate(MODELS.items()):
            rep = entry["representations"]["pooled"]
            mlp = np.array([np.mean(m["metrics"]["test"]["r2"][dims]) for m in rep["mlp"]])
            ridge = np.mean(rep["linear"]["test"]["r2"][dims])
            ax.plot([ridge, mlp.mean()], [row, row], color="#c8ceca", zorder=1)
            ax.scatter(ridge, row, marker="s", color=COLORS[0], s=32, label="Ridge" if row == 0 else None)
            ax.scatter(mlp, row + np.array([-.15, 0, .15]), color=COLORS[1], s=10, alpha=.55)
            ax.scatter(mlp.mean(), row, marker="D", color=COLORS[1], s=34, label="MLP mean" if row == 0 else None)
        ax.axvline(0, color=MUTED, lw=.8)
        ax.grid(axis="x")
        ax.set(title=title, xlabel="Test R² (mean across two components)")
        ax.set_yticks(range(len(MODELS)), [
            f"{NAMES[name]} · e{entry['checkpoint_epoch']}" + (" †" if name == "b3" else "")
            for name, entry in MODELS.items()])
        ax.invert_yaxis()
    axes[0].legend(frameon=False, loc="upper right")
    fig.text(.085, .035, "Pooled embeddings throughout · Small dots show MLP probe seeds 11, 22, 33 · † State-supervised encoder", color=MUTED)
    fig.subplots_adjust(left=.115, right=.975, bottom=.16, top=.79, wspace=.28)
    export(fig, "linear_vs_nonlinear")

def prediction():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9))
    header(fig, "Does better latent prediction imply better state recovery?",
           "Experiment 2 · Final epoch 5 · Frozen ridge on four observed test frames · One point per encoder seed")
    handles = []
    for family, color in zip(("b1", "b2", "b3"), COLORS):
        for seed, marker in zip((42, 43, 44), ("o", "s", "^")):
            run = RUNS[f"{family}_seed{seed}"]
            gain = 100 * (1 - run["subsets"]["all"]["ratio"])
            for ax, scores in zip(axes, np.split(np.asarray(run["probe"]["r2"]), 2)):
                point = ax.scatter(gain, np.mean(scores), color=color, marker=marker, s=57,
                                   edgecolor=BG, linewidth=.6, zorder=3)
            if seed == 42:
                handles.append(point)
    for ax, title in zip(axes, ("Position", "Velocity")):
        ax.set(title=title, xlabel="Latent MSE reduction vs own persistence (%)",
               ylabel="Test ridge R² (component mean)")
        ax.grid()
        ax.axvline(0, color=MUTED, lw=.8)
        ax.axhline(0, color=MUTED, lw=.8)
    fig.legend(handles, ["B1", "B2", "B3 · state-supervised"], frameon=False,
               loc="lower left", bbox_to_anchor=(.075, .012), ncol=3)
    fig.text(.57, .054, "Encoder seeds   ● 42      ■ 43      ▲ 44", color=MUTED)
    fig.subplots_adjust(left=.09, right=.97, bottom=.22, top=.77, wspace=.31)
    export(fig, "prediction_vs_recoverability")

def load_clips():
    records = [json.loads(line) for line in (ROOT / "data/manifests/test.jsonl").read_text().splitlines()]
    with np.load(E4 / "data.npz") as cache:
        np.testing.assert_array_equal(cache["test_ids"], [row["id"] for row in records])
        np.testing.assert_array_equal(cache["test_simulation_ids"], [row["simulation_id"] for row in records])
        seeds, targets = cache["test_simulator_seeds"], cache["test_y"]
    frames, states = [], []
    for row, seed in zip(records, seeds, strict=True):
        with np.load(ROOT / row["path"]) as clip:
            assert clip["seed"] == seed
            frames.append(clip["frames"][:4])
            keys = clip["state_keys"].tolist()
            states.append(clip["states"][:4, [keys.index(k) for k in CONFIG["target_order"]]])
    states = np.array(states)
    np.testing.assert_array_equal(states[:, 3], targets)
    return records, seeds, np.array(frames), states

def temporal(records, seeds, frames):
    conditions = {"Correct": [0, 1, 2, 3], "Reversed": [3, 2, 1, 0], "Shuffled": [2, 0, 1, 3]}
    fig = plt.figure(figsize=(11, 7.5))
    header(fig, "What changes when frame order changes?",
           f"Experiment 3 · Saved footage of {records[0]['id']} (simulator seed {seeds[0]}) · Frame indices shown below")
    film = fig.add_gridspec(3, 4, left=.25, right=.795, bottom=.465, top=.83, wspace=.04, hspace=.23)
    for row, (label, indices) in enumerate(conditions.items()):
        for col, idx in enumerate(indices):
            ax = fig.add_subplot(film[row, col])
            ax.imshow(frames[0, idx], interpolation="nearest")
            ax.set_axis_off()
            ax.text(.04, .05, f"t={idx}", transform=ax.transAxes, fontsize=8, color="white",
                    bbox=dict(facecolor=INK, edgecolor="none", pad=2))
            if col == 0:
                ax.text(-.17, .5, label, transform=ax.transAxes, ha="right", va="center", fontweight="bold")
    plots = fig.add_gridspec(1, 3, left=.085, right=.97, bottom=.14, top=.385, wspace=.43)
    for panel, title in enumerate(("Embedding change", "Position probe", "Velocity probe")):
        ax = fig.add_subplot(plots[0, panel])
        for model_index, (model, color) in enumerate(zip(("t0", "t1", "t2"), COLORS)):
            for j, condition in enumerate((None, "reverse", "shuffle")):
                for si, (seed, marker) in enumerate(zip((42, 43, 44), ("o", "s", "^"))):
                    run = ORDER[f"{model}_seed{seed}"]
                    if panel == 0:
                        value = run["order"][condition]["representation_rms_change"] / (run["std"] + 1e-12) if condition else 0
                    else:
                        r2 = run["order"][condition]["probe_against_original_state"]["r2"] if condition else run["probe"]["r2"]
                        value = np.mean(r2[2 * (panel - 1):2 * panel])
                    ax.scatter(j + .19 * (model_index - 1) + .053 * (si - 1), value,
                               color=color, marker=marker, s=22,
                               label=model.upper() if panel == j == si == 0 else None)
        ax.set_title(title)
        ax.set_ylabel("RMS change / mean std" if panel == 0 else "Test R² (component mean)")
        ax.set_xticks(range(3), list(conditions), fontsize=8)
        ax.axhline(0, color=MUTED, lw=.7)
        ax.grid(axis="y")
        if panel == 0:
            ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    fig.text(.085, .065, "Responses summarize all 300 test clips per seed. The ridge target stays at original t=3 in every condition.", color=MUTED)
    fig.text(.085, .035, "Validation-selected checkpoints · Encoder seeds   ● 42   ■ 43   ▲ 44 · T0 reads the last input frame", color=MUTED)
    export(fig, "temporal_order")

def recovery(records, seeds, frames, states):
    with np.load(E4 / "b1_flat_predictions.npz") as saved:
        actual, decoded = saved["y"], saved["mlp"]
    np.testing.assert_array_equal(actual, states[:, 3])
    mlp = MODELS["b1"]["representations"]["flat"]["mlp"][0]
    assert mlp["seed"] == 11
    np.testing.assert_allclose(1 - ((actual - decoded)**2).sum(0) /
                               ((actual - actual.mean(0))**2).sum(0),
                               mlp["metrics"]["test"]["r2"], atol=1e-6)
    errors = np.linalg.norm(actual[:, :2] - decoded[:, :2], axis=1)
    sorted_ids = np.lexsort((np.array([r["id"] for r in records]), errors))
    picks = [(q, sorted_ids[round(q * (len(errors) - 1))]) for q in (.1, .5, .9)]
    fig = plt.figure(figsize=(10, 7), dpi=96)
    header(fig, "Recovering the ball’s observed position",
           "Causal Pinball simulator footage · B1 final epoch 5 / encoder seed 42 · Flat-slot MLP / seed 11")
    ax = fig.add_axes([.255, .19, .49, .63])
    image = ax.imshow(frames[0, 0], extent=(0, 32, 0, 32), origin="upper", interpolation="nearest")
    ax.set(xlim=(0, 32), ylim=(0, 32), xlabel="x (simulator units)", ylabel="y (simulator units)")
    ax.set_xticks([0, 8, 16, 24, 32])
    ax.set_yticks([0, 8, 16, 24, 32])
    truth, = ax.plot([], [], "o", ms=13, mfc="none", mec=TRUTH, mew=2)
    probe, = ax.plot([], [], "+", ms=15, color=PROBE, mew=2.4)
    gap, = ax.plot([], [], ls=":", color=INK, lw=1.2)
    status = fig.text(.085, .070, "", fontsize=11)
    detail = fig.text(.085, .032, "", color=MUTED)
    fig.text(.76, .60, "○  Ground truth", color="#006c76", fontsize=11)
    fig.text(.76, .55, "+  MLP decode", color="#a54917", fontsize=11)
    fig.text(.76, .45, "Markers show the\nsame observed step\nat t=3.", color=MUTED)
    fps = 10
    mp4 = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    gif = PillowWriter(fps=fps)
    selection = []
    with mp4.saving(fig, OUT / "state_recovery.mp4", 96), gif.saving(fig, OUT / "state_recovery.gif", 64):
        for q, idx in picks:
            assert np.all((decoded[idx, :2] >= 0) & (decoded[idx, :2] <= 32))
            selection.append({"id": records[idx]["id"], "simulator_seed": int(seeds[idx]),
                              "error_quantile": q, "rank_1_based": int(np.where(sorted_ids == idx)[0][0]) + 1,
                              "timestep": 3, "truth_xy": actual[idx, :2].tolist(),
                              "decoded_xy": decoded[idx, :2].tolist(), "error": float(errors[idx])})
            for step, count in enumerate((5, 5, 5, 22)):
                image.set_data(frames[idx, step])
                status.set_text(f"{records[idx]['id']}   ·   simulator seed {seeds[idx]}   ·   t={step}   ·   error percentile {q:.0%}")
                if step == 3:
                    x, y = actual[idx, :2]
                    px, py = decoded[idx, :2]
                    truth.set_data([x], [y])
                    probe.set_data([px], [py])
                    gap.set_data([x, px], [y, py])
                    detail.set_text(f"Observed-state decode from frame 3   ·   Position error {errors[idx]:.2f} simulator units")
                else:
                    for artist in (truth, probe, gap):
                        artist.set_data([], [])
                    detail.set_text("Observed lead-in footage")
                for _ in range(count):
                    mp4.grab_frame()
                    gif.grab_frame()
    plt.close(fig)
    (OUT / "state_recovery_selection.json").write_text(json.dumps(selection, indent=2) + "\n")

if __name__ == "__main__":
    verify_checkpoints()
    clips = load_clips()
    spectrum()
    comparison()
    prediction()
    temporal(*clips[:3])
    recovery(*clips)

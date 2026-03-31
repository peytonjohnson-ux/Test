"""
Hyperparameter Tuning Loop for the Seizure Detection CNN.

Iterates over a grid of hyperparameters for both 32×32 and 64×64 scalogram
resolutions, trains a model for each combination, and reports the best
configuration ranked by validation AUC.

Usage
-----
    python hyperparameter_tuning.py --eeg_file path/to/unicorn.csv
    python hyperparameter_tuning.py --synthetic          # no hardware needed
"""

import argparse
import itertools
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from cnn_model import build_seizure_cnn, get_callbacks
from eeg_preprocessing import preprocess_eeg, generate_synthetic_eeg

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── Hyperparameter search grid ────────────────────────────────────────────────
PARAM_GRID: dict[str, list[Any]] = {
    # Convolutional filter counts (3 layers)
    "filters":        [(32, 64, 128), (64, 128, 256)],
    # Conv kernel sizes
    "kernel_sizes":   [(3, 3, 3), (5, 3, 3)],
    # Dense head size
    "dense_units":    [64, 128],
    # Dropout regularisation
    "dropout_rate":   [0.3, 0.5],
    # L2 weight decay
    "l2_lambda":      [1e-4, 1e-3],
    # Adam learning rate
    "learning_rate":  [1e-3, 5e-4],
    # Input scalogram resolution
    "image_size":     [32, 64],
}

# Training settings
MAX_EPOCHS   = 50
BATCH_SIZE   = 32
VAL_SPLIT    = 0.2
TEST_SPLIT   = 0.1


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    trial_id:      int
    params:        dict
    val_auc:       float
    val_accuracy:  float
    val_loss:      float
    test_auc:      float
    test_accuracy: float
    duration_sec:  float


# ── Data helpers ──────────────────────────────────────────────────────────────

def prepare_splits(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray,
           np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """Stratified train / val / test split."""
    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, stratify=y, random_state=SEED
    )
    val_frac = VAL_SPLIT / (1 - TEST_SPLIT)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=val_frac, stratify=y_tr, random_state=SEED
    )
    return X_tr, X_val, X_test, y_tr, y_val, y_test


def class_weights(y_train: np.ndarray) -> dict[int, float]:
    """Compute balanced class weights to handle seizure/non-seizure imbalance."""
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return dict(zip(classes.tolist(), weights.tolist()))


# ── Single trial ──────────────────────────────────────────────────────────────

def run_trial(
    trial_id: int,
    params: dict,
    data_cache: dict[int, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> TrialResult:
    """Train one model with the given hyperparameters and return metrics."""
    size = params["image_size"]
    X, y = data_cache[size]
    X_tr, X_val, X_test, y_tr, y_val, y_test = prepare_splits(X, y)

    cw = class_weights(y_tr)
    input_shape = (size, size, 1)
    log_dir = str(output_dir / f"trial_{trial_id:04d}")
    os.makedirs(log_dir, exist_ok=True)

    tf.keras.backend.clear_session()
    model = build_seizure_cnn(
        input_shape    = input_shape,
        filters        = params["filters"],
        kernel_sizes   = params["kernel_sizes"],
        dense_units    = params["dense_units"],
        dropout_rate   = params["dropout_rate"],
        l2_lambda      = params["l2_lambda"],
        learning_rate  = params["learning_rate"],
    )

    t0 = time.time()
    history = model.fit(
        X_tr, y_tr,
        validation_data  = (X_val, y_val),
        epochs           = MAX_EPOCHS,
        batch_size       = BATCH_SIZE,
        class_weight     = cw,
        callbacks        = get_callbacks(log_dir=log_dir, patience=10),
        verbose          = 0,
    )
    duration = time.time() - t0

    val_metrics  = model.evaluate(X_val,  y_val,  verbose=0)
    test_metrics = model.evaluate(X_test, y_test, verbose=0)

    # Metric order: loss, accuracy, auc, precision, recall
    result = TrialResult(
        trial_id      = trial_id,
        params        = params,
        val_loss      = float(val_metrics[0]),
        val_accuracy  = float(val_metrics[1]),
        val_auc       = float(val_metrics[2]),
        test_auc      = float(test_metrics[2]),
        test_accuracy = float(test_metrics[1]),
        duration_sec  = duration,
    )

    # Save individual trial summary
    with open(f"{log_dir}/result.json", "w") as f:
        json.dump(asdict(result), f, indent=2)

    print(
        f"  Trial {trial_id:04d} | size={size}×{size} "
        f"| val_auc={result.val_auc:.4f} "
        f"| test_auc={result.test_auc:.4f} "
        f"| {duration:.1f}s"
    )
    return result


# ── Main tuning loop ──────────────────────────────────────────────────────────

def run_hyperparameter_search(
    eeg_file: str | None = None,
    output_dir: str = "tuning_results",
    max_trials: int | None = None,
    dry_run: bool = False,
) -> list[TrialResult]:
    """
    Full hyperparameter search.

    Parameters
    ----------
    eeg_file    : path to Unicorn CSV; if None, synthetic data is used.
    output_dir  : directory where per-trial logs and the summary are written.
    max_trials  : cap on the number of trials (None = run all combinations).
    dry_run     : if True, only print the grid without training.

    Returns
    -------
    results sorted by val_auc descending.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Build / cache data for each image size ─────────────────────────────
    sizes = PARAM_GRID["image_size"]
    data_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    if eeg_file:
        print(f"\n[tuning] Preprocessing EEG file: {eeg_file}")
        size_results = preprocess_eeg(eeg_file, sizes=tuple(sizes))
        for sz, (X, y) in size_results.items():
            data_cache[sz] = (X, y)
    else:
        print("\n[tuning] Generating synthetic EEG (no hardware file provided)")
        from eeg_preprocessing import (
            extract_windows, bandpass_filter, notch_filter,
            compute_scalogram, scalogram_to_image, UNICORN_FS
        )
        eeg_raw, labels = generate_synthetic_eeg(n_seconds=300, seed=SEED)
        eeg_f = notch_filter(eeg_raw, fs=UNICORN_FS)
        eeg_f = bandpass_filter(eeg_f, 0.5, 50.0, UNICORN_FS)
        windows, win_labels = extract_windows(eeg_f, labels, UNICORN_FS,
                                              window_sec=4.0, overlap=0.5)
        for sz in sizes:
            n = len(windows)
            X = np.zeros((n, sz, sz, 1), dtype=np.float32)
            for i, win in enumerate(windows):
                seg = win[:, 0]
                scalo = compute_scalogram(seg, fs=UNICORN_FS)
                X[i, :, :, 0] = scalogram_to_image(scalo, size=sz)
            data_cache[sz] = (X, win_labels.copy())
            print(f"  Synthetic {sz}×{sz}: {n} windows | "
                  f"seizure={win_labels.sum()} ({win_labels.mean()*100:.1f}%)")

    # ── Enumerate the grid (excluding image_size – handled via data_cache) ──
    grid_keys = [k for k in PARAM_GRID if k != "image_size"]
    grid_vals = [PARAM_GRID[k] for k in grid_keys]
    combos = list(itertools.product(*grid_vals))

    # Cross product with image_size
    all_trials: list[dict] = []
    for size in sizes:
        for combo in combos:
            p = dict(zip(grid_keys, combo))
            p["image_size"] = size
            all_trials.append(p)

    if max_trials:
        # Random subsample for quick exploration
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(all_trials), size=min(max_trials, len(all_trials)),
                         replace=False)
        all_trials = [all_trials[i] for i in idx]

    total = len(all_trials)
    print(f"\n[tuning] {total} trial(s) in search grid")

    if dry_run:
        for i, p in enumerate(all_trials):
            print(f"  [{i:04d}] {p}")
        return []

    # ── Training loop ──────────────────────────────────────────────────────
    results: list[TrialResult] = []
    for trial_id, params in enumerate(all_trials):
        print(f"\n[{trial_id + 1}/{total}] Params: {params}")
        try:
            r = run_trial(trial_id, params, data_cache, out)
            results.append(r)
        except Exception as exc:
            print(f"  ERROR in trial {trial_id}: {exc}")

    # ── Ranking & summary ──────────────────────────────────────────────────
    results.sort(key=lambda r: r.val_auc, reverse=True)

    summary_path = out / "summary.json"
    with open(summary_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"\n{'='*60}")
    print(f"Top-5 Trials (by val_auc)")
    print(f"{'='*60}")
    for r in results[:5]:
        print(
            f"  Trial {r.trial_id:04d} | "
            f"val_auc={r.val_auc:.4f} | "
            f"test_auc={r.test_auc:.4f} | "
            f"size={r.params['image_size']}×{r.params['image_size']}\n"
            f"           params={r.params}"
        )
    print(f"\n[tuning] Full summary saved → {summary_path}")
    return results


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for EEG seizure detection CNN"
    )
    parser.add_argument(
        "--eeg_file", type=str, default=None,
        help="Path to Unicorn Black Hybrid Suite CSV export. "
             "Omit to run on synthetic data."
    )
    parser.add_argument(
        "--output_dir", type=str, default="tuning_results",
        help="Directory for trial logs and summary (default: tuning_results)"
    )
    parser.add_argument(
        "--max_trials", type=int, default=None,
        help="Randomly subsample this many trials from the full grid."
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print the parameter grid without training."
    )
    args = parser.parse_args()

    run_hyperparameter_search(
        eeg_file    = args.eeg_file,
        output_dir  = args.output_dir,
        max_trials  = args.max_trials,
        dry_run     = args.dry_run,
    )

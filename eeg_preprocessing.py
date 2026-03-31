"""
EEG Preprocessing Pipeline for Unicorn Black Hybrid Suite
Converts raw EEG signals into scalograms via Continuous Wavelet Transform (CWT).
Outputs both 32x32 and 64x64 image arrays suitable for CNN input.
"""

import numpy as np
import os
import csv
from scipy import signal
from scipy.signal import butter, filtfilt
import pywt


# ── Unicorn Black Hybrid Suite channel layout ─────────────────────────────────
UNICORN_CHANNELS = [
    "EEG 1", "EEG 2", "EEG 3", "EEG 4",
    "EEG 5", "EEG 6", "EEG 7", "EEG 8",
]
UNICORN_FS = 250          # Hz – default sample rate for Unicorn Black
SEIZURE_LABEL_COL = "Label"  # column name for seizure/non-seizure label (1 / 0)


# ── Signal filtering ──────────────────────────────────────────────────────────

def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float,
                    fs: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter applied per channel."""
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data, axis=0)


def notch_filter(data: np.ndarray, notch_freq: float = 50.0,
                 fs: float = UNICORN_FS, quality: float = 30.0) -> np.ndarray:
    """50 Hz (or 60 Hz) power-line notch filter."""
    b, a = signal.iirnotch(notch_freq, quality, fs)
    return filtfilt(b, a, data, axis=0)


# ── Scalogram generation ──────────────────────────────────────────────────────

def compute_scalogram(segment: np.ndarray, fs: float = UNICORN_FS,
                      wavelet: str = "morl",
                      n_scales: int = 64) -> np.ndarray:
    """
    Compute a CWT scalogram for a single-channel EEG segment.

    Parameters
    ----------
    segment : 1-D array of shape (n_samples,)
    fs      : sampling frequency in Hz
    wavelet : PyWavelets continuous wavelet name
    n_scales: number of frequency scales (rows of the scalogram)

    Returns
    -------
    scalogram : 2-D array of shape (n_scales, n_time_cols)
                Values are the absolute CWT coefficients.
    """
    # Logarithmically spaced scales covering ~1–50 Hz
    freqs = np.logspace(np.log10(1), np.log10(50), n_scales)
    scales = pywt.frequency2scale(wavelet, freqs / fs)

    coeffs, _ = pywt.cwt(segment, scales, wavelet, sampling_period=1.0 / fs)
    return np.abs(coeffs)


def scalogram_to_image(scalogram: np.ndarray,
                       size: int = 64) -> np.ndarray:
    """
    Resize a scalogram to (size, size) using bilinear interpolation
    and normalise to [0, 1].

    Parameters
    ----------
    scalogram : 2-D array (n_scales, n_time)
    size      : target square dimension

    Returns
    -------
    image : float32 array of shape (size, size)
    """
    from PIL import Image  # lazy import – only needed here
    img = Image.fromarray(scalogram.astype(np.float32))
    img = img.resize((size, size), resample=Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    # Min-max normalise per image
    vmin, vmax = arr.min(), arr.max()
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    return arr


# ── EEG file loading ──────────────────────────────────────────────────────────

def load_unicorn_csv(filepath: str) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Load a Unicorn Black Hybrid Suite CSV export.

    The Unicorn software saves a CSV with a header row.  Column names are
    assumed to match UNICORN_CHANNELS, with an optional 'Label' column.

    Returns
    -------
    eeg     : float64 array of shape (n_samples, n_channels)
    labels  : int array of shape (n_samples,)  – 0 if no label column found
    fs      : sampling frequency (float)
    """
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)

    # Try to infer sample rate from a 'SampleRate' header comment if present
    fs = UNICORN_FS

    eeg_data = []
    label_data = []
    channel_cols = [c for c in UNICORN_CHANNELS if c in header]

    if not channel_cols:
        raise ValueError(
            f"No matching EEG channels found in {filepath}. "
            f"Expected columns: {UNICORN_CHANNELS}"
        )

    for row in rows:
        eeg_data.append([float(row[c]) for c in channel_cols])
        if SEIZURE_LABEL_COL in header:
            label_data.append(int(float(row[SEIZURE_LABEL_COL])))
        else:
            label_data.append(0)

    eeg = np.array(eeg_data, dtype=np.float64)
    labels = np.array(label_data, dtype=np.int32)
    return eeg, labels, fs


# ── Windowing & full preprocessing pipeline ───────────────────────────────────

def extract_windows(eeg: np.ndarray, labels: np.ndarray, fs: float,
                    window_sec: float = 4.0,
                    overlap: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over the EEG recording and assign majority-vote labels.

    Returns
    -------
    windows : array of shape (n_windows, n_samples_per_window, n_channels)
    win_labels : array of shape (n_windows,)
    """
    win_len = int(window_sec * fs)
    step = int(win_len * (1 - overlap))
    n_samples = eeg.shape[0]

    windows, win_labels = [], []
    start = 0
    while start + win_len <= n_samples:
        seg = eeg[start: start + win_len]
        lbl = labels[start: start + win_len]
        # Majority vote: mark as seizure if >50 % of samples are labelled 1
        windows.append(seg)
        win_labels.append(1 if lbl.mean() > 0.5 else 0)
        start += step

    return np.array(windows), np.array(win_labels, dtype=np.int32)


def preprocess_eeg(filepath: str,
                   sizes: tuple[int, ...] = (32, 64),
                   lowcut: float = 0.5,
                   highcut: float = 50.0,
                   window_sec: float = 4.0,
                   overlap: float = 0.5,
                   wavelet: str = "morl",
                   n_scales: int = 64,
                   channel_idx: int = 0
                   ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Full preprocessing pipeline: load → filter → window → scalogram → resize.

    Parameters
    ----------
    filepath    : path to Unicorn CSV file
    sizes       : tuple of target image sizes, e.g. (32, 64)
    lowcut      : bandpass lower cutoff (Hz)
    highcut     : bandpass upper cutoff (Hz)
    window_sec  : epoch duration in seconds
    overlap     : fractional overlap between consecutive windows
    wavelet     : CWT mother wavelet
    n_scales    : number of scale/frequency rows before resizing
    channel_idx : which EEG channel to use for the scalogram
                  (set to None to average all channels)

    Returns
    -------
    results : dict mapping each size → (X, y)
              X has shape (n_windows, size, size, 1)  – single-channel image
              y has shape (n_windows,)
    """
    print(f"[preprocess] Loading {filepath}")
    eeg, labels, fs = load_unicorn_csv(filepath)

    print(f"[preprocess] Applying notch + bandpass filters "
          f"({lowcut}–{highcut} Hz)")
    eeg = notch_filter(eeg, fs=fs)
    eeg = bandpass_filter(eeg, lowcut, highcut, fs)

    print(f"[preprocess] Extracting {window_sec}s windows "
          f"(overlap={overlap*100:.0f} %)")
    windows, win_labels = extract_windows(eeg, labels, fs, window_sec, overlap)
    n_windows, win_len, n_ch = windows.shape
    print(f"[preprocess] {n_windows} windows × {win_len} samples × {n_ch} ch")

    results: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for size in sizes:
        X = np.zeros((n_windows, size, size, 1), dtype=np.float32)
        for i, win in enumerate(windows):
            if channel_idx is None:
                seg = win.mean(axis=1)
            else:
                seg = win[:, channel_idx]
            scalo = compute_scalogram(seg, fs=fs, wavelet=wavelet,
                                      n_scales=n_scales)
            img = scalogram_to_image(scalo, size=size)
            X[i, :, :, 0] = img
        results[size] = (X, win_labels.copy())
        print(f"[preprocess] Size {size}×{size}: X.shape={X.shape}")

    return results


# ── Synthetic data generator (for testing without a physical device) ──────────

def generate_synthetic_eeg(n_seconds: int = 120,
                            fs: float = UNICORN_FS,
                            n_channels: int = 8,
                            seizure_ratio: float = 0.3,
                            seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic EEG with labelled seizure bursts for unit testing.

    Seizure segments contain a 3–30 Hz burst superimposed on background noise.
    Non-seizure segments contain only 1/f background noise.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(n_seconds * fs)
    t = np.arange(n_samples) / fs

    # Background: coloured noise (approximate 1/f)
    white = rng.standard_normal((n_samples, n_channels))
    b, a = butter(1, [0.5 / (fs / 2), 40.0 / (fs / 2)], btype="band")
    bg = filtfilt(b, a, white, axis=0) * 20.0  # µV scale

    labels = np.zeros(n_samples, dtype=np.int32)
    n_seizure = int(n_samples * seizure_ratio)
    seizure_start = rng.integers(0, n_samples - n_seizure)
    labels[seizure_start: seizure_start + n_seizure] = 1

    # Seizure: add 10 Hz sinusoidal burst with amplitude ramp
    burst_t = t[seizure_start: seizure_start + n_seizure]
    envelope = np.hanning(n_seizure)
    for ch in range(n_channels):
        freq = rng.uniform(8, 12)
        bg[seizure_start: seizure_start + n_seizure, ch] += (
            50.0 * envelope * np.sin(2 * np.pi * freq * burst_t)
        )

    return bg, labels

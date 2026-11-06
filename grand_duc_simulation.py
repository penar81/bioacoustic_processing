"""Create a simple synthetic European Eagle Owl call from a reference WAV.

This educational script estimates a time-varying fundamental frequency (F0)
and amplitude envelope from one recording, then uses them to shape three sine
wave harmonics. It adds a small fixed echo model and low-level random noise.
The result is a demonstration of signal synthesis, not a biological model or a
replacement for the original recording.

Input: ``grand_duc.wav`` in the project root.
Outputs: ``grand_duc_synthetique.wav`` and ``spectrogram_comparison.png``.
"""

from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import hilbert


ROOT = Path(__file__).resolve().parent
INPUT_WAV = ROOT / "grand_duc.wav"
OUTPUT_WAV = ROOT / "grand_duc_synthetique.wav"
OUTPUT_FIGURE = ROOT / "spectrogram_comparison.png"
FMIN_HZ = 200
FMAX_HZ = 500
FRAME_LENGTH = 2048
HOP_LENGTH = 512
RANDOM_SEED = 42

def normalise(signal: np.ndarray) -> np.ndarray:
    """Scale a waveform to peak amplitude one, leaving silence unchanged."""
    peak = np.max(np.abs(signal))
    return signal if peak == 0 else signal / peak


def estimate_f0_samples(y: np.ndarray, sr: int) -> np.ndarray:
    """Estimate F0 per frame, fill unvoiced gaps, and expand it per sample.

    ``pyin`` may return missing values in unvoiced or uncertain frames. Linear
    interpolation produces a continuous control signal for synthesis; it does
    not make uncertain estimates biologically reliable. The function raises an
    error when too few voiced frames exist to build that signal.
    """
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        sr=sr,
        fmin=FMIN_HZ,
        fmax=FMAX_HZ,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )
    del voiced_flag, voiced_prob  # Only the pitch trajectory is used below.

    f0 = np.nan_to_num(f0)
    voiced_indices = np.flatnonzero(f0 > 0)
    if len(voiced_indices) <= 2:
        raise RuntimeError("Unable to estimate enough F0 values for synthesis")

    interpolated_f0 = np.interp(np.arange(len(f0)), voiced_indices, f0[voiced_indices])
    return np.resize(np.repeat(interpolated_f0, HOP_LENGTH), len(y))


def estimate_amplitude_envelope(y: np.ndarray) -> np.ndarray:
    """Create a smoothed, sample-level amplitude envelope from the waveform.

    The Hilbert transform gives an instantaneous magnitude envelope. Averaging
    it in frames removes fast oscillations, leaving a slower loudness contour
    that can be applied to the synthetic harmonics.
    """
    envelope = np.abs(hilbert(y))
    envelope = np.abs(librosa.effects.preemphasis(envelope, coef=-0.97))
    frame_envelope = librosa.util.frame(
        envelope,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    ).mean(axis=0)
    frame_envelope /= frame_envelope.max()
    return np.resize(np.repeat(frame_envelope, HOP_LENGTH), len(y))


def forest_reverb(signal: np.ndarray, sr: int) -> np.ndarray:
    """Add a few quiet delayed copies as a deliberately simple echo effect."""
    output = signal.copy()
    reflections = [
        (0.030, 0.28),
        (0.060, 0.18),
        (0.110, 0.10),
        (0.180, 0.05),
    ]
    for delay_s, gain in reflections:
        delay_samples = int(delay_s * sr)
        echo = np.zeros_like(signal)
        echo[delay_samples:] = signal[:-delay_samples]
        output += gain * echo
    return output

def save_spectrogram_comparison(original: np.ndarray, synthetic: np.ndarray, sr: int) -> None:
    """Save a two-panel spectrogram image for visual comparison only."""
    figure, axes = plt.subplots(2, 1, figsize=(12, 8))
    for axis, signal, title in zip(axes, (original, synthetic), ("Original", "Synthetic")):
        spectrum_db = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
        librosa.display.specshow(spectrum_db, sr=sr, x_axis="time", y_axis="hz", ax=axis)
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(OUTPUT_FIGURE, dpi=200)
    plt.close(figure)


def run() -> Path:
    """Build the synthetic call, write it to disk, and return its output path."""
    y, sr = librosa.load(INPUT_WAV, sr=None, mono=True)
    y = normalise(y)

    f0_samples = estimate_f0_samples(y, sr)
    phase = 2 * np.pi * np.cumsum(f0_samples) / sr
    # Harmonic weights control tone colour: fundamental, second, then third harmonic.
    synthetic = np.sin(phase) + 0.40 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
    synthetic *= estimate_amplitude_envelope(y)
    synthetic = forest_reverb(synthetic, sr)
    synthetic += np.random.default_rng(RANDOM_SEED).normal(0, 0.002, len(synthetic))
    synthetic = normalise(synthetic)

    sf.write(OUTPUT_WAV, synthetic, sr)
    save_spectrogram_comparison(y, synthetic, sr)
    return OUTPUT_WAV


if __name__ == "__main__":
    output_path = run()

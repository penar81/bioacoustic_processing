"""Extract acoustic features from manually annotated owl vocalisations.

Use this script when you want to analyse one WAV recording and enter its call
boundaries yourself. It does **not** detect calls: an expert supplies each
``start_s`` and ``end_s`` interval in :data:`MANUAL_SEGMENTS`.

Workflow
--------
1. Set ``Config.audio_path`` to a WAV file.
2. Add one dictionary per call to ``MANUAL_SEGMENTS``.
3. Run this file. It writes the annotations, calculated features, and the
   analysis settings to ``Config.output_dir``.

Each call is divided into short overlapping frames. Measurements calculated
per frame (pitch, spectrum, amplitude, and MFCCs) are summarised across the
whole annotated call. Calls shorter than ``frame_length`` samples are retained
in the CSV but their calculated measurements are missing (``NaN``).
"""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np


@dataclass(frozen=True)
class Config:
    """User-editable input, output, pitch, and frame-analysis settings.

    ``sr=None`` in :func:`librosa.load` preserves the recording sample rate.
    A larger ``frame_length`` improves frequency resolution but reduces the
    ability to follow rapid changes through time. ``hop_length`` is the number
    of samples between adjacent analysis frames.
    """

    audio_path: str = "data/RPE068_001.wav"
    output_dir: str = "outputs"

    # Search only for F0 values biologically plausible for these calls.
    # These bounds are not filters: inspect the resulting pitch estimates.
    f0_min_hz: float = 300.0
    f0_max_hz: float = 550.0

    # One analysis frame is 4,096 samples; adjacent frames start 256 samples apart.
    frame_length: int = 4096
    hop_length: int = 256

CONFIG = Config()

# Edit this list to annotate calls. Times are measured in seconds from the
# beginning of the audio file. Calls with the same sequence_id are treated as
# a repetition sequence when calculating inter_onset_interval_s.
MANUAL_SEGMENTS = [
    {"label": "chant_01", "sequence_id": "sequence_01", "start_s": 7.41, "end_s": 8.34},
    {"label": "chant_02", "sequence_id": "sequence_01", "start_s": 20.60, "end_s": 21.45},
    {"label": "chant_03", "sequence_id": "sequence_01", "start_s": 35.12, "end_s": 35.94},
    {"label": "chant_04", "sequence_id": "sequence_01", "start_s": 43.53, "end_s": 44.43},
    {"label": "chant_05", "sequence_id": "sequence_01", "start_s": 50.75, "end_s": 51.60},
    {"label": "chant_06", "sequence_id": "sequence_01", "start_s": 59.33, "end_s": 60.11},
    {"label": "chant_07", "sequence_id": "sequence_01", "start_s": 67.28, "end_s": 68.16},
    {"label": "chant_08", "sequence_id": "sequence_01", "start_s": 76.32, "end_s": 77.10},
    {"label": "chant_09", "sequence_id": "sequence_01", "start_s": 85.43, "end_s": 86.30},
    {"label": "chant_10", "sequence_id": "sequence_01", "start_s": 95.11, "end_s": 95.95},
]


def extract_features(y, sr, annotation, config):
    """Return one feature row for a manually delimited call.

    Parameters
    ----------
    y, sr
        Mono waveform samples and their sample rate in Hz.
    annotation
        A dictionary containing ``label``, ``sequence_id``, ``start_s``, and
        ``end_s``. Boundary times are converted to sample positions here.
    config
        Pitch range and frame settings.

    Notes
    -----
    ``librosa.yin`` estimates F0 once per frame. The spectral features use an
    STFT magnitude spectrum, while MFCCs compactly represent its shape. The
    final CSV contains one summary value per call, rather than per-frame data."""
    start = int(annotation["start_s"] * sr)
    end = int(annotation["end_s"] * sr)
    segment = y[start:end]

    # Begin with a complete schema. NaN makes unavailable measurements explicit
    # and keeps the CSV columns consistent, including for very short calls.
    result = {
        **annotation,
        "duration_s": annotation["end_s"] - annotation["start_s"],
        "inter_onset_interval_s": np.nan,
        "f0_median_hz": np.nan,
        "f0_iqr_hz": np.nan,
        "centroid_mean_hz": np.nan,
        "bandwidth_mean_hz": np.nan,
        "rolloff_mean_hz": np.nan,
        "flatness_mean": np.nan,
        "rms_mean": np.nan,
        "rms_mean_dbfs": np.nan,
        "zcr_mean": np.nan,
        **{f"mfcc_{i}": np.nan for i in range(1, 14)},
    }

    if len(segment) < config.frame_length:
        return result

    # F0 is the fundamental frequency (perceived pitch). YIN estimates one
    # value per frame but does not explicitly validate voicing, so inspect
    # estimates before drawing biological conclusions.
    f0 = librosa.yin(
        segment,
        fmin=config.f0_min_hz,
        fmax=config.f0_max_hz,
        sr=sr,
        frame_length=config.frame_length,
        hop_length=config.hop_length,
        center=False,
    )
    f0 = f0[np.isfinite(f0)]

    spectrum = np.abs(
        librosa.stft(
            segment,
            n_fft=config.frame_length,
            hop_length=config.hop_length,
            center=False,
        )
    )
    # Spectral descriptors describe how sound energy is distributed by frequency.
    centroid = librosa.feature.spectral_centroid(S=spectrum, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(
        S=spectrum, sr=sr, centroid=centroid[np.newaxis, :]
    )[0]
    rolloff = librosa.feature.spectral_rolloff(S=spectrum, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=spectrum)[0]
    rms = librosa.feature.rms(
        S=spectrum, frame_length=config.frame_length, hop_length=config.hop_length
    )[0]
    zcr = librosa.feature.zero_crossing_rate(
        segment,
        frame_length=config.frame_length,
        hop_length=config.hop_length,
        center=False,
    )[0]
    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=sr,
        n_mfcc=13,
        n_fft=config.frame_length,
        hop_length=config.hop_length,
        center=False,
    )

    # Reduce frame-level measurements to one call-level value. The median and
    # interquartile range make F0 less sensitive to occasional extreme frames.
    rms_mean = float(np.mean(rms))
    result.update(
        {
            "f0_median_hz": float(np.median(f0)) if f0.size else np.nan,
            "f0_iqr_hz": (
                float(np.percentile(f0, 75) - np.percentile(f0, 25))
                if f0.size
                else np.nan
            ),
            "centroid_mean_hz": float(np.mean(centroid)),
            "bandwidth_mean_hz": float(np.mean(bandwidth)),
            "rolloff_mean_hz": float(np.mean(rolloff)),
            "flatness_mean": float(np.mean(flatness)),
            "rms_mean": rms_mean,
            "rms_mean_dbfs": float(20 * np.log10(max(rms_mean, np.finfo(float).tiny))),
            "zcr_mean": float(np.mean(zcr)),
            **{f"mfcc_{i}": float(np.mean(row)) for i, row in enumerate(mfcc, start=1)},
        }
    )
    return result


def write_csv(records, path):
    """Write rows with shared keys as a UTF-8 comma-separated file."""
    fieldnames = list(records[0])
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def run(config=CONFIG, annotations=MANUAL_SEGMENTS):
    """Validate annotations, calculate call features, and write three outputs.

    The generated ``*_segments.csv`` preserves manual boundaries,
    ``*_features.csv`` contains one row per call, and ``*_config.json`` records
    settings needed to reproduce this extraction. The returned list is the
    in-memory feature table, useful when importing this module in Python."""
    y, sr = librosa.load(config.audio_path, sr=None, mono=True)
    # Sorting means interval calculations follow time even if the annotation
    # list was entered in another order.
    annotations = sorted(annotations, key=lambda annotation: annotation["start_s"])
    previous_onset = {}
    features = []
    for annotation in annotations:
        start_s = annotation["start_s"]
        end_s = annotation["end_s"]
        if not 0 <= start_s < end_s <= len(y) / sr:
            raise ValueError(
                f"Invalid boundaries for {annotation['label']}: {start_s}, {end_s}"
            )
        sequence_id = annotation["sequence_id"]
        row = extract_features(y, sr, annotation, config)
        if sequence_id in previous_onset:
            row["inter_onset_interval_s"] = start_s - previous_onset[sequence_id]
        previous_onset[sequence_id] = start_s
        features.append(row)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(config.audio_path).stem
    # Keep annotations and calculated measurements separate for traceability.
    write_csv(annotations, output_dir / f"{stem}_segments.csv")
    write_csv(features, output_dir / f"{stem}_features.csv")
    (output_dir / f"{stem}_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return features


if __name__ == "__main__":
    results = run()

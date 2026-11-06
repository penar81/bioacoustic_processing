"""Recalculate acoustic features for all manually annotated recordings.

This is the batch version of ``features_extraction_manual.py``. It reads the
four annotation columns from every ``outputs/input_pca/*_features.csv`` file,
then matches each file to ``data/<recording_id>.wav``. Source CSV files remain
unchanged; new feature tables and settings files are written to ``outputs/``.

Before running, check that every annotation file has ``label``, ``sequence_id``,
``start_s``, and ``end_s`` columns and that its matching WAV exists. The script
stops at the first invalid file so boundary or naming issues can be corrected.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ANNOTATIONS_DIR = ROOT / "outputs" / "input_pca"
OUTPUT_DIR = ROOT / "outputs"
REQUIRED_ANNOTATION_COLUMNS = {"label", "sequence_id", "start_s", "end_s"}


@dataclass(frozen=True)
class Config:
    """Analysis parameters deliberately shared by every input recording.

    F0 is searched only inside the stated range. ``frame_length`` sets the
    samples in each analysis window; ``hop_length`` sets the spacing between
    consecutive windows. Changing these settings changes every derived feature,
    so the written JSON configuration is part of reproducible analysis."""

    f0_min_hz: float = 300.0
    f0_max_hz: float = 550.0
    frame_length: int = 4096
    hop_length: int = 256


CONFIG = Config()


def read_annotations(path: Path) -> list[dict[str, str | float]]:
    """Read, validate, and time-sort the manual annotations in one CSV file.

    Numeric boundaries are checked before loading audio, which gives clear
    errors for malformed annotation files. A ``sequence_id`` identifies calls
    whose onset times can be compared to obtain a repetition interval."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_ANNOTATION_COLUMNS - columns
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

        annotations = []
        for row_number, row in enumerate(reader, start=2):
            try:
                annotation = {
                    "label": row["label"].strip(),
                    "sequence_id": row["sequence_id"].strip(),
                    "start_s": float(row["start_s"]),
                    "end_s": float(row["end_s"]),
                }
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(f"{path.name}, row {row_number}: invalid annotation") from error

            if not annotation["label"] or not annotation["sequence_id"]:
                raise ValueError(f"{path.name}, row {row_number}: empty label or sequence_id")
            if not np.isfinite(annotation["start_s"]) or not np.isfinite(annotation["end_s"]):
                raise ValueError(f"{path.name}, row {row_number}: non-finite boundaries")
            if annotation["start_s"] >= annotation["end_s"]:
                raise ValueError(f"{path.name}, row {row_number}: start_s must be before end_s")
            annotations.append(annotation)

    if not annotations:
        raise ValueError(f"{path.name}: no annotations found")
    return sorted(annotations, key=lambda annotation: annotation["start_s"])


def extract_features(
    y: np.ndarray,
    sr: int,
    annotation: dict[str, str | float],
    config: Config,
) -> dict[str, str | float]:
    """Calculate one call-level feature row from a bounded waveform segment.

    The waveform is split into short frames. YIN produces a frame-level F0
    estimate; STFT-derived measures describe the spectrum; RMS and ZCR describe
    waveform level and rapid sign changes; and 13 MFCCs summarise spectral
    shape. Each measurement is reduced to one mean, median, or IQR for the
    complete annotation. A segment shorter than one frame gets ``NaN`` values."""
    start_s = float(annotation["start_s"])
    end_s = float(annotation["end_s"])
    segment = y[int(start_s * sr):int(end_s * sr)]

    result: dict[str, str | float] = {
        **annotation,
        "duration_s": end_s - start_s,
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
        **{f"mfcc_{index}": np.nan for index in range(1, 14)},
    }

    if len(segment) < config.frame_length:
        return result

    # F0 (fundamental frequency) is an estimate, not a confirmed voiced-call
    # label. Inspect it before interpreting a difference as biological.
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
    # Compute frame-level spectral, amplitude, and cepstral descriptors.
    centroid = librosa.feature.spectral_centroid(S=spectrum, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(
        S=spectrum,
        sr=sr,
        centroid=centroid[np.newaxis, :],
    )[0]
    rolloff = librosa.feature.spectral_rolloff(S=spectrum, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=spectrum)[0]
    rms = librosa.feature.rms(
        S=spectrum,
        frame_length=config.frame_length,
        hop_length=config.hop_length,
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

    # Use robust F0 summaries and means for the remaining frame-level features.
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
            **{
                f"mfcc_{index}": float(np.mean(coefficient))
                for index, coefficient in enumerate(mfcc, start=1)
            },
        }
    )
    return result


def write_csv(records: list[dict[str, str | float]], path: Path) -> None:
    """Write a rectangular UTF-8 CSV feature table."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def process_recording(annotation_path: Path, config: Config) -> Path:
    """Match one annotation CSV to audio, then export recalculated features.

    The first call in a sequence has no preceding onset, so its
    ``inter_onset_interval_s`` remains missing rather than being set to zero.
    The returned path identifies the newly written feature table."""
    stem = annotation_path.name.removesuffix("_features.csv")
    audio_path = DATA_DIR / f"{stem}.wav"
    if not audio_path.exists():
        raise FileNotFoundError(f"{annotation_path.name}: matching audio file not found: {audio_path}")

    annotations = read_annotations(annotation_path)
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration_s = len(y) / sr

    features = []
    previous_onset: dict[str, float] = {}
    for annotation in annotations:
        start_s = float(annotation["start_s"])
        end_s = float(annotation["end_s"])
        if not 0 <= start_s < end_s <= duration_s:
            raise ValueError(
                f"{annotation_path.name}: invalid boundaries for {annotation['label']}: "
                f"{start_s}, {end_s}"
            )

        row = extract_features(y, sr, annotation, config)
        sequence_id = str(annotation["sequence_id"])
        if sequence_id in previous_onset:
            row["inter_onset_interval_s"] = start_s - previous_onset[sequence_id]
        previous_onset[sequence_id] = start_s
        features.append(row)

    feature_path = OUTPUT_DIR / f"{stem}_features.csv"
    write_csv(features, feature_path)
    (OUTPUT_DIR / f"{stem}_config.json").write_text(
        json.dumps(
            {
                "audio_path": str(audio_path.relative_to(ROOT)),
                "annotation_path": str(annotation_path.relative_to(ROOT)),
                **asdict(config),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return feature_path


def run(config: Config = CONFIG) -> list[Path]:
    """Process every PCA input file and return paths to its feature outputs.

    The list is sorted by filename, which makes batch processing deterministic.
    No source file in ``outputs/input_pca`` is overwritten."""
    annotation_paths = sorted(ANNOTATIONS_DIR.glob("*_features.csv"))
    if not annotation_paths:
        raise FileNotFoundError(f"No *_features.csv files found in {ANNOTATIONS_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return [process_recording(path, config) for path in annotation_paths]


if __name__ == "__main__":
    output_paths = run()
    for output_path in output_paths:
        print(output_path)


# European Eagle Owl territorial-call analysis

[Documentation en français](README_fr.md)

This project extracts acoustic descriptors from manually annotated male territorial calls of the European Eagle Owl (*Bubo bubo*), then explores vocalisation structure using principal component analysis (PCA) and boxplots.

All data come from recordings and field observations made by the author.

# ...existing code...

## Analysis principles

- An expert defines the start and end of each vocalisation.
- The scripts do **not** automatically detect call boundaries.
- Acoustic features, including F0, are calculated automatically inside each annotated interval.
- PCA is exploratory and unsupervised: individual IDs are used only to colour plots after PCA fitting.
- The workflow is designed to investigate potential individual acoustic signatures. It does not establish generalisable individual identification from one recording per individual.

## Project layout

```text
signal_processing/
├── data/                         # Source WAV recordings
├── outputs/
│   ├── input_pca/                # Feature CSVs used as annotation sources and PCA input
│   ├── *_segments.csv            # Expert annotations for individual recordings
│   ├── *_features.csv            # Recalculated feature tables
│   └── *.png, *.csv              # PCA and boxplot results
├── analysis_specs/
│   ├── acoustic_feature_dictionary.csv
│   └── acoustic_features_template.csv
├── features_extraction_manual.py # Extract features for one configured recording
├── batch_features_extraction.py  # Re-extract features for all PCA input files
├── pca_analysis.py               # PCA: acoustic, MFCC, and combined feature sets
├── boxplot_analysis.py           # Per-individual feature distributions
├── grand_duc_simulation.py       # Create a synthetic call from F0 and amplitude envelope
├── analysis.ipynb                # Interactive acoustic-signal learning notebook
├── pyproject.toml
├── README.md                     # English documentation
└── README_fr.md                  # French documentation
```

## Requirements

The project uses Python 3.12 or later and the dependencies declared in `pyproject.toml`. PCA also requires `scikit-learn`.

With `uv`, initialise the project environment from the project root:

```bash
uv sync
```

Run scripts with:

```bash
uv run python3 <script_name>.py
```

## Interactive acoustic-analysis notebook

`analysis.ipynb` is an **educational notebook** for learning the main signal-processing representations used in this project. It is separate from the manually annotated feature-extraction pipeline: it does not create `*_features.csv` files, PCA outputs, or boxplots.

The notebook first uses synthetic sine waves to illustrate period, frequency components, the FFT, and the spectrogram. It then loads one Eagle Owl WAV file and displays:

- the waveform (oscillogram);
- a conventional STFT spectrogram;
- a mean spectral envelope;
- Mel-spectrum representations;
- MFCC coefficient maps and their relationship to a conventional spectrogram.

### Run the notebook

1. Open `analysis.ipynb` in Positron.
2. Select the project Python environment (the `.venv` created by `uv sync`) as the notebook kernel.
3. In the `Acoustic features for male Eurasian Eagle Owl` section, update `audio_file` to a WAV file available in `data/`. The current absolute path to `data/hou.wav` is machine-specific and must be changed if that file is unavailable.
4. Run the import cell, then run the cells from top to bottom. Several later cells reuse the audio waveform `y` and sample rate `sr` loaded by earlier cells.

The plots are intended for visual exploration. Values and images in the notebook are not automatically saved to `outputs/`, and its example settings (such as STFT, Mel, and MFCC parameters) may differ from the settings used by `features_extraction_manual.py` and `batch_features_extraction.py`. Use the extraction scripts for reproducible measurements.

## Input data and annotations

### Data provenance and versioning

All data used in this project come from recordings and field observations collected by the author. The `data/` WAV recordings, including `data/hou.wav` used in the notebook, and the annotated feature tables in `outputs/input_pca/` are kept in the repository as analysis inputs. Generated files elsewhere in `outputs/` are ignored by Git; they can be recreated by running the scripts.

### Audio recordings

Place each WAV file in `data/`. File names must match the recording stem used by the feature file. For example:

```text
data/RPE027_001.wav
outputs/input_pca/RPE027_001_features.csv
```

### Annotation columns

Each `*_features.csv` file in `outputs/input_pca/` also acts as the annotation source for batch extraction. It must contain these columns:

| Column | Meaning |
|---|---|
| `label` | Unique vocalisation label, e.g. `chant_01`. |
| `sequence_id` | Identifier of the repetition sequence containing the vocalisation. |
| `start_s` | Expert-annotated onset in seconds. |
| `end_s` | Expert-annotated offset in seconds. |

`sequence_id` is required to calculate `inter_onset_interval_s`. The first vocalisation in each sequence has no preceding onset and therefore receives a missing value for this variable.

## Feature extraction

### Extract one recording

Configure `audio_path` and `MANUAL_SEGMENTS` in `features_extraction_manual.py`, then run:

```bash
uv run python3 features_extraction_manual.py
```

This creates the following files in `outputs/`:

```text
<recording>_segments.csv
<recording>_features.csv
<recording>_config.json
```

### Re-extract all recordings

Use the batch script after changing the extraction procedure or adding descriptors:

```bash
uv run python3 batch_features_extraction.py
```

The script reads annotations from all files matching:

```text
outputs/input_pca/*_features.csv
```

It matches each feature file to `data/<recording>.wav` and writes updated feature tables to `outputs/`. It does **not** modify the source annotation files in `outputs/input_pca/`.

After batch extraction, copy the regenerated `*_features.csv` files from `outputs/` to `outputs/input_pca/` before running PCA or boxplots. Preserve the annotation columns when doing so.

## Extracted descriptors

Each row in a feature table represents one manually delimited vocalisation.

| Group | Variables | Interpretation |
|---|---|---|
| Annotation | `label`, `sequence_id`, `start_s`, `end_s` | Expert-defined boundaries and sequence membership. |
| Timing | `duration_s`, `inter_onset_interval_s` | Call duration and time since the preceding call in the same sequence. |
| Fundamental frequency | `f0_median_hz`, `f0_iqr_hz` | Median F0 and within-call F0 variability, estimated with `librosa.yin()`. |
| Spectral shape | `centroid_mean_hz`, `bandwidth_mean_hz`, `rolloff_mean_hz`, `flatness_mean` | Energy distribution and tonal/noisy spectral properties. |
| Amplitude and waveform | `rms_mean`, `rms_mean_dbfs`, `zcr_mean` | Level and zero-crossing rate. RMS depends on recording conditions and must be interpreted cautiously. |
| Cepstral coefficients | `mfcc_1` to `mfcc_13` | Compact description of spectral envelope/timbre. |

### Detailed descriptor definitions

All descriptors except the manually supplied annotations are calculated from the waveform inside the annotated interval. Frame-level descriptors are calculated with `frame_length = 4096` samples and `hop_length = 256` samples, then summarised as a mean or a robust statistic across frames.

#### Annotation and timing descriptors

| Variable | Unit | Definition and interpretation | Important caveat |
|---|---:|---|---|
| `label` | — | Unique identifier for an annotated vocalisation within a recording (for example, `chant_01`). It is metadata, not an acoustic descriptor. | Do not use as an active PCA variable. |
| `sequence_id` | — | Identifier of the repetition sequence to which the vocalisation belongs. It defines which calls can be compared for rhythmic measures. | Do not use as an active PCA variable. Its biological definition must be made consistently by the expert. |
| `start_s` | s | Expert-annotated onset time, measured from the start of the WAV file. | Its precision depends on the annotation rule used to identify call onset. |
| `end_s` | s | Expert-annotated offset time, measured from the start of the WAV file. | Its precision depends on the annotation rule used to identify call offset. |
| `duration_s` | s | `end_s − start_s`: duration of the complete expert-annotated vocalisation. It can reflect call structure, production rate, or behavioural context. | This descriptor inherits annotation uncertainty; use a documented and repeatable onset/offset rule. |
| `inter_onset_interval_s` | s | Time between the onset of the current vocalisation and the onset of the preceding vocalisation in the same `sequence_id`. It describes repetition tempo. | The first vocalisation in each sequence has no preceding onset, so its value is missing. Do not replace this structural missing value with zero. |

#### Fundamental-frequency descriptors

| Variable | Unit | Definition and interpretation | Important caveat |
|---|---:|---|---|
| `f0_median_hz` | Hz | Median of the frame-level F0 estimates returned by YIN in the selected 300–550 Hz range. The median is used because it is less affected by occasional erroneous frames than the mean. It characterises the typical pitch of a vocalisation. | YIN returns an estimate for each analysable frame but does not itself provide an explicit voicing-confidence decision. Inspect F0 plausibility and stability before biological interpretation. |
| `f0_iqr_hz` | Hz | Interquartile range of frame-level F0: 75th percentile minus 25th percentile. A low value indicates a stable F0; a high value indicates within-call modulation or an unstable estimate. | It is meaningful only when frame-level F0 estimates are credible. It can increase because of tracking errors as well as true modulation. |

#### Spectral descriptors

| Variable | Unit | Definition and interpretation | Important caveat |
|---|---:|---|---|
| `centroid_mean_hz` | Hz | Mean spectral centroid across frames: the energy-weighted mean frequency of the spectrum. Higher values indicate that relatively more energy lies at higher frequencies, often perceived as a brighter sound. | It is influenced by harmonics, propagation, noise, microphone response, and recording distance; it is not equivalent to F0. |
| `bandwidth_mean_hz` | Hz | Mean spectral bandwidth across frames: the energy-weighted spread of frequencies around the spectral centroid. Higher values indicate a broader spectral distribution. | Strongly related to spectral centroid and high-frequency noise; inspect correlations before including both in a reduced feature set. |
| `rolloff_mean_hz` | Hz | Mean spectral roll-off frequency. With the default `librosa` setting, it is the frequency below which 85% of the frame spectral energy is contained. Higher values indicate relatively more high-frequency energy. | The value depends on the analysis bandwidth, recording chain, and background noise. It may be correlated with centroid and bandwidth. |
| `flatness_mean` | ratio | Mean spectral flatness, calculated from the geometric-to-arithmetic mean ratio of spectral energy. Values closer to 0 indicate a more tonal, peaked spectrum; values closer to 1 indicate a flatter, more noise-like spectrum. | It is sensitive to noise and should not be interpreted alone as a direct measure of call quality. |
| `zcr_mean` | proportion of samples | Mean zero-crossing rate: proportion of successive sample pairs whose sign differs. It is a time-domain proxy for rapid waveform changes and often increases with high-frequency or noisy content. | It is not a direct frequency estimate and is sensitive to noise, filtering, gain-independent waveform changes, and sample rate. |

#### Amplitude descriptors

| Variable | Unit | Definition and interpretation | Important caveat |
|---|---:|---|---|
| `rms_mean` | digital amplitude | Mean root-mean-square amplitude across analysis frames. It measures average signal level in the digital recording. | It depends strongly on distance to the microphone, gain, orientation, propagation, equipment, and background conditions. It should not automatically be treated as vocal output intensity. |
| `rms_mean_dbfs` | dBFS | `20 × log10(rms_mean)`: RMS expressed in decibels relative to full scale. Values are typically negative for non-clipped digital recordings; values closer to 0 dBFS correspond to higher recorded levels. | It contains the same information as `rms_mean` on a logarithmic scale. Include only one of these two variables in a given PCA. |

#### MFCC descriptors

`mfcc_1` to `mfcc_13` are the means of the first 13 mel-frequency cepstral coefficients across frames. MFCCs are derived by applying a discrete cosine transform to a log mel-spectrum. Together, they compactly describe the spectral envelope, often interpreted as a measure of timbre.

| Variables | Unit | Interpretation | Important caveat |
|---|---:|---|---|
| `mfcc_1` | coefficient value | In this implementation, it corresponds to the first coefficient returned by `librosa` (often called c0). It is usually strongly related to overall log spectral energy. | It may retain recording-level differences. Amplitude normalisation would be needed if the goal were to make MFCCs less sensitive to recording level. |
| `mfcc_2`–`mfcc_13` | coefficient values | Describe progressively finer patterns in the spectral envelope. They may capture individual differences in harmonic balance and timbre that are not represented by F0 or simple spectral moments. | Individual MFCC coefficients do not have a simple one-to-one biological interpretation. Interpret them as a multivariate set, especially in MFCC-only or combined PCA. |

MFCC values can be positive or negative. A "low" MFCC value does not mean that a vocalisation is weak, poor quality, or low-pitched. Before PCA, MFCCs are z-score standardised, so their influence depends on covariance structure rather than raw numerical magnitude.

### F0 settings

The default F0 range is 300–550 Hz and analysis settings are:

```python
f0_min_hz = 300.0
f0_max_hz = 550.0
frame_length = 4096
hop_length = 256
```

`frame_length` controls frequency resolution; `hop_length` controls the temporal step between frames. F0 is estimated automatically within expert-annotated calls, not used for boundary detection.

## PCA workflow

Run:

```bash
uv run python3 pca_analysis.py
```

The script uses the same complete cases for all three PCAs and standardises every active variable with a z-score. It runs:

1. **Acoustic PCA**: duration, inter-onset interval, F0 and interpretable spectral descriptors.
2. **MFCC PCA**: `mfcc_1` to `mfcc_13`.
3. **Combined PCA**: acoustic descriptors plus MFCCs.

The active variables are defined near the top of `pca_analysis.py` in `ACOUSTIC_FEATURES` and `MFCC_FEATURES`. `rms_mean_dbfs` is optional and is disabled by default. Do not include both `rms_mean` and `rms_mean_dbfs` in the same PCA.

Because `inter_onset_interval_s` is an active variable by default, the first vocalisation from every sequence is excluded from all three PCAs. Remove it from `ACOUSTIC_FEATURES` to retain those vocalisations consistently.

### PCA outputs

For each feature set (`acoustic`, `mfcc`, `combined`), the script produces:

```text
outputs/pca_scores_<feature-set>.csv
outputs/pca_contributions_<feature-set>.csv
outputs/pca_scatterplot_<feature-set>.png
outputs/pca_biplot_<feature-set>.png
outputs/pca_scatterplot_3d_<feature-set>.png
outputs/pca_loadings_<feature-set>.png
```

In the current script, `<feature-set>` is one of `acoustic features`, `MFCCs features`, or `Acoustic and MFCCs features`. These names are used verbatim in output file names.

It also creates `outputs/pca_comparison.csv`, summarising explained variance for PC1–PC3.

- **Scores** locate vocalisations in PCA space.
- **Loadings/contributions** identify variables associated with each principal component.
- **Biplots** combine PC1–PC2 scores and variable directions.
- Individual IDs only colour and mark observations; they are not used to calculate PCA axes.

## Boxplots

Run:

```bash
uv run python3 boxplot_analysis.py
```

The script produces:

```text
outputs/boxplots_acoustic_features.png
outputs/boxplots_mfcc.png
outputs/boxplots_rhythm.png
```

These figures display distributions by individual, with each vocalisation overlaid as a point. They are useful for assessing within-individual variability, between-individual overlap, and potential subgroups before interpreting PCA or clustering.

## Call simulation

`grand_duc_simulation.py` builds a synthetic vocalisation from a `grand_duc.wav` file in the project root. It estimates F0 with `librosa.pyin()`, applies the original signal's amplitude envelope to a harmonic signal, and adds simple reverberation and low-level noise.

Run:

```bash
uv run python3 grand_duc_simulation.py
```

The script writes these files to the project root:

```text
grand_duc_synthetique.wav
spectrogram_comparison.png
```

## Interpretation and study design

- Treat PCA and clustering as exploratory tools.
- A group separated in PCA may reflect individual traits, call type, behavioural context, propagation, microphone settings, distance, or recording session.
- RMS is especially sensitive to recording conditions; compare analyses with and without it.
- MFCCs may improve separation but are less directly interpretable biologically than duration, F0, or simple spectral descriptors.
- A robust test of individual vocal signatures requires multiple independent recordings per individual, ideally across sessions and contexts.

## Reference

A directly relevant study is:

> Grava, T., Mathevon, N., Place, E., & Balluet, P. (2008). Individual acoustic monitoring of the European Eagle Owl *Bubo bubo*. *Ibis*, 150, 279–287.

"""Compare distributions of acoustic measurements between individuals.

For each selected feature, a boxplot summarises values per individual and
jittered dots show the underlying calls. Use these graphics to inspect
within-individual variation, overlap between individuals, unusual calls, and
possible data-quality problems before interpreting PCA results. They are
exploratory visualisations, not significance tests.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "outputs" / "input_pca"
OUTPUT_DIR = ROOT / "outputs"

# Select the variables displayed in each panel. RMS in dBFS depends strongly on
# recording level, distance, and equipment, so compare it with particular care.
ACOUSTIC_FEATURES = [
    "duration_s",
    "f0_median_hz",
    "f0_iqr_hz",
    "centroid_mean_hz",
    "bandwidth_mean_hz",
    "rolloff_mean_hz",
    "flatness_mean",
    "rms_mean_dbfs",
    "zcr_mean",
]

# MFCCs jointly describe spectral shape. Their individual units are coefficients,
# not direct acoustic quantities such as frequency or sound level.
MFCC_FEATURES = [f"mfcc_{index}" for index in range(1, 14)]
# The first call in a repetition sequence has no interval and is omitted here.
RHYTHM_FEATURES = ["inter_onset_interval_s"]


def load_data() -> pd.DataFrame:
    """Load feature tables and derive individual IDs from recording filenames.

    The segment tables are intentionally not used: this script needs calculated
    ``*_features.csv`` files from the PCA input directory."""
    paths = sorted(DATA_DIR.glob("*_features.csv"))
    if not paths:
        raise FileNotFoundError(f"No *_features.csv files found in {DATA_DIR}")

    tables = []
    for path in paths:
        table = pd.read_csv(path)
        table.insert(0, "Id", path.stem.split("_")[0])
        tables.append(table)

    data = pd.concat(tables, ignore_index=True)
    required = ["Id", *ACOUSTIC_FEATURES, *MFCC_FEATURES, *RHYTHM_FEATURES]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return data


def save_boxplot_grid(
    data: pd.DataFrame,
    features: list[str],
    output_path: Path,
    title: str,
    col_wrap: int = 3,
) -> None:
    """Save one faceted boxplot figure with individual call points.

    ``melt`` changes a wide feature table into a long table so Seaborn can draw
    one facet per feature. Box boundaries are quartiles, the central line is the
    median, and dots retain observations that would otherwise be hidden by the
    summary. Missing values are dropped only for their own feature."""
    long_data = data.melt(
        id_vars="Id",
        value_vars=features,
        var_name="feature",
        value_name="value",
    ).dropna(subset=["value"])

    grid = sns.catplot(
        data=long_data,
        x="Id",
        y="value",
        col="feature",
        col_wrap=col_wrap,
        kind="box",
        sharey=False,
        height=3.5,
        aspect=1.1,
        showfliers=False,
    )

    for axis, feature in zip(grid.axes.flat, features):
        feature_data = long_data.loc[long_data["feature"].eq(feature)]
        sns.stripplot(
            data=feature_data,
            x="Id",
            y="value",
            color="black",
            alpha=0.60,
            size=3,
            jitter=0.14,
            ax=axis,
        )
        axis.tick_params(axis="x", rotation=45)
        axis.set_title(feature)
        axis.set_xlabel("")

    grid.fig.subplots_adjust(top=0.90)
    grid.fig.suptitle(title)
    grid.fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(grid.fig)


def run() -> dict[str, Path]:
    """Create acoustic, MFCC, and rhythm plots and return their output paths."""
    data = load_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "acoustic": OUTPUT_DIR / "boxplots_acoustic_features.png",
        "mfcc": OUTPUT_DIR / "boxplots_mfcc.png",
        "rhythm": OUTPUT_DIR / "boxplots_rhythm.png",
    }

    save_boxplot_grid(
        data,
        ACOUSTIC_FEATURES,
        outputs["acoustic"],
        "Distribution of acoustic features by individual",
    )
    save_boxplot_grid(
        data,
        MFCC_FEATURES,
        outputs["mfcc"],
        "Distribution of MFCCs by individual",
        col_wrap=4,
    )
    save_boxplot_grid(
        data,
        RHYTHM_FEATURES,
        outputs["rhythm"],
        "Distribution of inter-onset intervals by individual",
        col_wrap=1,
    )

    return outputs


if __name__ == "__main__":
    generated_files = run()

"""Explore owl-call variation with three principal component analyses (PCAs).

A PCA replaces correlated acoustic variables with new axes (PC1, PC2, PC3)
that successively capture as much variation as possible. It is an exploratory
method: individual IDs colour the plots *after* fitting and do not teach the
PCA how to separate individuals.

The script fits three analyses to exactly the same complete calls: interpretable
acoustic features, MFCCs, and both groups together. Before each PCA, every
feature is z-score standardised, so a variable measured in Hz cannot dominate
one measured on a small numerical scale.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "outputs" / "input_pca"
OUTPUT_DIR = ROOT / "outputs"

# Choose active variables here. The first call in every sequence has no prior
# onset, so retaining inter_onset_interval_s removes it from all three PCAs.
# Remove that variable if those calls should instead be retained.
ACOUSTIC_FEATURES = [
    "duration_s",
    "inter_onset_interval_s",
    "f0_median_hz",
    "f0_iqr_hz",
    "centroid_mean_hz",
    "bandwidth_mean_hz",
    "rolloff_mean_hz",
    "flatness_mean",
    "zcr_mean",
    # "rms_mean_dbfs",  # Optional: do not also add rms_mean.
]
# MFCCs are a multivariate representation of spectral shape; interpret them
# together rather than assigning a simple biological meaning to one coefficient.
MFCC_FEATURES = [f"mfcc_{i}" for i in range(1, 14)]
FEATURE_GROUPS = {
    "acoustic features": ACOUSTIC_FEATURES,
    "MFCCs features": MFCC_FEATURES,
    "Acoustic and MFCCs features": ACOUSTIC_FEATURES + MFCC_FEATURES,
}
MARKERS = ["o", "s", "^", "D", "P", "X", "h", "v", "<", ">"]

def load_data() -> pd.DataFrame:
    """Combine input tables and retain calls complete for every selected feature.

Keeping one common sample makes the three PCA outputs comparable. Text,
missing, and infinite feature values are converted to missing values; a call
missing any feature in the combined set is excluded from all analyses.
"""
    paths = sorted(DATA_DIR.glob("*_features.csv"))
    if not paths:
        raise FileNotFoundError(f"No *_features.csv files found in {DATA_DIR}")

    print(f"Found {len(paths)} feature files in: {DATA_DIR}")
    tables = []
    raw_counts = {}
    for path in paths:
        table = pd.read_csv(path)
        recording_id = path.stem.removesuffix("_features")
        # The filename convention uses the part before the first underscore as
        # the individual identifier, for example RPE027 from RPE027_001.
        individual_id = recording_id.split("_")[0]
        table["Id"] = individual_id
        table["recording_id"] = recording_id
        table["source_file"] = path.name
        raw_counts[path.name] = len(table)
        tables.append(table)
        print(
            f"  Read {path.name}: {len(table)} vocalisations "
            f"(individual={individual_id}, recording={recording_id})"
        )
    data = pd.concat(tables, ignore_index=True)

    required = list(dict.fromkeys(FEATURE_GROUPS["Acoustic and MFCCs features"]))
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Columns missing from CSV files: {missing}")

    data[required] = data[required].apply(pd.to_numeric, errors="coerce")
    data[required] = data[required].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=required).reset_index(drop=True)
    if data.empty:
        raise ValueError("No complete vocalisations for the selected features")
    retained_counts = data["source_file"].value_counts()
    print("Complete vocalisations retained for all three PCAs:")
    for path in paths:
        retained = int(retained_counts.get(path.name, 0))
        print(f"  {path.name}: {retained}/{raw_counts[path.name]} retained")
    return data


def save_plots(
    name: str,
    scores: pd.DataFrame,
    loadings: pd.DataFrame,
    variance: np.ndarray,
    palette: dict,
) -> None:
    """Save score plots, a biplot, and signed-loading charts for one PCA.

Scores locate calls on principal-component axes. Loadings quantify each
original variable's signed association with an axis. The biplot rescales
loading arrows only for visual readability; use the exported contribution
CSV (squared loadings in percent) for numerical comparisons.
"""
    figure, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=scores,
        x="PC1",
        y="PC2",
        hue="Id",
        style="Id",
        palette=palette,
        s=80,
        ax=ax,
    )
    ax.set(
        xlabel=f"PC1 ({variance[0]:.1%})",
        ylabel=f"PC2 ({variance[1]:.1%})",
        title=f"PCA — {name}",
    )

    ax.legend(title="Individuals", bbox_to_anchor=(0.75, 0.70), loc="lower left")

    figure.tight_layout()
    scatterplot_path = OUTPUT_DIR / f"pca_scatterplot_{name}.png"
    figure.savefig(scatterplot_path, dpi=300)
    plt.close(figure)
    print(f"  Saved 2D score plot: {scatterplot_path.name}")

    # Scale loading vectors to the score coordinate system solely for plotting.
    score_scale_pc1 = np.max(np.abs(scores["PC1"]))
    score_scale_pc2 = np.max(np.abs(scores["PC2"]))
    loading_scale_pc1 = score_scale_pc1 / np.max(np.abs(loadings["PC1"])) * 0.80
    loading_scale_pc2 = score_scale_pc2 / np.max(np.abs(loadings["PC2"])) * 0.80

    figure, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=scores,
        x="PC1",
        y="PC2",
        hue="Id",
        style="Id",
        palette=palette,
        s=70,
        alpha=0.80,
        ax=ax,
    )
    for feature, loading in loadings.iterrows():
        arrow_x = loading["PC1"] * loading_scale_pc1
        arrow_y = loading["PC2"] * loading_scale_pc2
        ax.arrow(
            0,
            0,
            arrow_x,
            arrow_y,
            color="black",
            alpha=0.70,
            head_width=0.06,
            length_includes_head=True,
        )
        ax.annotate(
            feature,
            (arrow_x, arrow_y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
            color="black",
        )
    ax.axhline(0, color="grey", linewidth=0.7)
    ax.axvline(0, color="grey", linewidth=0.7)
    ax.set(
        xlabel=f"PC1 ({variance[0]:.1%})",
        ylabel=f"PC2 ({variance[1]:.1%})",
        title=f"PCA biplot — {name}",
    )
    figure.tight_layout()
    biplot_path = OUTPUT_DIR / f"pca_biplot_{name}.png"
    figure.savefig(biplot_path, dpi=300)
    plt.close(figure)
    print(f"  Saved PC1–PC2 biplot: {biplot_path.name}")

    figure = plt.figure(figsize=(11, 8))
    ax = figure.add_subplot(111, projection="3d")
    for individual_id, group in scores.groupby("Id", sort=True):
        ax.scatter(
            group["PC1"], group["PC2"], group["PC3"],
            color=palette[individual_id], label=individual_id,
            s=70, alpha=0.85, edgecolors="black", linewidths=0.4,
        )
    ax.set_xlabel(f"PC1 ({variance[0]:.1%})", labelpad=10)
    ax.set_ylabel(f"PC2 ({variance[1]:.1%})", labelpad=10)
    # Matplotlib may hide the native Z-axis label in exported 3D PNG files.
    figure.text(0.775, 0.50, f"PC3 ({variance[2]:.1%})", rotation=90,
                va="center", ha="center", fontsize=plt.rcParams["axes.labelsize"])
    ax.set_title(f"3D PCA — {name}", pad=18)
    ax.view_init(elev=22, azim=45)
    ax.legend(title="Individuals", bbox_to_anchor=(1.08, 1.0), loc="lower left")
    figure.subplots_adjust(left=0.04, right=0.74, bottom=0.08, top=0.90)
    scatterplot_3d_path = OUTPUT_DIR / f"pca_scatterplot_3d_{name}.png"
    figure.savefig(scatterplot_3d_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"  Saved 3D score plot: {scatterplot_3d_path.name}")

    figure, axes = plt.subplots(1, 2, figsize=(12, max(5, len(loadings) * 0.32)))
    for ax, pc in zip(axes, ("PC1", "PC2")):
        loadings[pc].sort_values().plot.barh(ax=ax)
        ax.set_title(f"{pc} — signed loadings ({name})")
    figure.tight_layout()
    loadings_path = OUTPUT_DIR / f"pca_loadings_{name}.png"
    figure.savefig(loadings_path, dpi=300)
    plt.close(figure)
    print(f"  Saved PC1–PC2 loading plot: {loadings_path.name}")


def run() -> pd.DataFrame:
    """Fit three standardised PCAs, write their results, and return a summary.

Output score CSVs preserve call metadata; contribution CSVs show each
feature's percentage contribution to PC1--PC3. ``pca_comparison.csv``
records explained variance, helping assess how much structure three axes
retain. PCA itself does not establish individual identification or causal
explanations for any visual separation.
"""
    print("Loading feature files...")
    data = load_data()
    print(
        f"Loaded {len(data)} complete vocalisations from "
        f"{data['Id'].nunique()} individuals and "
        f"{data['recording_id'].nunique()} recordings."
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    individuals = sorted(data["Id"].unique())
    colors = sns.color_palette(["black","blue","red","green","purple"], n_colors=len(individuals))
    palette = dict(zip(individuals, colors))
    overview = []

    for name, features in FEATURE_GROUPS.items():
        print(f"Running PCA: {name} ({len(features)} features)...")
        if len(features) != len(set(features)):
            raise ValueError(f"Duplicate variables in the {name} feature set")
        # Standardisation gives each feature mean 0 and standard deviation 1.
        # Constant columns cannot contribute to PCA and are rejected explicitly.
        values = StandardScaler().fit_transform(data[features])
        if np.any(np.std(values, axis=0) == 0):
            raise ValueError(f"Constant variable in the {name} feature set")
        n_components = min(3, len(data), len(features))
        if n_components < 3:
            raise ValueError(f"At least three components are required for {name}")

        pca = PCA(n_components=n_components)
        coordinates = pca.fit_transform(values)
        components = ["PC1", "PC2", "PC3"]
        scores = pd.DataFrame(coordinates, columns=components)
        scores.insert(0, "Id", data["Id"])
        for column in (
            "recording_id",
            "source_file",
            "label",
            "sequence_id",
            "start_s",
            "end_s",
        ):
            if column in data.columns:
                scores[column] = data[column].to_numpy()
        loadings = pd.DataFrame(pca.components_.T, index=features, columns=components)
        # Squared loading: relative variable contribution, expressed as percent.
        contributions = loadings.pow(2).mul(100)
        variance = pca.explained_variance_ratio_

        scores_path = OUTPUT_DIR / f"pca_scores_{name}.csv"
        scores.to_csv(scores_path, index=False)
        contributions_path = OUTPUT_DIR / f"pca_contributions_{name}.csv"
        contributions.rename_axis("feature").to_csv(contributions_path)
        print(f"  Saved scores: {scores_path.name}")
        print(f"  Saved contributions: {contributions_path.name}")
        print(f"Saving plots: {name}...")
        save_plots(name, scores, loadings, variance, palette)
        overview.append({
            "group": name,
            "n_vocalisations": len(data),
            "n_features": len(features),
            "PC1": variance[0],
            "PC2": variance[1],
            "PC3": variance[2],
            "cumulative_PC1_PC2_PC3": variance.sum(),
        })
        print(f"Completed: {name}.")

    result = pd.DataFrame(overview)
    comparison_path = OUTPUT_DIR / "pca_comparison.csv"
    result.to_csv(comparison_path, index=False)
    print(f"Saved PCA comparison: {comparison_path.name}")
    print(f"PCA analysis completed. Results saved to: {OUTPUT_DIR}")
    return result


if __name__ == "__main__":
    comparison = run()

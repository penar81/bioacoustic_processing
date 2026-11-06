# Analyse des chants territoriaux du Grand-duc d’Europe

[English documentation](README.md)

Ce projet extrait des descripteurs acoustiques à partir de chants territoriaux de mâles Grand-duc d’Europe (*Bubo bubo*) délimités manuellement, puis explore la structure des vocalisations par analyse en composantes principales (ACP) et boîtes à moustaches.

Toutes les données sont issues des enregistrements et relevés de l’auteur.

## Principes de l’analyse

- Un expert détermine le début et la fin de chaque vocalisation.
- Les scripts ne détectent **pas** automatiquement les limites des chants.
- Les caractéristiques acoustiques, dont la F0, sont calculées automatiquement à l’intérieur de chaque intervalle annoté.
- L’ACP est exploratoire et non supervisée : les identifiants individuels servent uniquement à colorer les graphiques après le calcul de l’ACP.
- Le flux de travail vise à explorer l’existence de signatures acoustiques individuelles. Avec un seul enregistrement par individu, il ne permet pas de démontrer une identification généralisable des individus.

## Organisation du projet

```text
signal_processing/
├── data/                         # Enregistrements WAV source
├── outputs/
│   ├── input_pca/                # CSV de caractéristiques servant aussi de sources d’annotations et d’entrées ACP
│   ├── *_segments.csv            # Annotations expertes par enregistrement
│   ├── *_features.csv            # Tables de caractéristiques recalculées
│   └── *.png, *.csv              # Résultats ACP et boxplots
├── analysis_specs/
│   ├── acoustic_feature_dictionary.csv
│   └── acoustic_features_template.csv
├── features_extraction_manual.py # Extraction sur un enregistrement configuré manuellement
├── batch_features_extraction.py  # Réextraction pour tous les fichiers d’entrée ACP
├── pca_analysis.py               # ACP : variables acoustiques, MFCC et jeu combiné
├── boxplot_analysis.py           # Distributions des variables par individu
├── analysis.ipynb                # Notebook interactif d’apprentissage du traitement acoustique
├── pyproject.toml
├── README.md                     # Documentation anglaise
└── README_fr.md                  # Documentation française
```

## Prérequis

Le projet utilise Python 3.12 ou une version ultérieure, ainsi que les dépendances déclarées dans `pyproject.toml`. L’ACP nécessite également `scikit-learn`.

Avec `uv`, initialisez l’environnement du projet à la racine du projet :

```bash
uv sync
```

Exécutez les scripts avec :

```bash
uv run python3 <nom_du_script>.py
```

## Notebook interactif d’analyse acoustique

`analysis.ipynb` est un **notebook pédagogique** destiné à apprendre les principales représentations de traitement du signal utilisées dans ce projet. Il est distinct du pipeline d’extraction fondé sur des vocalises annotées manuellement : il ne crée ni fichiers `*_features.csv`, ni résultats d’ACP, ni boxplots.

Le notebook utilise d’abord des ondes sinusoïdales synthétiques pour illustrer la période, les composantes fréquentielles, la FFT et le spectrogramme. Il charge ensuite un fichier WAV de Grand-duc et affiche :

- la forme d’onde (oscillogramme) ;
- un spectrogramme STFT classique ;
- une enveloppe spectrale moyenne ;
- des représentations sur l’échelle de Mel ;
- les cartes de coefficients MFCC et leur relation avec le spectrogramme classique.

### Exécuter le notebook

1. Ouvrez `analysis.ipynb` dans Positron.
2. Sélectionnez comme noyau Python l’environnement du projet (le `.venv` créé par `uv sync`).
3. Dans la section `Acoustic features for male Eurasian Eagle Owl`, remplacez `audio_file` par le chemin d’un WAV présent dans `data/`. Le chemin absolu actuel vers `data/hou.wav` dépend de la machine ; il doit être modifié si ce fichier n’est pas disponible.
4. Exécutez la cellule d’importation, puis les cellules de haut en bas. Plusieurs cellules réutilisent la forme d’onde `y` et la fréquence d’échantillonnage `sr` chargées par des cellules précédentes.

Les graphiques servent à l’exploration visuelle. Le notebook ne sauvegarde pas automatiquement ses figures ni ses valeurs dans `outputs/`, et ses paramètres d’exemple (STFT, Mel et MFCC, notamment) peuvent être différents de ceux de `features_extraction_manual.py` et `batch_features_extraction.py`. Utilisez les scripts d’extraction pour obtenir des mesures reproductibles.

## Données d’entrée et annotations

### Provenance des données et versionnement

Toutes les données utilisées dans ce projet sont issues des enregistrements et des relevés effectués par l’auteur. Les enregistrements WAV de `data/`, dont `data/hou.wav` utilisé dans le notebook, et les tables de caractéristiques annotées de `outputs/input_pca/` sont conservés dans le dépôt comme données d’entrée. Les fichiers générés ailleurs dans `outputs/` sont ignorés par Git ; ils peuvent être recréés en exécutant les scripts.

### Enregistrements audio

Placez chaque fichier WAV dans `data/`. Son nom doit correspondre au préfixe du fichier de caractéristiques. Par exemple :

```text
data/RPE027_001.wav
outputs/input_pca/RPE027_001_features.csv
```

### Colonnes d’annotation

Chaque fichier `*_features.csv` dans `outputs/input_pca/` sert aussi de source d’annotations pour l’extraction par lot. Il doit contenir les colonnes suivantes :

| Colonne | Signification |
|---|---|
| `label` | Identifiant unique de la vocalise, par exemple `chant_01`. |
| `sequence_id` | Identifiant de la séquence de répétition à laquelle appartient la vocalise. |
| `start_s` | Début annoté par l’expert, en secondes. |
| `end_s` | Fin annotée par l’expert, en secondes. |

`sequence_id` est nécessaire au calcul de `inter_onset_interval_s`. La première vocalise de chaque séquence n’a pas de vocalise précédente : cette variable y est donc manquante.

## Extraction des caractéristiques

### Extraire un enregistrement

Configurez `audio_path` et `MANUAL_SEGMENTS` dans `features_extraction_manual.py`, puis exécutez :

```bash
uv run python3 features_extraction_manual.py
```

Le script crée dans `outputs/` :

```text
<enregistrement>_segments.csv
<enregistrement>_features.csv
<enregistrement>_config.json
```

### Réextraire tous les enregistrements

Utilisez le script par lot après une modification de la procédure d’extraction ou l’ajout de variables :

```bash
uv run python3 batch_features_extraction.py
```

Le script lit les annotations dans tous les fichiers :

```text
outputs/input_pca/*_features.csv
```

Il associe chaque fichier au WAV `data/<enregistrement>.wav` et écrit les nouvelles tables de caractéristiques dans `outputs/`. Il ne modifie **jamais** les fichiers source présents dans `outputs/input_pca/`.

Après l’extraction par lot, copiez les fichiers `*_features.csv` régénérés depuis `outputs/` vers `outputs/input_pca/` avant de relancer l’ACP et les boxplots. Conservez les colonnes d’annotation lors de cette opération.

## Variables extraites

Chaque ligne d’une table de caractéristiques correspond à une vocalise délimitée manuellement.

| Groupe | Variables | Interprétation |
|---|---|---|
| Annotation | `label`, `sequence_id`, `start_s`, `end_s` | Bornes temporelles définies par l’expert et appartenance à une séquence. |
| Temps | `duration_s`, `inter_onset_interval_s` | Durée de la vocalise et temps écoulé depuis la vocalise précédente dans la même séquence. |
| Fréquence fondamentale | `f0_median_hz`, `f0_iqr_hz` | F0 médiane et variabilité de F0 au sein de la vocalise, estimées avec `librosa.yin()`. |
| Forme spectrale | `centroid_mean_hz`, `bandwidth_mean_hz`, `rolloff_mean_hz`, `flatness_mean` | Répartition de l’énergie et caractère tonal ou bruité du spectre. |
| Amplitude et forme d’onde | `rms_mean`, `rms_mean_dbfs`, `zcr_mean` | Niveau et taux de passages par zéro. Le RMS dépend des conditions d’enregistrement et doit être interprété avec prudence. |
| Coefficients cepstraux | `mfcc_1` à `mfcc_13` | Description compacte de l’enveloppe spectrale et du timbre. |

### Définition détaillée des variables temporelles et de F0

Toutes les variables, à l’exception des annotations manuelles, sont calculées dans l’intervalle annoté. Les variables calculées trame par trame utilisent `frame_length = 4096` échantillons et `hop_length = 256` échantillons, puis sont résumées sur les trames.

| Variable | Unité | Définition et interprétation | Précaution |
|---|---:|---|---|
| `label` | — | Identifiant unique d’une vocalise, par exemple `chant_01`. C’est une métadonnée, et non une mesure acoustique. | Ne pas l’utiliser comme variable active dans l’ACP. |
| `sequence_id` | — | Identifiant d’une séquence de répétition. Il indique quelles vocalises peuvent être comparées pour une mesure rythmique. | Ne pas l’utiliser comme variable active dans l’ACP. La règle de définition des séquences doit rester cohérente. |
| `start_s` | s | Instant de début annoté par l’expert depuis le début du WAV. | La précision dépend de la règle utilisée pour définir le début du chant. |
| `end_s` | s | Instant de fin annoté par l’expert depuis le début du WAV. | La précision dépend de la règle utilisée pour définir la fin du chant. |
| `duration_s` | s | `end_s − start_s` : durée totale de la vocalise délimitée par l’expert. Elle peut refléter la structure du chant, le mode de production ou le contexte comportemental. | Elle hérite de l’incertitude d’annotation. Utilisez une règle explicite et répétable de début et de fin. |
| `inter_onset_interval_s` | s | Temps entre le début de la vocalise courante et le début de la vocalise précédente dans la même `sequence_id`. Elle décrit le tempo de répétition. | La première vocalise d’une séquence n’a pas de vocalise précédente : la valeur est structurellement manquante et ne doit pas être remplacée par zéro. |
| `f0_median_hz` | Hz | Médiane des F0 estimées par trame par YIN dans la plage 300–550 Hz. La médiane est moins sensible que la moyenne aux trames erronées. Elle représente la hauteur typique de la vocalise. | YIN ne fournit pas de décision explicite de confiance de voisement. Vérifiez la plausibilité et la stabilité du suivi de F0 avant l’interprétation biologique. |
| `f0_iqr_hz` | Hz | Écart interquartile de F0 : 75e percentile moins 25e percentile des valeurs par trame. Une faible valeur indique une F0 stable ; une valeur élevée indique une modulation intra-vocalise ou une estimation instable. | Elle est pertinente uniquement si les F0 par trame sont crédibles ; les erreurs de suivi peuvent aussi augmenter cette valeur. |

##### Variables spectrales, d’amplitude et MFCC

| Variable | Unité | Définition et interprétation | Précaution |
|---|---:|---|---|
| `centroid_mean_hz` | Hz | Moyenne du centroïde spectral : moyenne des fréquences pondérée par l’énergie spectrale. Une valeur élevée indique relativement plus d’énergie aux hautes fréquences, souvent perçue comme un son plus brillant. | Elle est influencée par les harmoniques, la propagation, le bruit, la réponse du microphone et la distance. Elle n’est pas équivalente à la F0. |
| `bandwidth_mean_hz` | Hz | Moyenne de la largeur spectrale : dispersion fréquentielle pondérée par l’énergie autour du centroïde. Une valeur élevée indique un spectre plus étalé. | Elle est liée au centroïde spectral et au bruit de haute fréquence ; examinez les corrélations avant de conserver les deux dans un jeu réduit. |
| `rolloff_mean_hz` | Hz | Moyenne de la fréquence de roll-off. Avec le paramétrage par défaut de `librosa`, c’est la fréquence sous laquelle se trouvent 85 % de l’énergie spectrale d’une trame. Une valeur élevée indique relativement plus d’énergie dans les hautes fréquences. | Elle dépend de la bande analysée, de la chaîne d’enregistrement et du bruit de fond. Elle peut être corrélée au centroïde et à la largeur spectrale. |
| `flatness_mean` | rapport | Moyenne de la platitude spectrale : rapport entre moyenne géométrique et moyenne arithmétique de l’énergie spectrale. Une valeur proche de 0 indique un spectre tonal avec des pics ; une valeur proche de 1 indique un spectre plus plat et bruité. | Sensible au bruit ; elle ne doit pas être considérée seule comme une mesure directe de qualité du chant. |
| `zcr_mean` | proportion d’échantillons | Moyenne du taux de passages par zéro : proportion de paires d’échantillons successifs dont le signe diffère. C’est un indicateur de changements rapides de forme d’onde, souvent plus élevé avec du contenu aigu ou bruité. | Ce n’est pas une estimation directe de fréquence. Elle est sensible au bruit, au filtrage, à la forme d’onde et à la fréquence d’échantillonnage. |
| `rms_mean` | amplitude numérique | Moyenne de l’amplitude efficace (root mean square) sur les trames. Elle représente le niveau moyen du signal dans l’enregistrement numérique. | Elle dépend fortement de la distance au microphone, du gain, de l’orientation, de la propagation, du matériel et du bruit de fond. Elle ne mesure donc pas automatiquement l’intensité vocale produite. |
| `rms_mean_dbfs` | dBFS | `20 × log10(rms_mean)` : RMS exprimé en décibels relatifs à la pleine échelle numérique. Les valeurs sont généralement négatives hors saturation ; des valeurs plus proches de 0 dBFS correspondent à un niveau enregistré plus élevé. | Elle contient la même information que `rms_mean`, à l’échelle logarithmique. N’incluez qu’une seule de ces deux variables dans une même ACP. |

`mfcc_1` à `mfcc_13` sont les moyennes des 13 premiers coefficients cepstraux en fréquences Mel sur les trames. Ils sont calculés en appliquant une transformée en cosinus discrète au spectre Mel logarithmique. Ensemble, ils décrivent de façon compacte l’enveloppe spectrale, souvent associée au timbre.

| Variables | Unité | Interprétation | Précaution |
|---|---:|---|---|
| `mfcc_1` | valeur de coefficient | Dans cette implémentation, c’est le premier coefficient renvoyé par `librosa` (souvent appelé c0). Il est généralement fortement lié à l’énergie spectrale globale en échelle logarithmique. | Il peut conserver des différences de niveau d’enregistrement. Une normalisation d’amplitude serait nécessaire pour rendre les MFCC moins sensibles au niveau d’acquisition. |
| `mfcc_2` à `mfcc_13` | valeurs de coefficients | Ils décrivent des motifs de plus en plus fins de l’enveloppe spectrale. Ils peuvent représenter des différences d’équilibre harmonique et de timbre qui ne sont pas décrites par la F0 ou les moments spectraux simples. | Chaque coefficient ne possède pas d’interprétation biologique simple et univoque. Interprétez-les comme un ensemble multivarié, notamment dans les ACP MFCC ou combinées. |

Les MFCC peuvent être positifs ou négatifs. Une valeur de MFCC « faible » ne veut pas dire qu’une vocalise est faible, de mauvaise qualité ou grave. Avant l’ACP, les MFCC sont standardisés par z-score : leur influence dépend donc de leur structure de covariance plutôt que de leur amplitude numérique brute.

### Paramètres de F0

La plage de F0 par défaut est de 300 à 550 Hz. Les paramètres d’analyse sont :

```python
f0_min_hz = 300.0
f0_max_hz = 550.0
frame_length = 4096
hop_length = 256
```

`frame_length` contrôle la résolution fréquentielle ; `hop_length` contrôle le pas temporel entre les trames. La F0 est estimée automatiquement à l’intérieur des chants annotés, sans être utilisée pour détecter leurs limites.

## Flux de travail ACP

Exécutez :

Le script standardise chaque variable active (z-score) et utilise les mêmes vocalises complètes dans les trois ACP :

1. **ACP acoustique** : durée, intervalle inter-onset, F0 et variables spectrales interprétables.
2. **ACP MFCC** : `mfcc_1` à `mfcc_13`.
3. **ACP combinée** : variables acoustiques et MFCC.

Les variables actives sont définies en tête de `pca_analysis.py`, dans `ACOUSTIC_FEATURES` et `MFCC_FEATURES`. `rms_mean_dbfs` est facultative et désactivée par défaut. N’incluez pas simultanément `rms_mean` et `rms_mean_dbfs` dans une même ACP.

Comme `inter_onset_interval_s` est active par défaut, la première vocalise de chaque séquence est exclue des trois ACP. Retirez cette variable de `ACOUSTIC_FEATURES` pour conserver systématiquement ces vocalises.

### Sorties de l’ACP

Pour chaque jeu de variables, le script produit :

```text
outputs/pca_scores_<jeu-de-variables>.csv
outputs/pca_contributions_<jeu-de-variables>.csv
outputs/pca_scatterplot_<jeu-de-variables>.png
outputs/pca_biplot_<jeu-de-variables>.png
outputs/pca_scatterplot_3d_<jeu-de-variables>.png
outputs/pca_loadings_<jeu-de-variables>.png
```

Dans le script actuel, `<jeu-de-variables>` vaut littéralement `acoustic features`, `MFCCs features` ou `Acoustic and MFCCs features`. Ces noms sont repris tels quels dans les noms de fichiers.

Il crée aussi `outputs/pca_comparison.csv`, qui récapitule la variance expliquée par PC1 à PC3.

- Les **scores** positionnent les vocalises dans l’espace ACP.
- Les **loadings/contributions** indiquent les variables associées à chaque composante principale.
- Les **biplots** combinent les scores PC1–PC2 et la direction des variables.
- Les identifiants individuels servent uniquement à colorer et différencier les points ; ils ne participent pas au calcul des axes.

## Boxplots

Exécutez :

```bash
uv run python3 boxplot_analysis.py
```

Le script génère :

```text
outputs/boxplots_acoustic_features.png
outputs/boxplots_mfcc.png
outputs/boxplots_rhythm.png
```

Les figures montrent les distributions par individu, avec les vocalises individuelles superposées sous forme de points. Elles permettent d’examiner la variabilité intra-individuelle, le chevauchement entre individus et d’éventuels sous-groupes avant d’interpréter l’ACP ou un clustering.

## Simulation de chant

`grand_duc_simulation.py` construit une vocalisation synthétique à partir d’un fichier `grand_duc.wav` situé à la racine du projet. Le script estime la F0 avec `librosa.pyin()`, applique l’enveloppe d’amplitude du signal d’origine à une somme d’harmoniques et ajoute une réverbération simple ainsi qu’un faible bruit.

Exécutez :

```bash
uv run python3 grand_duc_simulation.py
```

Le script crée les fichiers suivants à la racine du projet :

```text
grand_duc_synthetique.wav
spectrogram_comparison.png
```

## Interprétation et plan d’étude

- Considérez l’ACP et le clustering comme des outils exploratoires.
- Une séparation en ACP peut refléter les traits individuels, le type de vocalise, le contexte comportemental, la propagation, le réglage du microphone, la distance ou la session d’enregistrement.
- Le RMS est particulièrement sensible aux conditions d’acquisition : comparez les résultats avec et sans RMS.
- Les MFCC peuvent améliorer la séparation, mais ils sont moins interprétables biologiquement que la durée, la F0 ou les descripteurs spectraux simples.
- Une évaluation robuste des signatures vocales individuelles exige plusieurs enregistrements indépendants par individu, idéalement dans plusieurs sessions et contextes.

## Référence

Une étude directement pertinente est :

> Grava, T., Mathevon, N., Place, E., & Balluet, P. (2008). Individual acoustic monitoring of the European Eagle Owl *Bubo bubo*. *Ibis*, 150, 279–287.

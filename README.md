# Vote pondéré de Golub appliqué au cancer du sein (WDBC)

Application de la méthode du vote pondéré (*weighted voting*) de Golub et al. (1999) au jeu de
données Wisconsin Diagnostic Breast Cancer (WDBC), puis comparaison avec une régression logistique.

Le projet répond à la question : *le vote pondéré de Golub classe-t-il les échantillons mammaires
malins de façon précise et robuste ?*

## Prérequis

- [uv](https://docs.astral.sh/uv/)
- Python 3.12 ou plus récent 

## Installation

Cloner le dépôt puis se placer dans le dossier du projet :

```bash
git clone https://github.com/Zoxxide/golub_wdbc.git
cd golub_wdbc
```

Installer les dépendances (étape facultative, `uv run` le fait automatiquement au premier lancement) :

```bash
uv sync
```

uv crée un environnement virtuel `.venv/` et y installe les versions exactes des dépendances
indiquées dans `uv.lock` (matplotlib, numpy, pandas, scikit-learn).

## Utilisation

Une seule commande lance tout le pipeline, depuis la racine du projet :

```bash
uv run src/main.py
```

Le script enchaîne les cinq étapes de l'analyse :

1. exploration des données ;
2. implémentation du vote pondéré ;
3. évaluation du vote pondéré par validation croisée stratifiée (5 folds répétés 3 fois, soit 15 modèles) ;
4. stabilité de la sélection des variables ;
5. comparaison avec une régression logistique sur les mêmes 15 découpages.

Les résultats chiffrés (AUC, accuracy, balanced accuracy, sensibilité, similarité de Jaccard)
s'affichent dans le terminal, et les figures sont enregistrées dans le dossier `results/`.

### Paramètres

Les paramètres se modifient en haut de `src/main.py` :

| Paramètre | Valeur | Rôle |
|---|---|---|
| `SEED` | `12` | graine personnelle, utilisée pour les découpages de la validation croisée |
| `N_FOLDS` | `5` | nombre de folds |
| `N_REPEATS` | `3` | nombre de répétitions de la validation croisée |

Avec ces valeurs, les résultats sont reproduits à l'identique à chaque lancement.

## Project structure

```
golub_wdbc/
├── data/
│   ├── wdbc.csv                 # jeu de données WDBC fourni (569 échantillons, 30 variables)
│   ├── MANIFEST.json            # provenance, effectifs des classes et SHA256 du fichier
│   └── DATA_DICTIONARY.md       # description des variables et de leur mesure
├── src/
│   ├── main.py                  # paramètres et enchaînement de toutes les étapes
│   ├── utils.py                 # chargement des données, création des splits, figures du modèle exemple
│   ├── explore_data.py          # partie 1 : exploration des données
│   ├── weight.py                # partie 2 : classe WeightedVoting (vote pondéré)
│   ├── evaluate_model.py        # partie 3 : évaluation du vote pondéré
│   ├── evaluate_stability.py    # partie 4 : stabilité de la sélection des variables
│   └── compare.py               # partie 5 : régression logistique et comparaison
├── results/                     # figures générées par le pipeline
├── pyproject.toml               # description du projet et dépendances
├── uv.lock                      # versions exactes des dépendances
└── README.md
```

Les fichiers de `data/` sont fournis avec le sujet et n'ont pas été modifiés.

## Figures produites

| Fichier | Contenu |
|---|---|
| `repartition_diagnostic.png` | répartition des diagnostics bénins et malins |
| `auc_variables.png` | lien de chaque variable avec le diagnostic (AUC de la variable prise seule) |
| `wv_confusion_matrix.png` | vote pondéré, modèle exemple : matrice de confusion |
| `wv_roc.png` | vote pondéré, modèle exemple : courbe ROC |
| `frequence_variable.png` | nombre de sélections de chaque variable sur les 15 modèles |
| `lr_confusion_matrix.png` | régression logistique, modèle exemple : matrice de confusion |
| `lr_roc.png` | régression logistique, modèle exemple : courbe ROC |

Le modèle exemple est le premier des 15 modèles (répétition 1, fold 1). Il est le même pour les
deux méthodes, ce qui permet de les comparer sur les mêmes échantillons de test.

## Résultats principaux

Moyenne ± écart type sur les 15 modèles, avec la graine 12 :

| Mesure | Vote pondéré | Régression logistique |
|---|---|---|
| AUC | 0.962 ± 0.017 | 0.994 ± 0.005 |
| Accuracy | 0.889 ± 0.026 | 0.978 ± 0.015 |
| Balanced accuracy | 0.855 ± 0.032 | 0.975 ± 0.017 |
| Sensibilité | 0.725 ± 0.059 | 0.962 ± 0.028 |

La sélection des variables du vote pondéré est très stable (similarité de Jaccard moyenne de 0.976),
mais le modèle ne détecte qu'environ trois cancers sur quatre. L'analyse complète est détaillée
dans le rapport.

## Références

- Golub, T. R. et al. (1999). Molecular classification of cancer: class discovery and class
  prediction by gene expression monitoring. *Science*, 286(5439), 531–537.
  doi:10.1126/science.286.5439.531
- Wolberg, W., Mangasarian, O., Street, N., Street, W. (1995). Breast Cancer Wisconsin (Diagnostic).
  UCI Machine Learning Repository. doi:10.24432/C5DW2B
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine
  Learning Research*, 12, 2825–2830. Utilisé pour la validation croisée, les métriques,
  `StandardScaler` et `LogisticRegression`.

## Author

Aurélien Ricaud – aurelien.ricaud@epita.fr

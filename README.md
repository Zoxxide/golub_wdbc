# Vote pondéré de Golub appliqué au cancer du sein (WDBC)

Application de la méthode du vote pondéré de Golub et al. (1999) au jeu de données
Wisconsin Diagnostic Breast Cancer (WDBC), et comparaison avec une régression logistique.

## Lancer le projet

Il faut avoir [uv](https://docs.astral.sh/uv/) installé. Depuis ce dossier, une seule commande
lance tout le pipeline :

```bash
uv run src/main.py
```

uv crée l'environnement et installe les dépendances à partir de `pyproject.toml` et `uv.lock`,
puis le script affiche les résultats dans le terminal et enregistre toutes les figures dans
le dossier `results/`.

La graine personnelle utilisée est **12**. Elle se règle avec la variable `SEED` dans `src/main.py`.

## Organisation du code

| Fichier | Rôle |
|---|---|
| `src/main.py` | paramètres (graine, nombre de folds et de répétitions) et enchaînement de toutes les étapes |
| `src/utils.py` | chargement des données, création des 15 splits, figures du modèle exemple |
| `src/explore_data.py` | partie 1 : exploration des données |
| `src/weight.py` | partie 2 : classe `WeightedVoting`, implémentation du vote pondéré |
| `src/evaluate_model.py` | partie 3 : évaluation du vote pondéré par validation croisée |
| `src/evaluate_stability.py` | partie 4 : stabilité de la sélection des variables |
| `src/compare.py` | partie 5 : régression logistique sur les mêmes splits et comparaison |

## Résultats produits

Les mesures (AUC, accuracy, balanced accuracy, sensibilité, similarité de Jaccard) sont
affichées dans le terminal. Les figures sont enregistrées dans `results/` :

| Figure | Contenu |
|---|---|
| `repartition_diagnostic.png` | répartition des diagnostics bénins et malins |
| `auc_variables.png` | lien de chaque variable avec le diagnostic (AUC) |
| `wv_confusion_matrix.png`, `wv_roc.png` | vote pondéré, modèle exemple : matrice de confusion et courbe ROC |
| `frequence_variable.png` | fréquence de sélection de chaque variable sur les 15 modèles |
| `lr_confusion_matrix.png`, `lr_roc.png` | régression logistique, modèle exemple : matrice de confusion et courbe ROC |

## Données

Le dossier `data/` contient les fichiers fournis, sans modification : `wdbc.csv`,
`MANIFEST.json` et `DATA_DICTIONARY.md`.

## Références

- Golub, T. R. et al. (1999). Molecular classification of cancer: class discovery and class
  prediction by gene expression monitoring. *Science*, 286(5439), 531–537.
  doi:10.1126/science.286.5439.531
- Wolberg, W., Mangasarian, O., Street, N., Street, W. (1995). Breast Cancer Wisconsin (Diagnostic).
  UCI Machine Learning Repository. doi:10.24432/C5DW2B
- Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine
  Learning Research*, 12, 2825–2830. Utilisé pour la validation croisée, les métriques,
  `StandardScaler` et `LogisticRegression`.

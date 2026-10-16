import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt


def load(data_dir):
    """
    Chargement des données et séparation des 30 variables et du diagnostic
    """
    df = pd.read_csv(data_dir / "wdbc.csv")
    labels = (df["diagnosis"] == "malignant").astype(int).to_numpy()
    features = df.drop(columns=["sample_id", "diagnosis"])
    return features, labels


def make_model_split(features, labels, n_folds, n_repeats, seed):
    """
    Crée les 15 splits avec la méthode RepeatedStratifiedKFold
    list() garde les 15 splits pour pouvoir les utiliser avec WeightedVoting et la régression logistique
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)
    return list(rskf.split(features, labels))


def save_example_figures(y_test, scores, cm, auc, bacc, results_dir, name, prefix):
    """
    Enregistre la matrice de confusion et la courbe ROC du modèle exemple
    """

    # Matrice de confusion du modèle
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Bénin (0)", "Malin (1)"]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{prefix.upper()} modèle exemple - balanced acc. = {bacc:.3f}")
    fig.tight_layout()
    fig.savefig(results_dir / f"{prefix}_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Courbe ROC du modèle
    fig, ax = plt.subplots(figsize=(4, 4))
    RocCurveDisplay.from_predictions(y_test, scores, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"{prefix.upper()} modèle exemple - AUC = {auc:.3f}")
    fig.tight_layout()
    fig.savefig(results_dir / f"{prefix}_roc.png", dpi=300)
    plt.close(fig)

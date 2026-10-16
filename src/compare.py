import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import save_example_figures

METRICS = ["auc", "accuracy", "balanced_accuracy", "sensibilite"]


def logistic_reg_model():
    """
    Crée la régression logistique avec standardisation.
    Le pipeline apprend la standardisation sur l'entraînement seulement
    """
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)
    )


def evaluate_logistic_regression(features, labels, split_list, results):
    """
    Evaluation de la régression logistique sur les 15 splits
    """
    temp = []
    for i, (train_index, tests_index) in enumerate(split_list):
        X_train, X_test = features.iloc[train_index], features.iloc[tests_index]
        y_train, y_test = labels[train_index], labels[tests_index]

        model = logistic_reg_model().fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auc = roc_auc_score(y_test, proba)
        bacc = balanced_accuracy_score(y_test, y_pred)

        temp.append({
            "modele": "regression_logistique",
            "auc": auc,
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": bacc,
            "sensibilite": recall_score(y_test, y_pred),
            # Part des malins bien prédits
        })

        if i == 0:
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            save_example_figures(y_test, proba, cm, auc, bacc, results,
                                 name="Logistic regression", prefix="lr")

    return pd.DataFrame(temp)


def compare(res_wv, res_lr):
    """
    Affichage des performances moyennes des deux modèles
    """
    res = pd.concat([res_wv, res_lr])
    summary = res.groupby("modele")[METRICS].agg(["mean", "std"]).round(4)
    print("\nPartie 5 : comparaison des deux modèles sur les 15 mêmes splits")
    print(summary.to_string())

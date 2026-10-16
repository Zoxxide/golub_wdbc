import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, recall_score, roc_auc_score

from weight import WeightedVoting

from utils import save_example_figures


def evaluate_model(features, labels, split_list, results_dir):
    """
    Evaluation du vote pondéré sur les 15 splits
    """
    temp = []
    for i, (train_index, tests_index) in enumerate(split_list):
        X_train, X_test = features.iloc[train_index], features.iloc[tests_index]
        y_train, y_test = labels[train_index], labels[tests_index]

        model = WeightedVoting().fit(X_train, y_train)

        scores = model.decision_function(X_test)
        y_pred = model.predict(X_test)

        auc = roc_auc_score(y_test, scores)
        acc = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        sens = recall_score(y_test, y_pred)
        # sensibilité : part des malins bien prédits

        temp.append({
            "modele": "vote_pondere",
            "auc": auc,
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "sensibilite": sens,
            "selected": model.features_,
        })

        if i == 0:
            print("\nPartie 3 : vote pondéré, modèle exemple (repeat 1, fold 1)")
            print(f"AUC : {auc:.3f}")
            print(f"Accuracy : {acc:.3f}")
            print(f"Balanced accuracy : {bacc:.3f}")
            print(f"Sensibilité : {sens:.3f}")
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            save_example_figures(y_test, scores, cm, auc, bacc, results_dir, name="Weighted voting", prefix="wv")

    res = pd.DataFrame(temp)

    print(f"\nPartie 3 : vote pondéré, moyenne et écart type sur {len(res)} modèles")
    print(f"AUC : {res.auc.mean():.3f} (écart type {res.auc.std():.3f})")
    print(f"Accuracy : {res.accuracy.mean():.3f} (écart type {res.accuracy.std():.3f})")
    print(f"Balanced accuracy : {res.balanced_accuracy.mean():.3f} "
          f"(écart type {res.balanced_accuracy.std():.3f})")
    print(f"Sensibilité : {res.sensibilite.mean():.3f} (écart type {res.sensibilite.std():.3f})")

    return res

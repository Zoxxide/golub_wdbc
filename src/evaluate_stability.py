import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations


def calc_freq(select_feat, features):
    """
    Compte le nombre de fois où chaque variable a été sélectionnée
    """
    counts = pd.Series(0, index=features)
    for s in select_feat:
        counts[s] += 1
    return counts.sort_values(ascending=False)


def evaluate_stability(model_val, features, results):
    """
    Affichage de la stabilité de la sélection et de la fréquence de sélection
    """
    select_features = model_val.selected.tolist()
    n = len(select_features)
    counts = calc_freq(select_features, features)

    jac = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in combinations(select_features, 2)]
    mean_jac = np.mean(jac)
    
    print("\nPartie 4 : stabilité de la sélection")
    print(f"Jaccard moyen sur les {n * (n - 1) // 2} paires : {mean_jac:.3f}")
    print(f"Variables choisies dans les {n} modèles : {(counts == n).sum()}")
    print("Fréquence de sélection :")
    print((counts[counts > 0] / n).round(3).to_string())
    print(f"Écart type de l'AUC : {model_val.auc.std():.3f}, "
          f"de la balanced accuracy : {model_val.balanced_accuracy.std():.3f}")

    fig, ax = plt.subplots(figsize=(6, 7))
    counts[::-1].plot.barh(ax=ax)
    ax.set_xlabel(f"Nombre de sélections (sur {n} modèles)")
    ax.set_title(f"Fréquence de sélection - Jaccard moyen = {mean_jac:.2f}")
    fig.tight_layout()
    fig.savefig(results / "frequence_variable.png", dpi=300)
    plt.close(fig)

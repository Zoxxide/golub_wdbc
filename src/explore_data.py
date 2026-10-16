import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def explore_data(features, labels, results_dir):
    """
    Exploration des données
    """
    print("Informations sur les données :")

    n_benin, n_malin = (labels == 0).sum(), (labels == 1).sum()

    print(f"Nombre d'observations (tumeurs) : {features.shape[0]}")
    print(f"Nombre de variables explicatives : {len(features.columns)}")
    print(f"Valeurs manquantes : {features.isna().sum().sum()}")
    print(f"Lignes dupliquées : {features.duplicated().sum()}")
    print(f"Diagnostic : {n_benin} bénins ({n_benin / len(labels):.1%}), "
          f"{n_malin} malins ({n_malin / len(labels):.1%})")

    # Figure Répartition des diagnostics
    fig, ax = plt.subplots(figsize=(5, 4))
    barres = ax.bar(["Bénin", "Malin"], [n_benin, n_malin])
    ax.bar_label(barres, padding=3)
    ax.set_ylim(0, 500)
    ax.set_title("Répartition du diagnostic")
    fig.tight_layout()
    fig.savefig(results_dir / "repartition_diagnostic.png", dpi=130)
    plt.close(fig)

    # Figure AUC de chaque variable
    # AUC de 0.5 : aucun lien avec le diagnostic
    # AUC au-dessus de 0.5 : valeurs plus grandes chez les malins
    # AUC en dessous de 0.5 : valeurs plus petites chez les malins
    aucs = features.apply(lambda col: roc_auc_score(labels, col))

    fig, ax = plt.subplots(figsize=(8, 10))
    barres = ax.barh(aucs.index, aucs.values - 0.5, left=0.5)
    ax.bar_label(barres, labels=[f"{a:.2f}" for a in aucs.values], padding=3, fontsize=8)
    ax.axvline(0.5, ls="--", c="grey", lw=1)
    ax.text(0.505, -1.2, "aucun lien (0,5)", ha="left", fontsize=8, color="grey")
    ax.set_xlim(0.4, 1.03)
    ax.set_ylim(len(aucs) - 0.5, -1.8)
    ax.set_xlabel("AUC (diagnostic malin)")
    ax.set_title("Lien de chaque variable avec le diagnostic")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(results_dir / "auc_variables.png", dpi=130)
    plt.close(fig)

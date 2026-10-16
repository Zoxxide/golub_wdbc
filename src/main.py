from pathlib import Path

from utils import load, make_model_split
from explore_data import explore_data
from evaluate_model import evaluate_model
from evaluate_stability import evaluate_stability
from compare import evaluate_logistic_regression, compare

# Paramètres
SEED = 12

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS = ROOT / "results"

N_REPEATS = 3
N_FOLDS = 5


def main():
    RESULTS.mkdir(exist_ok=True)
    print(f"Seed utilisée : {SEED}")
    features, labels = load(DATA_DIR)

    # Partie 1
    explore_data(features, labels, RESULTS)

    # Génération des 15 splits utilisés pour les parties 3 et 5
    split_list = make_model_split(features, labels, N_FOLDS, N_REPEATS, SEED)

    # Parties 2 et 3
    res_weight_voting = evaluate_model(features, labels, split_list, RESULTS)

    # Partie 4
    evaluate_stability(res_weight_voting, features.columns, RESULTS)

    # Partie 5
    res_logistic_reg = evaluate_logistic_regression(features, labels, split_list, RESULTS)
    compare(res_weight_voting, res_logistic_reg)

    print(f"\nFigures dans le dossier {RESULTS.name}")


if __name__ == "__main__":
    main()

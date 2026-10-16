class WeightedVoting:
    def __init__(self, n_features=10):
        self.n_features = n_features

    def fit(self, features, labels):
        """
        Fonction d'entraînement du modèle
        Tout est calculé sur le fold d'entraînement uniquement
        """
        # Application de la formule
        mu1, mu0 = features[labels == 1].mean(), features[labels == 0].mean()
        sigma1, sigma0 = features[labels == 1].std(), features[labels == 0].std()
        w = (mu1 - mu0) / (sigma1 + sigma0)

        # On garde les 10 variables de plus grand |w_j|
        self.features_ = w.abs().sort_values(ascending=False).index[: self.n_features]
        self.w_ = w[self.features_]
        self.mid_ = ((mu1 + mu0) / 2)[self.features_]
        return self

    def decision_function(self, X):
        """
        Calcule le score V(x)
        """
        return ((X[self.features_] - self.mid_) * self.w_).sum(axis=1).to_numpy()

    def predict(self, X):
        """
        Fonction de prédiction du diagnostic : 1 pour malin et 0 pour bénin
        """
        return (self.decision_function(X) >= 0).astype(int)

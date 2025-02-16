import numpy as np


class PolynomialRegression:

    def __init__(self, degree=2, lrnRate=0.01, nIterations=100):
        """
        Initialisation de la classe de régression polynomiale.

        :param degree: Degré du polynôme
        :param lrnRate: Taux d'apprentissage
        :param nIterations: Nombre d'itérations pour l'entraînement
        """
        self.degree = degree
        self.lrnRate = lrnRate
        self.nIterations = nIterations
        self.weights = None
        self.bias = None

    def _transform_features(self, X):
        """
        Transformation des caractéristiques en puissances polynomiales.

        :param X: Matrice des caractéristiques d'entrée (n_samples, n_features)
        :return: Matrice transformée (n_samples, n_features * degree)
        """
        X_poly = X.copy()
        for _ in range(2, self.degree + 1):
            X_poly = np.hstack((X_poly, X ** _))

        return X_poly

    def fit(self, X, y):
        """
        Entraînement du modèle en utilisant la descente de gradient.

        :param X: Matrice des caractéristiques d'entrée (n_samples, n_features)
        :param y: Vecteur des cibles (n_samples,)
        """
        print("== Training start ==")

        # Transformation des donnees
        X_poly = self._transform_features(X)

        # Initialisation des parametres
        n_samples, n_features = X_poly.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Algorithme de la descente de Gradient
        for _ in range(self.nIterations):
            y_h = np.dot(X_poly, self.weights) + self.bias

            dw = (2/n_samples) * np.dot(X_poly.T, (y_h - y))
            db = (2/n_samples) * np.sum(y_h - y)

            self.weights -= self.lrnRate * dw
            self.bias -= self.lrnRate * db

            print("\r> Epoch [{}{}] {}/{}".format("=" * int(_ + 1 * 100 / self.nIterations), "-" * (100 - int(_ * 100 / self.nIterations)), _, self.nIterations), end="")

        print("== Training Complete ==")

    def predict(self, X):
        """
        Prédiction des valeurs cibles pour les données d'entrée.

        :param X: Matrice des caractéristiques d'entrée (n_samples, n_features)
        :return: Vecteur des prédictions (n_samples,)
        """
        X_poly = self._transform_features(X)
        y_pred = np.dot(X_poly, self.weights) + self.bias
        return y_pred

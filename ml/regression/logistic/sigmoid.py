import numpy as np

class LogisticRegression:

    def __init__(self, lrnRate= 0.01, nIterations= 1000):
        self.lrnRate = lrnRate              # Taux d'apprentissage
        self.nIterations = nIterations      # Nombre d'Iterations
        self.weights = None                 # Poids des caracteristiques
        self.bias = None                    # Biais

    def _sigmoid(self, t):
        """Calcul de la foncttion sigmoide"""
        return 1 / (1 + np.exp(-t))

    def fit(self, X, y):
        """
        Entrainement du modele

        :param X: Matrice des caracteristiques d'entree (n_samples, n_features)
        :param y: Vecteur des etiquettes (n_samples,)
        """
        print("== Training start ==")

        # Initialisation des parametres
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Algorithme de la descente de Gradient
        for _ in range(self.nIterations):
            y_h = X.dot(self.weights) + self.bias
            y_h = self._sigmoid(y_h) # Appliquer la fonction sigmoide au modele lineaire

            # Calcul des derivees
            dw = (1/n_samples) * np.dot(X.T, (y_h - y))
            db = (1/n_samples) * np.sum(y_h - y)

            # M.a.j des valeurs des parametres
            self.weights -= self.lrnRate * dw
            self.bias -= self.lrnRate * db

            print("\r> Epoch [{}{}] {}/{}".format("=" * int(_ + 1 * 100 / self.nIterations),"-"  * (100 - int(_  * 100 /self.nIterations)), _, self.nIterations ), end=" ")

        print("== Training complete ==")


    def predict(self, X):
        """
        Predit les etiquettes
        
        :param X: Matrice des caracteristiques
        :return: Predictions (0 ou 1)
        """
        y_pred = X.dot(self.weights) + self.bias
        y_pred = self._sigmoid(y_pred)

        class_pred = [0 if y <=0.5 else 1 for y in y_pred]

        return class_pred
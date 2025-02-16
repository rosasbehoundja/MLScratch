import numpy as np

class SoftmaxRegression:
    """
    Softmax Regression or Multinomial Logistic Regression is a generalized
    version of the Logistic Regression function that support multiples classes
    directly without having to train and combine multiple binary classifiers.
    """

    def __init__(self, lrnRate= 0.01, nIterations= 1000):
        self.lrnRate = lrnRate                  # Seuil d'apprentissage
        self.nIterations = nIterations          # Nombre d'iterations
        self.weights = None                     # Poids des caracteristiques
        self.bias = None                        # Terme de biais
        self.classes_ = None                    # Classes

    def _softmax(self, z):
        # Reduire les valeurs pour ameliorer la stabilite numerique
        shifted_z = z - np.max(z, axis= 1, keepdims=True)
        exp_scores = np.exp(shifted_z)
        # Calcul de la fonction Softmax
        return exp_scores / np.sum(exp_scores, axis= 1, keepdims=True)
    
    def fit(self,X, y):
        # Initialiser les parametres
        n_samples, n_features = X.shape

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros((1, n_classes))

        # Encodage des etiquettes
        y_one_hot = np.zeros((n_samples, n_classes))
        y_one_hot[np.arange(n_samples), y] = 1

        # Descente de Gradient
        for _ in range(self.nIterations):

            z = np.dot(X, self.weights) + self.bias
            y_pred = self._softmax(z)

            error = y_pred - y_one_hot

            # Calcul des derivees
            dw = (1/n_samples) * np.dot(X.T, error)
            db = (1/n_samples) * np.sum(error, axis=0, keepdims=True)

            # M.a.j des parametres
            self.weights -= self.lrnRate * dw
            self.bias -= self.lrnRate * db

            print("\r> Epoch [{}{}] {}/{}".format("=" * int(_ + 1 * 100 / self.nIterations),"-"  * (100 - int(_  * 100 /self.nIterations)), _, self.nIterations ), end="")

        print("== Training Complete ==")
    
    def predict_proba(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return self._softmax(y_pred)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis= 1)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
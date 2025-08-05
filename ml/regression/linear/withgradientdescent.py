import numpy as np

class LinearRegressionWithGD:

    def __init__(self, lrnRate = 0.01, nIterations = 100):
        self.lrnRate = lrnRate             # Seuil d'apprentissage
        self.nIterations = nIterations     # Nombre d'iterations
        self.weights = None                # Poids wi 
        self.bias = None                   # Biais beta


    def fit(self, X, y):
        """Training Function"""
        print("== Training Initialisation ==")
        # initialisation des parametres
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Algorithme de la descente de Gradient
        for _ in range(self.nIterations):
            y_h = np.dot(X, self.weights) + self.bias

            # calcul des derivees partielles
            dw = (2/(n_samples)) * np.dot(X.T, (y_h - y))
            db = (2/(n_samples)) * np.sum(y_h - y)
            
            # ajustement des parametres
            self.weights -= self.lrnRate * dw
            self.bias -= self.lrnRate * db

            print("\r> Epoch [{}{}] {}/{}".format("=" * int(_ + 1 * 100 / self.nIterations),"-"  * (100 - int(_  * 100 /self.nIterations)), _, self.nIterations ), end="")
        
        print("== Training Complete ==")


    def predict(self, X):
        """
        Make predictions

        :param X: numpy array
        """
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred


    def error(self, y, y_pred):
        err = np.mean((y - y_pred)**2) * 100
        return err
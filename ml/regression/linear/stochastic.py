import numpy as np

class LinearRegressionWithSGD:

    def __init__(self, lrnRate= 0.01, nIterations= 1000):
        self.lrnRate = lrnRate
        self.nIterations = nIterations
        self.weights = None
        self.bias = None
        

    def fit(self, X, y):
        """Training Function using Stochastic Gradient descent"""

        # Initialisation des parametres
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Algorithme de la descente de gradient stochastique
        for _ in range(self.nIterations):
            for j in range(n_samples):
                # Select random sample
                idx = np.random.randint(0, n_samples)
                X_i = X[idx, :].reshape(1, -1)
                y_i = y[idx]

                # Prediction for the random sample
                y_h = np.dot(X_i, self.weights) + self.bias

                # partial derivatives of weights and bias
                dw = 2 * np.dot(X_i.T, (y_h - y_i))
                db = 2 * (y_h - y_i)

                # update the weights and bias values
                self.weights -= self.lrnRate * dw.flatten()
                self.bias -= self.lrnRate * db

            print("\r> Epoch [{}{}] {}/{}".format("=" * int(_ + 1 * 100 / self.nIterations),"-"  * (100 - int(_  * 100 /self.nIterations)), _, self.nIterations ), end="")


    def predict(self, X):
        """
        Prediction function

        :param X: np.array
        """
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
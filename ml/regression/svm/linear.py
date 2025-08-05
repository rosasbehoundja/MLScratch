import numpy as np

class LinearSVM:

    def __init__(self, c= 0.01, lrnRate= 0.01, nIterations= 1000):
        self.c = c                              # coefficient de regularisation
        self.lrnRate = lrnRate
        self.nIterations = nIterations
        self.weights = None
        self.bias = None

    def _hinge_loss(self, X, y):
        """
        Compute hinge loss for SVM
        """
        decisions = y * (np.dot(X, self.weights) + self.bias)
        hinge = np.maximum(0, 1 - decisions)
        return np.sum(hinge)

    def fit(self,X, y):
        print("== Training Start ==")
        # Initialisation des parametres
        y = np.where(y <=0, -1, 1)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        # Descente de Gradient
        for _ in range(self.nIterations):
            for idx, xi in enumerate(X):
                # Hinge Loss
                condition = y[idx] * (np.dot(xi, self.weights) + self.bias) >= 1
                if condition:
                    # Dans la marge alors pas de penalite
                    dw = 2 * self.c * self.weights
                    db = 0
                else:
                    # Violation de la marge
                    dw = 2 * self.c * self.weights - np.dot(xi, y[idx])
                    db = -y[idx]

                # M.a.j des parametres
                self.weights -= self.lrnRate * dw
                self.bias -= self.lrnRate * db

            current_loss = self._hinge_loss(X, y)
            print("\r> Epoch [{}{}] {}/{}".format("=" * int(_ + 1 * 100 / self.nIterations),"-"  * (100 - int(_  * 100 /self.nIterations)), _, self.nIterations ), end="")

        print("== Training Complete ==")
    
    def predict(self, X):
        y_pred = np.dot(X, self.weights) - self.bias
        return np.sign(y_pred)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
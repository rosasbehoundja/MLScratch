import numpy as np
import logging

class KernelSVM:

    def __init__(self, C= 1.0, lrnRate= 0.01, gamma= 0.5, nIterations= 1000):
        self.C = C                          # lambda parameter
        self.lrnRate = lrnRate              # seuil d'apprentissage
        self.nIterations = nIterations      # nombre d'iterations
        self.gamma = gamma                  # parametre du kernel rbf
        self.alpha = None                   # multiplicateur de lagrange
        self.bias = None                    # biais
        self.support_vectors = []           # matrice des vecteurs des supports
        self.support_labels = []            # matrice des etiquettes des supports

    def _rbf_kernel(self, x1, x2):
        """
        Calculer le kernel rbf entre deux points x1 et x2
        """
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2)**2)

    def _compute_kernel(self,X):
        """
        Determiner la matrice du kernel rbf de X
        """
        n_samples, _ = X.shape
        K = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(n_samples):
                K[i][j] = self._rbf_kernel(X[i], X[j])
        return K

    def fit(self, X, y):
        """
        Entrainement du modele avec le kernel rbf
        """
        print("== Training Start ==")
        
        y = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        self.bias = 0

        K = self._compute_kernel(X)

        # Algorithme de la descent de gradient
        for _ in range(self.nIterations):
            for idx in range(n_samples):
                margin = y[idx] * (np.sum(self.alpha * y * K[:, idx]) + self.bias)
                if margin < 1:
                    # Ajustement car violation de la marge
                    self.alpha[idx] += self.lrnRate * (1 - margin) * y[idx]
                    self.alpha[idx] = max(0, min(self.alpha[idx], self.C))
                else:
                    # Pas de violtion de la marge
                    self.alpha[idx] -= self.lrnRate * 2 * self.alpha[idx]

            # Mise a jour du biais
            self.bias = np.mean(y - np.sum(self.alpha * y * K, axis=0))
            print("\r> Epoch [{}{}] {}/{}".format("=" * int(_ + 1 * 100 / self.nIterations),"-"  * (100 - int(_  * 100 /self.nIterations)), _, self.nIterations ), end="")

        # Identification des supports
        idx = self.alpha > 1e-5
        self.support_vectors = X[idx]
        self.support_labels = y[idx]
        self.alpha = self.alpha[idx]

        print("== Training complete ==")

    def predict(self, X):
        """
        Predire les classes de nouvelles donnees
        """
        y_pred = []
        for x in X:
            decisions = np.sum(self.alpha * self.support_labels 
                            * np.array([self._rbf_kernel(x, sv) for sv in self.support_vectors])) + self.bias
            y_pred.append(np.sign(decisions))

        return np.array(y_pred)


    def score(self, y_pred, y_test):
        """
        Calcul de la precision
        """
        y_test = np.where(y_test<=0, -1 , 1)
        return np.mean(y_pred == y_test)
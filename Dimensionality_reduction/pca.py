import numpy as np
from sklearn.decomposition import PCA as skPCA

"""
Analyse des composantes principales ou Principal Components Analysis en anglais.

1. Determiner la moyenne de la matrice X par colonne
2. Determiner la matrice centrée X_centred
3. Determiner la matrice de covariance X_cov
4. Determiner les vecteurs et valeurs propres de la matrice de covariance
5. Ordonner par ordre decroissant les valeurs propres
6. Ordonner les vecteurs propres en fonction des index des valeurs propres associées
7. Recuperer les k  meilleures composantes
8. Faire la projection sur la matrice centrée
9. Recuperer la matrice reduite
"""

class PrincipalComponentAnalysis:

    def __init__(self, k):
        self.k = k
        self.mu = None # moyenne
        self.components = None # k meilleures composantes

    def fit(self, X):
        # determiner la moyenne
        self.mu = np.mean(X, axis=0)

        # determiner la matrice centrée
        X_centred = X - self.mu

        # determiner la matrice de covariance
        X_cov = np.cov(X_centred.T)

        # determiner les valeurs propres et les vecteurs propres
        eigvalues, eigvect = np.linalg.eig(X_cov)

        # trier les valeurs propres par ordre decroissant
        indices = np.argsort(eigvalues)[::-1]

        # recuperer les k meilleures composantes
        eigvalues = eigvalues[indices]
        V = eigvect[:, indices]
        self.components = V[:, :self.k]
    
    def transform(self, X):
        X_centred = X - self.mu
        return X_centred @ self.components


if __name__ == "__main__":
    pca = PrincipalComponentAnalysis(k = 1)
    X = np.array([[2, 3], [3, 4], [4, 5], [5, 6]])
    pca.fit(X)
    X_reduit = pca.transform(X)
    print("Matrice reduite: \n", X_reduit)
    
    sk = skPCA(n_components=1)
    sk.fit(X)
    X_sk = sk.transform(X)
    print("Matrice Sklearn: \n", X_sk)
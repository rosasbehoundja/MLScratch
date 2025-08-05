import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import scipy as sp

# Sklearn
from sklearn.manifold import LocallyLinearEmbedding as LSklearn

class LocallyLinearEmbedding:

    def __init__(self, n_neighbors = 10, n_components = 2):
        self.n_neighbors = n_neighbors
        self.n_components = n_components

    def fit(self, X):
        # Standardisation des donnees
        X = StandardScaler().fit_transform(X)

        n_samples, n_features = X.shape

        # Ensure n_neighbors is less than or equal to n_features
        if self.n_components > n_features:
            raise ValueError("n_components must be less than or equal to the number of features")

        if self.n_neighbors >= n_samples:
            raise ValueError("n_neighbors must be less than or equal to the number of samples")

        # Initialisation de la matrice de reconstruction
        W = np.zeros((n_samples, n_samples))

        # Determination des plus proches voisins
        neighbors  = NearestNeighbors(n_neighbors= self.n_neighbors + 1)
        neighbors.fit(X)
        distances, idx = neighbors.kneighbors(X, return_distance=True)

        for i in range(n_samples):
            # Selection des k plus proches voisins
            voisins = idx[i, 1:]

            # Sous-matrice des voisins
            Z = X[voisins, :] - X[i, :]

            # Matrice de Gramm
            C = Z @ Z.T
            
            # Régularisation pour stabilité numérique
            reg = 1e-3 * np.trace(C) / len(C)
            C.flat[::len(C) + 1] += reg  # Add to diagonal
            
            try:
                # Calcul des poids w qui minimisent l'erreur
                w = np.linalg.solve(C, np.ones(self.n_neighbors))
            except np.linalg.LinAlgError:
                # Fallback si la matrice est singulière
                w = np.linalg.lstsq(C, np.ones(self.n_neighbors), rcond=None)[0]
                
            w /= np.sum(w)
            W[i, voisins] = w

        # Construction de la matrice de reconstruction M = (I-W)^T(I-W)
        I = np.eye(n_samples)
        M = (I - W).T @ (I - W)
        
        # Make sure M is symmetric
        M = (M + M.T) / 2
        
        # Extraction des meilleures composantes
        # Pour LLE, on cherche les vecteurs propres associés aux plus petites valeurs propres non nulles
        eigval, eigvect = np.linalg.eigh(M)
        
        # Skip the smallest eigenvalue (should be close to zero) and take the next n_components
        idx = np.argsort(eigval)[1:self.n_components+1]
        
        # Sort eigenvectors by eigenvalue
        Y = eigvect[:, idx]
        
        # Normalize the embedding vectors
        Y = Y * np.sqrt(n_samples)
        
        self.embedding_ = Y
        return self
    
    def fit_transform(self, X):
        self.fit(X)
        return self.embedding_
    
if __name__ == "__main__":
    # Démonstration
    def demo_lle():
        import matplotlib.pyplot as plt
        
        # Génération du Swiss Roll
        np.random.seed(42)
        t = np.pi * (1 + 2 * np.random.rand(1000))
        height = 21 * np.random.rand(1000)
        X = np.zeros((1000, 3))
        X[:, 0] = t * np.cos(t)
        X[:, 1] = height
        X[:, 2] = t * np.sin(t)

        # Application de LLE
        lle = LocallyLinearEmbedding(n_neighbors=12)
        X_lle = lle.fit_transform(X)

        lsk = LSklearn(n_neighbors=12, n_components=2)
        X_sk = lsk.fit_transform(X)
        
        # Visualisation
        plt.figure(figsize=(12, 5))
        
        plt.subplot(131)
        plt.scatter(X[:, 0], X[:, 2], c=t, cmap='viridis')
        plt.title('Données Originales')
        
        plt.subplot(132)
        plt.scatter(X_lle[:, 0], X_lle[:, 1], c=t, cmap='viridis')
        plt.title('Données après LLE')

        plt.subplot(133)
        plt.scatter(X_sk[:, 0], X_sk[:, 1], c=t, cmap='viridis')
        plt.title('Données Sklearn')

        plt.tight_layout()
        plt.show()

    # Exécution de la démonstration
    demo_lle()
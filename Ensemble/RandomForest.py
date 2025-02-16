import numpy as np
from DecisionTrees.DecisionTrees import DecisionTree
from collections import Counter

class RandomForest:

    def __init__(self, n_trees=10, max_depth= 10, min_samples_split= 2, n_features= None):
        """
        Initialisation des parametres

        :param n_trees: Nombre d'arbres de decision
        :param max_depth: Profondeur maximale des arbres
        :param min_samples_split: Nombre minimum d'echantillons pour splitter
        :param n_features: Nombre de caracteristiques
        :param trees: Liste des arbres de decision
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split= min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        """
        Entrainement du modele

        :param X: Matrice des echanitllons de donnees
        :param y: Vecteur des etiquettes        
        """
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split, 
                        max_depth= self.max_depth, n_features= self.n_features)
            
            X_sample, y_sample = self._boostrap_samples(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _boostrap_samples(self, X, y):
        """
        Effectue l'echantillonnage aleatoire des donnees
        """
        n_samples = X.shape[0]
        idx = np.random.choice(n_samples, n_samples, replace=True)
        return X[idx], y[idx]
    
    def _most_common_label(self, y):
        """
        Recupere la classe la plus frequente pour chaque y_i de y
        """
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common

    def predict(self, X):
        """
        Effectue des predictions sur les valeurs du vecteur y
        """
        prediction = np.array([tree.predict(X) for tree in self.trees])
        prediction = np.swapaxes(prediction, 0, 1)
        prediction = np.array([self._most_common_label(pred) for pred in prediction]) # VotingClassfier
        return prediction

    def score(self, y_true, y_pred):
        return np.mean(y_true == y_pred) * 100
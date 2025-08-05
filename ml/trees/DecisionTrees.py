import numpy as np
from collections import Counter

class Node:
    
    def __init__(self, feature= None, threshold= None,left= None, right= None, *, value=None):
        """
        Noeud de l'arbre de decision

        :param feature: Caracteristique
        :param threshold: Seuil de splitting
        :param left: Descendance a gauche
        :param right: Descendance a droit
        :param value: Valeur du noeud      
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """
        Informe si le noeud est le dernier noeud de l'arbre
        """
        return self.value is not None
    

class DecisionTree:

    def __init__(self, min_samples_split= 2, max_depth= 100, n_features = None):
        """
        Initialisation des parametres

        :param min_samples_split: Nbre mininum d'echantillon avant de splitter
        :param max_depth: Profondeur maximale de l'arbre de decision
        :param n_features: Nombre de caracteristiques
        :param root: Racine de l'arbre de decision
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        """
        Entrainement du modele
        """
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth= 0):
        """
        Construction de l'arbre de decision

        :param X: Matrice des echantillons de donnees
        :param y: Vecteur des etiquettes
        """
        n_samples, n_features = X.shape
        n_labels = np.unique(y)

        # Verifier le critere d'arret
        if (depth>=self.max_depth or len(n_labels) == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idx = np.random.choice(n_features, self.n_features, replace=False)

        # Trouver la meilleure separation
        best_features, best_threshold= self._best_split(X, y, feat_idx)

        # Trouver les enfants (noeuds)
        left_idx, right_idx = self._split(X[:, best_features], best_threshold)
        left = self._grow_tree(X[left_idx, :], y[left_idx], depth+1)
        right = self._grow_tree(X[right_idx, :], y[right_idx], depth+1)
        return Node(best_features, best_threshold, left, right)

    def _best_split(self, X, y, idx):
        """
        Recuperer les meilleures features et le meilleur seuil de splitting

        :param X: Matrice des echantillons de donnees
        :param y: Vecteur des etiquettes
        """
        best_gain = -1
        split_idx, split_threshold = None, None

        for i in idx:
            X_column = X[:, i]
            thresholds = np.unique(X_column)
            for thr in thresholds:
                # Calculer le gain d'informations
                gain = self._information_gain(X_column, y, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = i
                    split_threshold = thr
        return split_idx, split_threshold

    def _entropy(self, y):
        """
        Calcule de l'entropy
        """
        hist = np.bincount(y)
        probability = hist/len(y)
        return -np.sum([p * np.log(p) for p in probability if p>0])

    def _split(self, X_column, threshold):
        """
        Separer les donnees du vecteur X_column en deux groupes
        en respectant un parametre threshold et recuperer les index
        des colonnes correspondantes
        """
        left_idx = np.argwhere(X_column<=threshold).flatten()
        rigth_idx = np.argwhere(X_column >= threshold).flatten()
        return left_idx, rigth_idx

    def _information_gain(self, X_column, y, threshold):
        """
        Retourne le gain d'information resultant d'un splitting

        `gain = parentGain - ChildrenGain`

        :param X_column: Vecteur de la caracteristique i
        :param y: Vecteur des etiquettes associes
        :param threshold: Seuil de splitting
        :return: gain d'information
        """
        # Parent entropy
        parent_entropy = self._entropy(y)

        # Create children
        left_idx, right_idx = self._split(X_column, threshold)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0
        
        # Calculer l'entropy des enfants
        nbr = len(y)
        nbr_l, nbr_r = len(left_idx), len(right_idx)
        entropy_left, entropy_right = self._entropy(y[left_idx]), self._entropy(y[right_idx])

        child_entropy = (nbr_l/nbr) * entropy_left + (nbr_r/nbr) * entropy_right

        # Information gain
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _most_common_label(self, y):
        """
        Retourne la classe la plus frequente pour chaque y_i du vecteur y
        """
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        Traverse l'arbre jusqu'aux feuilles et recupere la valeur predite
        """
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def score(self, y, y_pred):
        return np.mean(y == y_pred)
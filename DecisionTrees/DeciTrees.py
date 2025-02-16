import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        """
        Represents a node in the decision tree.
        
        Parameters:
        - feature: Index of the feature used for splitting at this node.
        - threshold: Threshold value for the split.
        - left: Left child node.
        - right: Right child node.
        - value: Class label if the node is a leaf.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Leaf node value

    def is_leaf_node(self):
        """Checks if the node is a leaf node."""
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        """
        Decision Tree Classifier.

        Parameters:
        - min_samples_split: Minimum number of samples required to split a node.
        - max_depth: Maximum depth of the tree.
        - n_features: Number of features to consider when looking for the best split.
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        """
        Builds the decision tree from the training data.

        Parameters:
        - X: Feature matrix (numpy array).
        - y: Target vector (numpy array).
        """
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _entropy(self, y):
        """
        Computes the entropy of a label distribution.

        Parameters:
        - y: Target vector (numpy array).

        Returns:
        - Entropy value.
        """
        hist = np.bincount(y)
        probabilities = hist / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    def _split(self, X_column, threshold):
        """
        Splits the data into two groups based on a threshold.

        Parameters:
        - X_column: Feature column (numpy array).
        - threshold: Threshold value for splitting.

        Returns:
        - Indices of the left and right splits.
        """
        left_idx = np.argwhere(X_column <= threshold).flatten()
        right_idx = np.argwhere(X_column > threshold).flatten()
        return left_idx, right_idx

    def _information_gain(self, X_column, y, threshold):
        """
        Calculates the information gain of a split.

        Parameters:
        - X_column: Feature column (numpy array).
        - y: Target vector (numpy array).
        - threshold: Threshold value for splitting.

        Returns:
        - Information gain value.
        """
        parent_entropy = self._entropy(y)

        # Split data
        left_idx, right_idx = self._split(X_column, threshold)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0

        # Weighted average of child entropies
        n = len(y)
        n_left, n_right = len(left_idx), len(right_idx)
        entropy_left = self._entropy(y[left_idx])
        entropy_right = self._entropy(y[right_idx])
        child_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right

        # Calculate information gain
        return parent_entropy - child_entropy

    def _most_common_label(self, y):
        """
        Finds the most common label in the target vector.

        Parameters:
        - y: Target vector (numpy array).

        Returns:
        - Most common label.
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _best_split(self, X, y, feature_indices):
        """
        Finds the best feature and threshold for splitting.

        Parameters:
        - X: Feature matrix (numpy array).
        - y: Target vector (numpy array).
        - feature_indices: Indices of features to consider.

        Returns:
        - Best feature index and threshold value.
        """
        best_gain = -1
        split_idx, split_threshold = None, None

        for feature_idx in feature_indices:
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(X_column, y, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively grows the decision tree.

        Parameters:
        - X: Feature matrix (numpy array).
        - y: Target vector (numpy array).
        - depth: Current depth of the tree.

        Returns:
        - Root node of the subtree.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Randomly select features
        feature_indices = np.random.choice(n_features, self.n_features, replace=False)

        # Find the best split
        best_feature, best_threshold = self._best_split(X, y, feature_indices)

        # Split data
        left_idx, right_idx = self._split(X[:, best_feature], best_threshold)
        left_child = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def predict(self, X):
        """
        Predicts class labels for input samples.

        Parameters:
        - X: Feature matrix (numpy array).

        Returns:
        - Predicted class labels (numpy array).
        """
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        Traverses the tree to predict the class label for a single sample.

        Parameters:
        - x: Input sample (numpy array).
        - node: Current node in the tree.

        Returns:
        - Predicted class label.
        """
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def score(self, y_true, y_pred):
        """
        Computes the accuracy of the predictions.

        Parameters:
        - y_true: True class labels (numpy array).
        - y_pred: Predicted class labels (numpy array).

        Returns:
        - Accuracy score.
        """
        return np.mean(y_true == y_pred)
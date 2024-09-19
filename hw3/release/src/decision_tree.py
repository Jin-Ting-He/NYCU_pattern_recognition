"""
You dont have to follow the stucture of the sample code.
However, you should checkout if your class/function meet the requirements.
"""
import numpy as np


class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth
        self.tree = None
        self.feature_importances_ = None

    def fit(self, X, y):
        self.feature_importances_ = np.zeros(X.shape[1])
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or depth == self.max_depth:
            return np.bincount(y).argmax()
        
        feature_index, threshold, gain, is_discrete = find_best_split(X, y)
        if feature_index is None:
            return np.bincount(y).argmax()

        # Split dataset
        indices_left, indices_right = split_dataset(X, feature_index, threshold, is_discrete)
        left = self._grow_tree(X[indices_left], y[indices_left], depth + 1)
        right = self._grow_tree(X[indices_right], y[indices_right], depth + 1)
        
        # Calculate importance
        impurity = entropy(y)
        left_impurity = entropy(y[indices_left])
        right_impurity = entropy(y[indices_right])
        n = len(y)
        n_left = len(indices_left)
        n_right = len(indices_right)
        weighted_impurity_reduction = impurity - (n_left / n) * left_impurity - (n_right / n) * right_impurity
        self.feature_importances_[feature_index] += abs(weighted_impurity_reduction)

        return {'feature_index': feature_index, 'threshold': threshold, 'left': left, 'right': right}

    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree_node):
        if not isinstance(tree_node, dict):
            return tree_node
        if x[tree_node['feature_index']] < tree_node['threshold']:
            return self._predict_tree(x, tree_node['left'])
        else:
            return self._predict_tree(x, tree_node['right'])

    def compute_feature_importance(self):
        # Normalize feature importance
        total_importance = np.sum(self.feature_importances_)
        if total_importance > 0:
            return self.feature_importances_ / total_importance
        else:
            return self.feature_importances_

def split_dataset(X, feature_index, threshold, is_discrete):
    if is_discrete:
        indices_left = X[:, feature_index] == threshold
        indices_right = X[:, feature_index] != threshold
    else:
        indices_left = X[:, feature_index] < threshold
        indices_right = X[:, feature_index] >= threshold
    return indices_left, indices_right

def find_best_split(X, y):
    best_feature, best_threshold = None, None
    is_discrete = False
    best_gain = -np.inf
    for feature_index in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_index])
        if len(thresholds) <= 1:
            is_discrete = True
        else:
            is_discrete = False
        for threshold in thresholds:
            indices_left, indices_right = split_dataset(X, feature_index, threshold, is_discrete)
            left = y[indices_left]
            right = y[indices_right]
            if len(left) == 0 or len(right) == 0:
                continue
            gain = entropy(y) - (entropy(left) * len(left) / len(y) + entropy(right) * len(right) / len(y))
            if gain > best_gain:
                best_gain, best_feature, best_threshold = gain, feature_index, threshold
    return best_feature, best_threshold, best_gain, is_discrete

def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

def gini_index(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return 1 - np.sum(probabilities**2)


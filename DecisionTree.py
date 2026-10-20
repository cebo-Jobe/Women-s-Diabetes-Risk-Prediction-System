import numpy as np
from collections import Counter


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.importance = 0  # Track importance at each node

    def is_leaf_node(self): #Quickly check if this node makes a final prediction
        return self.value is not None


class DecisionTree:
    def __init__(self, min_sample_split=2, max_depth=100, n_features=None):
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        self.feature_importances_ = None # Feature importance scores

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.feature_importances_ = np.zeros(X.shape[1]) #Each position will accumulate importance for that feature[0,0,0,0,0,0,0,0,0,0]
        self.root = self.grow_tree(X, y)
        # Normalize feature importances
        self.feature_importances_ /= np.sum(self.feature_importances_)

    def grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or
                n_samples < self.min_sample_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        #Randomly selects a subset of features to consider for splitting at this node.
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feature is None:  # No split improved purity
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_idx, right_idx = self._split(X[:, best_feature], best_thresh)
        left = self.grow_tree(X[left_idx, :], y[left_idx], depth + 1)
        right = self.grow_tree(X[right_idx, :], y[right_idx], depth + 1)

        # Calculate node importance (weighted by samples)
        #Features that separate patients well (high information gain) get more importance
        node_importance = (len(y) * self._information_gain(y, X[:, best_feature], best_thresh))
        self.feature_importances_[best_feature] += node_importance

        return Node(best_feature, best_thresh, left, right)

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def _split(self, X_col, split_thresh):
        left_idxs = np.argwhere(X_col <= split_thresh).flatten()
        right_idxs = np.argwhere(X_col > split_thresh).flatten()
        return left_idxs, right_idxs

    #This method is the heart of the decision tree algorithm. It searches through all possible features
    #and thresholds to find the best possible split that maximizes information gain.
    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_col = X[:, feat_idx]
            thresholds = np.unique(X_col)

            for thr in thresholds:
                gain = self._information_gain(y, X_col, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, X_col, thr):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_col, thr)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        return parent_entropy - child_entropy

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def print_tree(self, node=None, depth=0, feature_names=None):
        """Print the decision tree in a readable format.

        Args:
            node: Current node to print (starts with root)
            depth: Current depth in the tree (for indentation)
            feature_names: Optional list of feature names for better readability
        """
        if node is None:
            node = self.root

        if node.is_leaf_node():
            print("  " * depth + f"Leaf: class {node.value}")
            return
        # If feature_names not given, use a generic label like 'feature_2'
        feature_name = f"feature_{node.feature}" if feature_names is None else feature_names[node.feature]
        print("  " * depth + f"[{feature_name} <= {node.threshold:.3f}]")

        # Print left subtree
        print("  " * (depth + 1) + "True:")
        self.print_tree(node.left, depth + 2, feature_names)

        # Print right subtree
        print("  " * (depth + 1) + "False:")
        self.print_tree(node.right, depth + 2, feature_names)
import numpy as np
from collections import Counter


class Node:
    # * : for attributes that are to be passed specifically with name else it won't be an attribute of the class object
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature  # which feature the node was divided with
        self.threshold = threshold  # which feature the node was divided with
        self.left = left  # left tree the node is pointing to
        self.right = right  # right tree the node is pointing to
        self.value = None  # whether leaf node or not

    def is_leaf_node(self):  # to check if leaf node or not
        return self.value is not None


class DecisionTree:
    def __init__(self, min_sample_split=2, max_depth=100, no_features=None):
        self.min_samples_split = min_sample_split
        self.max_depth = max_depth
        self.no_features = no_features
        self.root = None

    def fit(self, X, y):
        # shape gives number of samples and number of features so shape[1] gives number of features.
        self.no_features = (
            X.shape[1] if not self.no_features else min(X.shape[1], self.no_features)
        )
        # helper function _grow_tree : returns root node at the end
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):  # helper function
        no_samples, no_feats = X.shape
        no_labels = len(np.unique(y))

        # stopping criteria
        if (
            depth >= self.max_depth
            or no_labels == 1
            or no_samples <= self.min_samples_split
        ):
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(
            no_feats, self.no_features, replace=False
        )  # to get random unique features
        # find best split
        best_feature, best_threshold = self._best_split(
            X, y, feat_idxs
        )  # feat_idx includes features considered in creating the next best split

        # create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for (feat_idx,) in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thresh in thresholds:
                # calculate info gain
                gain = self._information_gain(X, y, thresh)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thresh

        return split_idx, split_threshold

    def _information_gain(self, X, y, X_column, threshold):
        # parent entropy
        p_entropy = self._entropy(y)

        # create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # calculate weighted average of entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        entropy_l, entropy_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        c_entropy = (n_l / n) * entropy_l + (n_r / n) * entropy_r

        # calculate IG
        information_gain = p_entropy - c_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()

        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        px = hist / len(y)  # p(x) = freq(x)/n

        return -np.sum([p * np.log(p) for p in px if p > 0])

    def most_common_labels(self, y):  # helper function
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

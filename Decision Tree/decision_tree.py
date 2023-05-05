

class Node:
    # * : for attributes that are to be passed specifically with name else it won't be an attribute of the class object
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature     # which feature the node was divided with
        self.threshold = threshold   # which feature the node was divided with
        self.left = left        # left tree the node is pointing to
        self.right = right       # right tree the node is pointing to
        self.value = None       # whether leaf node or not

    def is_leaf_node(self):   # to check if leaf node or not
        return self.value is not None


class DecisionTree:
    def __init__():

import pandas as pd

data = pd.read_csv('decision_data.csv')

for cols in data:
    for rows in data.index:
        print(data[cols][rows])


# EQUATIONS
# IG(information gain) = Eparent(Entropy of Parent) - [weighted avg]*Echildren
# E (Entropy) = sum(p(X)*log2(p(X)))

# STOPPING CRITERIA
# maximum depth , min no of samples,min impurity decrease

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = None       # whether leaf or not

    def is_leaf(self):
        return self.value is not None


class DecisionTree:

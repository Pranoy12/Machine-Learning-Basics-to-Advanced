import numpy as np


class LinearRegression:
    def __init__(self, learningrate=0.001, no_iterations=1000):
        self.learningrate = learningrate
        self.no_iterations = no_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):  # for training
        no_samples, no_features = X.shape
        self.weights = np.zeros(no_features)
        self.bias = 0

        for _ in range(self.no_iterations):
            # equation y=wx+b
            y_pred = np.dot(X, self.weights) + self.bias

            # calculating new weight and bias
            dw = (1/no_samples) * np.dot(X, (y_pred-y))
            db = (1/no_samples) * np.sum(y_pred-y)

            self.weights = self.weights - self.learningrate*dw
            self.bias = self.bias - self.learningrate*db

    def predict(self, X):  # for inference
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

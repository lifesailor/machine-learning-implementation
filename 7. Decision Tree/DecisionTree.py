from datetime import datetime
import numpy as np


def get_data():
    width = 8
    height = 8

    N = width * height

    X = np.zeros((N, 2))
    Y = np.zeros(N)
    n = 0

    start_t = 0

    for i in range(width):
        t = start_t
        for j in range(height):
            X[n] = [i, j]
            Y[n] = t

            n += 1
            t = (t + 1) % 2

        start_t = (start_t + 1) % 2

    return X, Y


def get_xor():
    X = np.zeros((200, 2))

    X[:50] = np.random.random((50, 2)) / 2 + 0.5  # (0.5 - 1, 0.5 - 1)
    X[50:100] = np.random.random((50, 2)) / 2  # (0 - 0.5, 0 - 0.5)
    X[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]])  # (0 - 0.5, 0.5 - 1)
    X[150:] = np.random.random((50, 2)) / 2 + np.array([[0.5, 0]])  # (0.5 - 1, 0 -0.5)

    Y = np.array([0] * 100 + [1] * 100)
    return X, Y


def get_donut():
    N = 200
    R_inner = 5
    R_outer = 10

    # distance from origin is radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)
    R1 = np.random.randn(N//2) + R_inner # gaussian noise
    theta = 2 * np.pi * np.random.random(N//2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N//2) + R_outer # gaussian noise
    theta = 2 * np.pi * np.random.random(N//2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([ X_inner, X_outer ])
    Y = np.array([0]*(N//2) + [1]*(N//2))
    return X, Y


def entropy(y):
    N = len(y)
    s1 = (y == 1).sum()

    if 0 == s1 or N == s1:
        return 0
    p1 = float(s1) / N
    p0 = 1 - p1
    return -p0 * np.log2(p0) - p1 * np.log2(p1)

class TreeNode:
    def __init__(self, depth=0, max_depth=None):
        self.depth = depth
        self.max_depth = max_depth

    def fit(self, X, Y):

        # base case 1. Y is just 1 or the same in the set.
        if len(Y) == 1 or len(set(Y)) == 1:
            self.col = None
            self.split = None
            self.left = None
            self.right = None
            self.prediction = Y[0]

        else:
            D = X.shape[1]
            cols = range(D)

            max_ig = 0
            best_col = None
            best_split = None

            for col in cols:
                ig, split = self.find_split(X, Y, col)
                if ig > max_ig:
                    max_ig = ig
                    best_col = col
                    best_split = split

            # base case 2. There is no more good split
            if max_ig == 0:
                self.col = None
                self.split = None
                self.left = None
                self.right = None
                self.prediction = np.round(Y.mean())
            else:
                self.col = best_col
                self.split = best_split

                # base case 3. max depth
                if self.depth == self.max_depth:
                    self.left = None
                    self.right = None
                    self.prediction = [
                        # majority class after splitting data
                        np.round(Y[X[:, best_col] < self.split].mean()),
                        np.round(Y[X[:, best_col] >= self.split].mean()),
                    ]
                # recursion
                else:
                    left_idx = (X[:, best_col] < best_split)
                    Xleft = X[left_idx]
                    Yleft = Y[left_idx]
                    self.left = TreeNode(self.depth + 1, self.max_depth)
                    self.left.fit(Xleft, Yleft)

                    right_idx = (X[:, best_col] >= best_split)
                    Xleft = X[right_idx]
                    Yleft = Y[right_idx]
                    self.right = TreeNode(self.depth + 1, self.max_depth)
                    self.right.fit(Xright, Yright)

    def find_split(self, X, Y, col):
        x_values = X[:, col]
        sort_idx = np.argsort(x_values)
        x_values = x_values[sort_idx]
        y_values = Y[sort_idx]

        boundaries = np.nonzero(y_values[:-1] != y_values[1:][0])
        best_split = None
        max_ig = 0

        for i in range(len(boundaries)):
            split = (x_values[i] + x_values[i + 1]) / 2
            ig = self.information_gain(x_values, y_values, split)

            if ig > max_ig:
                max_ig = ig
                best_split = split

        return max_ig, best_split

    def information_gain(self, x, y, split):
        y0 = y[x < split]
        y1 = y[x >= split]
        N = len(y)
        y0len = len(y0)

        if y0len == 0 or y0len == N:
            return 0
        p0 = float(len(y0)) / N
        p1 = 1 - p0

        return entropy(y) - p0 * entropy(y0) - p1 * entropy(y1)

    def predict_one(self, x):
        if self.col is not None and self.split is not None:
            feature = x[self.col]
            if feature < self.split:
                if self.left:
                    p = self.left.predict_one(x)
                else:
                    p = self.prediction[0]
            else:
                if self.right:
                    p = self.right.predict_one(x)
                else:
                    p = self.prediction[1]
        else:
            p = self.prediction
        return p

    def predict(self, X):
        N = len(X)
        P = np.zeros(N)

        for i in range(N):
            P[i] = self.predict_one(X[i])
        return P


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, Y):
        self.root = TreeNode(max_depth=self.max_depth)
        self.root.fit(X, Y)

    def predict(self, X):
        return self.root.predict(X)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)


if __name__ == "__main__":
    X, Y = get_data()
    idx = np.logical_or(Y == 0, Y == 1)
    X = X[idx]
    Y = Y[idx]

    Ntrain = int(len(Y) / 2)
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = DecisionTree()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print("Training Time: ", datetime.now() - t0)

    t0 = datetime.now()
    print("Train Accuracy: ", model.score(Xtrain, Ytrain))
    print("Time to compute train accuracy: ", datetime.now() - t0)

    t0 = datetime.now()
    print("Test Accuracy: ", model.score(Xtest, Ytest))
    print("Time to compute test accuracy: ", datetime.now() - t0)

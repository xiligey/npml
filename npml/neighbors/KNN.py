import numpy as np


class KNN(object):

    def fit(self, X, y, k=2):
        """KNN无需训练"""
        self.k = k
        return self

    def predict(self, X_test):
        """预测
        X_test 1维
        """
        distances = np.sqrt(np.sum((self.X - X_test) ** 2, axis=1))
        index = np.argpartition(self.X, self.k)[:self.k]  # 距离最小的k个点的索引
        return np.argmax(np.bincount(self.y[index]))

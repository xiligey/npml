import numpy as np
from numpy.linalg import pinv

from base import Regressor


class Ridge(Regressor):
    """"""

    def fit(self, X, y, alpha=0.1):
        """训练模型
        Parameters
        ----------
        X: 二维数组，训练集
        y: 一维数组，对应的标签值
        alpha: 岭回归二范式惩罚项系数
        """
        self.alpha = alpha
        n_samples, n_features = X.shape

        # 给X添加一列1， 将y转换成(n_samples, 1) 便于计算
        X = np.concatenate((np.ones(n_samples).reshape((n_samples, 1)), X), axis=1)
        y = y.reshape((n_samples, 1))

        self.theta = pinv(X.T @ X + self.alpha) @ X.T @ y  # A@B 等于 np.dot(A, B)
        self.intercept = self.theta[0, 0]  # 截距项
        self.coef = self.theta[1:, 0]  # 系数

        return self

    def predict(self, X):
        """预测
        Parameters
        ----------
            X: 待预测的二维数组
        Return
        ------
            预测标签的一维数组
        """
        return X @ self.coef + self.intercept

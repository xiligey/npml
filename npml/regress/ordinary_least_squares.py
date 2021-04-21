"""普通最小二乘法模块
相关理论 👉 https://github.com/xiligey/npml_theories/blob/master/regress/ordinary_least_squares.md
"""
import numpy as np
from numpy import ndarray
from numpy.linalg import inv, pinv

from npml.base import Regressor


class OrdinaryLeastSquares(Regressor):
    """普通最小二乘法"""

    def __init__(self):
        super().__init__()
        self.intercept = None  # 截距项
        self.coefficient = None  # 系数

    def fit(self, train_features: ndarray, train_values: ndarray, use_pinv: bool = False) -> None:
        """训练
        Args:
            train_features: (n_samples, n_features), 训练数据
            train_values: (n_samples,), 训练数据的值
            use_pinv: bool, 是否使用pinv来计算矩阵的逆，(当矩阵不可逆时使用pinv来计算)
        """
        n_samples, n_features = train_features.shape

        # 给train_features添加一列1(代表常数项)，将train_values转换成(n_samples, 1)
        train_features = np.concatenate(
            (np.ones(n_samples).reshape((n_samples, 1)), train_features), axis=1)
        train_values = train_values.reshape((n_samples, 1))

        # inverse = (X.T * X)^{-1}
        # A @ B == np.dot(A, B)
        inverse = pinv(train_features.T @ train_features) if use_pinv else inv(train_features.T @ train_features)
        theta = inverse @ train_features.T @ train_values

        self.intercept = theta[0, 0]
        self.coefficient = theta[1:, 0]

    def predict(self, test_features):
        """预测
        Args:
            test_features: (n_samples, n_features), 测试数据
        Returns:
            (n_samples,), 测试数据的预测值
        """
        return test_features @ self.coefficient + self.intercept

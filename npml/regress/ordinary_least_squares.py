"""普通最小二乘法"""
import numpy as np
from numpy import ndarray
from numpy.linalg import inv, pinv

from ..model import Regressor


class OrdinaryLeastSquares(Regressor):
    """普通最小二乘法
    Methods:
        fit(X, y)
        predict(X)
    Attributes:
        intercept: 截距项
        coefficient: 系数
    """

    def __init__(self):
        super().__init__()
        self.intercept = None  # 截距项
        self.coefficient = None  # 系数

    def fit(self, x_train: ndarray, y_train: ndarray, method="matrix", use_pinv=False) -> None:
        """训练
        Args:
            x_train: (n_samples, n_features), 训练数据
            y_train: (n_samples,), 训练数据的值
            method: str, 计算普通最小二乘法的方法，
                    可选["matrix", "gradient_descent", "gradient_descent_batch", "gradient_descent_random"]，
                    分别对应[矩阵、梯度下降法、批量梯度下降法、随机梯度下降法]
            use_pinv: bool, 是否使用pinv来计算矩阵的逆，(当矩阵不可逆时使用pinv来计算)
        """
        n_samples, n_features = x_train.shape

        # 给train_features添加一列1(代表常数项)，将train_values转换成(n_samples, 1)
        x_train = np.concatenate((np.ones(n_samples).reshape((n_samples, 1)), x_train), axis=1)
        y_train = y_train.reshape((n_samples, 1))

        if method == "matrix":
            self._fit_with_matrix(x_train, y_train, use_pinv)
        elif method == "gradient_descent":
            self._fit_with_gradient_descent(x_train, y_train)
        elif method == "gradient_descent_batch":
            self._fit_with_gradient_descent_batch(x_train, y_train)
        elif method == "gradient_descent_random":
            self._fit_with_gradient_descent_random(x_train, y_train)
        else:
            raise ValueError(
                f"method {method} is not supported, please use one of 'matrix', 'gradient_descent', "
                f"'gradient_descent_batch', 'gradient_descent_random'"
            )

    def _fit_with_matrix(self, x_train: ndarray, y_train: ndarray, use_pinv=False):
        """
        使用矩阵计算来训练普通最小二乘法
        Args:
            x_train: (n_samples + 1, n_features), 训练数据添加一列常数项（伴随矩阵）
            y_train: (n_samples, 1), 训练数据的值
            use_pinv: 是否使用pinv来计算矩阵的逆，(当矩阵不可逆时使用pinv来计算)
        """
        # inverse = (X.T * X)^{-1}
        # A @ B == np.dot(A, B)
        inverse = pinv(x_train.T @ x_train) if use_pinv else inv(x_train.T @ x_train)
        theta = inverse @ x_train.T @ y_train

        self.intercept = theta[0, 0]
        self.coefficient = theta[1:, 0]

    def _fit_with_gradient_descent(self, x_train, y_train):
        pass

    def _fit_with_gradient_descent_batch(self, x_train, y_train):
        pass

    def _fit_with_gradient_descent_random(self, x_train, y_train):
        pass

    def predict(self, test_features):
        """预测
        Args:
            test_features: (n_samples, n_features), 测试数据
        Returns:
            (n_samples,), 测试数据的预测值
        """
        return test_features @ self.coefficient + self.intercept

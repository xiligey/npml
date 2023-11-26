"""
普通最小二乘法

相关理论 👉 https://blog.csdn.net/xiligey1/article/details/81369713
"""

import numpy as np
from numpy import ndarray as array
from numpy.linalg import inv, pinv, LinAlgError

from npml.model import Regressor


class OrdinaryLeastSquares(Regressor):

    def fit(self, x_train: array, y_train: array, method: str = "mat", batch_size: int = 32,
            num_iter: int = 500) -> None:
        """训练
        :param x_train: (n_samples, n_features), 训练数据
        :param y_train: (n_samples,), 训练数据的值
        :param method: str, 计算普通最小二乘法的方法，可选mat、gd、gdb、sgd，分别对应矩阵法、梯度下降法、批量梯度下降法、随机梯度下降法
        :param batch_size: int, 批量梯度下降法的批大小
        :param num_iter: int, 梯度下降时的迭代次数
        """
        n_samples, n_features = x_train.shape

        # 给train_features添加一列1(代表常数项)，将train_values转换成(n_samples, 1)
        x_train = np.concatenate((np.ones(n_samples).reshape((n_samples, 1)), x_train), axis=1)
        y_train = y_train.reshape((n_samples, 1))

        if method == "mat":
            self._fit_with_matrix(x_train, y_train)
        elif method == "gd":
            self._fit_with_gradient_descent(x_train, y_train, num_iter)
        elif method == "dgb":
            self._fit_with_gradient_descent_batch(x_train, y_train, num_iter, batch_size)
        elif method == "sgd":
            self._fit_with_gradient_descent_random(x_train, y_train, num_iter)
        else:
            raise ValueError(f"method {method} is not supported, please use one of 'mat', 'gd', 'gdb', 'sgd'")

    def _fit_with_matrix(self, x_train: array, y_train: array) -> None:
        """使用矩阵计算得出普通最小二乘法的截距项和系数
        :param x_train: (n_samples + 1, n_features), 训练数据添加一列常数项（伴随矩阵）
        :param y_train: (n_samples, 1), 训练数据的值
        :exception  如果无法计算逆矩阵，则捕获LinAlgError异常，然后计算伪逆矩阵
        """
        # inverse = (X.T * X)^{-1}
        # A @ B == np.dot(A, B)
        x_train_square = x_train.T @ x_train
        try:
            inverse = inv(x_train_square)
        except LinAlgError as e:
            print(e)
            inverse = pinv(x_train_square)
        theta = inverse @ x_train.T @ y_train
        self.intercept, self.coeffs = theta[0, 0], theta[1:, 0]

    def _fit_with_gradient_descent(self, x_train, y_train, num_iter):
        """梯度下降法"""

    def _fit_with_gradient_descent_batch(self, x_train, y_train, num_iter, batch_size):
        """批量梯度下降法"""

    def _fit_with_gradient_descent_random(self, x_train, y_train, num_iter):
        """随机梯度下降法"""

    def predict(self, test_features):
        """预测
        :param test_features: (n_samples, n_features), 测试数据
        :return (n_samples,), 测试数据的预测值
        pred = X * coeffs + intercept
        """
        return test_features @ self.coeffs + self.intercept

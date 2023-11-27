"""
普通最小二乘法
支持矩阵法、批量梯度下降法、小批量梯度下降法、随机梯度下降法四种计算方式

相关理论 👉 https://blog.csdn.net/xiligey1/article/details/81369713
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray as array
from numpy.linalg import inv, pinv, LinAlgError

from npml.model import Regressor
from npml.utils.arrays import transform_array
from npml.utils.exceptions import NotFittedWithGradientDescentError
from npml.utils.metrics.regression import mse


class OrdinaryLeastSquares(Regressor):

    def __init__(self):
        super().__init__()
        # 模型的特征数量
        self.n_features = None
        # 记录迭代过程的损失值
        self.losses = None

    def fit(self, x_train: array, y_train: array, method: str = "mat", alpha: float = 0.01,
            batch_size: int = 32, num_iters: int = 500) -> None:
        """训练
        :param x_train: (n_samples, n_features), 训练数据
        :param y_train: (n_samples,), 训练数据的值
        :param method: str, 计算OLS的方法，可选mat、gd、gdb、sgd，分别对应矩阵法、批量梯度下降法、小批量梯度下降法、随机梯度下降法
        :param alpha: float, 学习率
        :param batch_size: int, 批量梯度下降法的批大小
        :param num_iters: int, 梯度下降时的迭代次数
        """
        n_samples, n_features = x_train.shape
        self.n_features = n_features

        # 设计矩阵，在原有数据的最左侧添加一列1
        x_train_design = transform_array(x_train)
        y_train = y_train.reshape((n_samples, 1))

        if method == "mat":
            self._fit_with_matrix(x_train_design, y_train)
        else:
            self.losses = []
            self.theta = np.zeros((self.n_features + 1, 1))
            if method == "gd":
                gd_func = self.gradient_descent_batch_func
            elif method == "gdb":
                gd_func = self.gradient_descent_mini_batch_func
            elif method == "sgd":
                gd_func = self.gradient_descent_mini_batch_func
                batch_size = 1
            else:
                raise ValueError(f"method {method} is not supported, please use one of 'mat', 'gd', 'gdb', 'sgd'")
            for _ in range(num_iters):
                self.theta = gd_func(*[x_train_design, y_train, self.theta, alpha, batch_size])
                y_train_pred = (x_train_design @ self.theta).flatten()
                # 损失函数为mse/2
                self.losses.append(mse(y_train, y_train_pred) / 2)
            self.losses = np.asarray(self.losses)

    def _fit_with_matrix(self, x_train_design: array, y_train: array) -> None:
        """使用矩阵计算得出普通最小二乘法的截距项和系数
        :param x_train_design: (n_samples, n_features + 1), 训练数据添加一列常数项（设计矩阵）
        :param y_train: (n_samples, 1), 训练数据的值
        :exception  如果无法计算逆矩阵，则捕获LinAlgError异常，然后计算伪逆矩阵
        """
        x_train_square = x_train_design.T @ x_train_design  # A @ B == np.dot(A, B)
        try:
            inverse = inv(x_train_square)
        except LinAlgError as e:
            print(e)
            inverse = pinv(x_train_square)
        self.theta = inverse @ x_train_design.T @ y_train

    @staticmethod
    def gradient_descent_batch_func(*args):
        """批量梯度下降更新系数方法"""
        x_train_design, y_train, theta, alpha, _ = args
        n_samples = x_train_design.shape[0]
        y_train_pred = x_train_design @ theta
        return theta - 1 / n_samples * alpha * x_train_design.T @ (y_train_pred - y_train)

    @staticmethod
    def gradient_descent_mini_batch_func(*args):
        """
        小批量/随机 梯度下降更新系数方法
        将batch_size设为1，即为随机梯度下降，大于1即为小批量梯度下降
        """
        x_train_design, y_train, theta, alpha, batch_size = args
        n_samples = x_train_design.shape[0]
        if batch_size >= n_samples:
            raise ValueError(f"Param batch_size[{batch_size}] should < n_samples[{n_samples}]")
        random_indices = np.random.choice(x_train_design.shape[0], size=batch_size, replace=False)
        x_train_sub, y_train_sub = x_train_design[random_indices], y_train[random_indices]
        y_train_pred = x_train_sub @ theta
        return theta - 1 / batch_size * alpha * x_train_sub.T @ (y_train_pred - y_train_sub)

    def predict(self, x_test, designed: bool = False):
        """预测
        :param x_test: 测试数据
        :param designed: 测试数据是否是设计矩阵（即是否已经添加一列常数列）
        :return (n_samples,), 测试数据的预测值
        """
        x_test_design = x_test if designed else transform_array(x_test)
        return x_test_design @ self.theta

    def plot_losses(self):
        """绘制迭代过程中的损失变化情况"""
        if self.losses is None:
            raise NotFittedWithGradientDescentError()
        plt.plot(range(1, self.losses.shape[0] + 1), self.losses)
        plt.title('Loss changes with the number of iterations')
        plt.xlabel('number of iter')
        plt.ylabel('current loss')
        plt.show()

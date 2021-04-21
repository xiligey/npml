"""岭回归
相关理论 👉 https://github.com/xiligey/npml_theories/blob/master/regress/ridge.md
"""
import numpy as np
from numpy.linalg import inv, pinv

from npml.base import Regressor


class Ridge(Regressor):

    def __init__(self):
        super().__init__()
        self.intercept = None  # 截距项
        self.coefficient = None  # 系数
        self.alpha = None  # 二范式正则化系数

    def fit(self, train_features, train_values, alpha=0.1, use_pinv: bool = False) -> None:
        """训练
        Args:
            train_features: (n_samples, n_features), 训练数据
            train_values: (n_samples,), 训练数据的值
            alpha: float, 二范式正则化系数
            use_pinv: bool, 是否使用pinv来计算矩阵的逆，(当矩阵不可逆时使用pinv来计算)
        """
        self.alpha = alpha
        n_samples, n_features = train_features.shape

        # 给X添加一列1， 将y转换成(n_samples, 1) 便于计算
        train_features = np.concatenate((np.ones(n_samples).reshape((n_samples, 1)), train_features), axis=1)
        train_values = train_values.reshape((n_samples, 1))

        inverse = pinv(train_features.T @ train_features + self.alpha) if use_pinv else inv(
            train_features.T @ train_features + self.alpha)

        theta = inverse @ train_features.T @ train_values
        self.intercept = theta[0, 0]  # 截距项
        self.coefficient = theta[1:, 0]  # 系数

    def predict(self, pred_features):
        """预测
        Args:
            pred_features: (n_samples, n_features), 测试数据
        Returns:
            (n_samples,), 测试数据的预测值
        """
        return pred_features @ self.coefficient + self.intercept

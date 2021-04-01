import numpy as np

from base import ClassifierMixin
from utils.activation_functions import sigmoid


class MultiLayerPerceptron(ClassifierMixin):
    """多层感知器"""

    def __init__(self, n_input, n_hidden, n_output):
        """初始化多层感知器

        Parameters
        ----------
        n_input：输入层神经元的数量
        n_hidden：隐藏层神经元的数量
        n_output：输出层神经元的数量

        Returns
        ------
        未经训练的初始多层感知器
        """
        self.n_input = n_input + 1  # 加上偏置项b
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.ai = np.ones(self.n_input)  # 初始化输入层的权重

    @staticmethod
    def dsigmoid(x):
        """sigmoid函数的导数"""
        return sigmoid(x) * (1 - sigmoid(x))

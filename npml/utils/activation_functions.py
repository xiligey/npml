"""激活函数"""
import numpy as np
from numpy import ndarray


def elu(x: ndarray, alpha: float) -> ndarray:
    """elu激活函数"""
    return np.array([i if i > 0 else alpha * (np.exp(i) - 1) for i in x])


def leaky_relu(x: ndarray) -> ndarray:
    """"""
    return np.array([max(0.01 * i, i) for i in x])


def prelu(x: ndarray, alpha=0.5) -> ndarray:
    return np.array([max(alpha * i, i) for i in x])


def relu(x: ndarray) -> ndarray:
    return np.array([max(0, i) for i in x])


def sigmoid(x: ndarray) -> ndarray:
    """sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))


def sign(x: ndarray) -> ndarray:
    """符号函数"""
    return np.sign(x)


def step(x: ndarray) -> ndarray:
    """TODO: step激活函数"""


def swish(x: ndarray) -> ndarray:
    """自控门激活函数"""
    return x / (1 + np.exp(-x))


def tanh(x: ndarray) -> ndarray:
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

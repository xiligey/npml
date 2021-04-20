"""各种距离
具体计算公式 👉 https://github.com/xiligey/npml_theories/blob/master/utils/distances.md

N维空间内两个点x=(x1,x2,...,xn)和点y=(y1,y2,...,yn)之间的各种距离
    - 欧氏距离
    - 曼哈顿距离
    - 切比雪夫距离
    - 闵可夫斯基距离
    - 标准化欧氏距离
    - 马氏距离
    - 巴氏距离
    - 余弦相似度
    - 编辑距离
    - 文本相似度
"""
import numpy as np
from numpy import ndarray


def euclidean_distance(x: ndarray, y: ndarray) -> float:
    """欧氏距离
    Args:
        x: (n_dim, 1), n维空间的一个样本点
        y: (n_dim, 1), n维空间的一个样本点
    Returns:
        两点的欧式距离
    """
    return np.sqrt(np.sum((x - y) ** 2))


def manhattan_distance(x: ndarray, y: ndarray) -> float:
    """曼哈顿距离
    Args:
        x: (n_dim, 1), n维空间的一个样本点
        y: (n_dim, 1), n维空间的一个样本点
    Returns:
        两点的曼哈顿距离
    """
    return float(np.sum(np.abs(x - y)))


def chebyshev_distance(x: ndarray, y: ndarray) -> float:
    """切比雪夫距离
    Args:
        x: (n_dim, 1), n维空间的一个样本点
        y: (n_dim, 1), n维空间的一个样本点
    Returns:
        两点的切比雪夫距离
    """
    return np.max(np.abs(x - y))


def minkowski_distance(x: ndarray, y: ndarray, p: int) -> float:
    """闵可夫斯基距离
    Args:
        x: (n_dim, 1), n维空间的一个样本点
        y: (n_dim, 1), n维空间的一个样本点
        p: int, 闵可夫斯基距离其实是一类距离，当p=2时，等价于欧氏距离
    Returns:
        两点的闵可夫斯基距离
    """
    return (np.sum((np.abs(x - y)) ** p)) ** (1 / p)


def standard_euclidean_distance(x: ndarray, y: ndarray, sigma: float) -> float:
    """标准化欧氏距离
    Args:
        x: (n_dim, 1), n维空间的一个样本点
        y: (n_dim, 1), n维空间的一个样本点
        sigma: x, y所在样本集的标准差
    Returns:
        两点的标准化欧氏距离
    """
    return np.sqrt(np.sum(((x - y) / sigma) ** 2))


def mahalanobis_distance(x: ndarray, y: ndarray, covariance_matrix: ndarray) -> float:
    """马氏距离
    Parameters
    ----------
        x: (n_dim, 1), n维空间的一个样本点
        y: (n_dim, 1), n维空间的一个样本点
        covariance_matrix: x和y所在样本集合的协方差矩阵
    Return
    ------
        两点的马氏距离
    """
    cov = np.linalg.inv(covariance_matrix)  # 协方差矩阵的逆
    subtract = (x - y).reshape((-1, 1))
    return np.sqrt(np.dot(np.dot(subtract.T, cov), subtract))[0, 0]


def bhattacharyya_distance(x: ndarray, y: ndarray) -> float:
    """TODO: 巴氏距离
    Args:
        x: (n_dim, 1), n维空间的一个样本点
        y: (n_dim, 1), n维空间的一个样本点
    Returns:
        两点的巴氏距离
    """


def cosine_similarity(x: ndarray, y: ndarray) -> float:
    """余弦相似度
    Parameters
    ----------
        x: (n_dim, 1), n维空间的一个样本点
        y: (n_dim, 1), n维空间的一个样本点

    Return
    ------
        两点的余弦相似度
    """
    return np.dot(x, y) / np.sqrt(np.sum(x ** 2)) / np.sqrt(np.sum(y ** 2))


def edit_distance(x: ndarray, y: ndarray) -> float:
    """TODO: 编辑距离"""


def text_similarity(x, y) -> float:
    """TODO: 文本相似度"""

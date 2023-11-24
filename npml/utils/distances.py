"""各种距离
具体计算公式 👉 https://blog.csdn.net/xiligey1/article/details/134589861

N维空间内两个点x=(x1,x2,...,xn)和点y=(y1,y2,...,yn)之间的各种距离
    - 欧氏距离
    - 曼哈顿距离
    - 切比雪夫距离
    - 闵可夫斯基距离
    - 标准化欧氏距离
    - 马氏距离
    - 余弦相似度
    - 编辑距离
    - 文本相似度
"""
import numpy as np
from numpy import ndarray
from numpy.linalg import LinAlgError


def euclidean_distance(x: ndarray, y: ndarray) -> float:
    """欧氏距离"""
    return np.sqrt(np.sum((x - y) ** 2))


def manhattan_distance(x: ndarray, y: ndarray) -> float:
    """曼哈顿距离"""
    return np.sum(np.abs(x - y))


def chebyshev_distance(x: ndarray, y: ndarray) -> float:
    """切比雪夫距离"""
    return np.max(np.abs(x - y))


def minkowski_distance(x: ndarray, y: ndarray, p: int) -> float:
    """
    闵可夫斯基距离
    :param x: (n_dim,), n维空间的一个样本点
    :param y: (n_dim,), n维空间的一个样本点
    :param p: 正整数, 闵可夫斯基距离其实是一类距离，当p=2时，等价于欧氏距离
    :return: 两点的闵可夫斯基距离
    """
    return np.power(np.sum((np.abs(x - y)) ** p), 1 / p)


def standard_euclidean_distance(x: ndarray, y: ndarray, stds: ndarray) -> float:
    """标准化欧氏距离
    :param x: (n_dim,), n维空间的一个样本点
    :param y: (n_dim,), n维空间的一个样本点
    :param stds: (n_dim,), x, y所在样本集的每个维度的标准差
    :return:两点的标准化欧氏距离
    """
    stds[stds == 0] = 1  # 令标准差中=0的转为1，防止除零
    return np.sqrt(np.sum(((x - y) / stds) ** 2))


def mahalanobis_distance(x: ndarray, y: ndarray, covariance_matrix: ndarray) -> float:
    """马氏距离
    Parameters
    ----------
        x: (n_dim,), n维空间的一个样本点
        y: (n_dim,), n维空间的一个样本点
        covariance_matrix: (n_dim, n_dim), x和y所在样本集合的协方差矩阵
    Return
    ------
        两点的马氏距离
    """
    try:
        cov = np.linalg.inv(covariance_matrix)  # 协方差矩阵的逆
    except LinAlgError as error:
        print(error)
        cov = np.linalg.pinv(covariance_matrix)  # 伪逆矩阵

    subtract = (x - y).reshape((-1, 1))
    return np.sqrt(np.dot(np.dot(subtract.T, cov), subtract))[0, 0]


def cosine_similarity(x: ndarray, y: ndarray) -> float:
    """余弦相似度"""
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

"""各种距离和相似度
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
    - 编辑距离 TODO
    - 文本相似度  TODO
"""
import numpy as np
from numpy import ndarray

from npml.utils.decorators import check_array_dimension


@check_array_dimension(ndim=1)
def euclidean_distance(x: ndarray, y: ndarray) -> float:
    """欧氏距离
    Args:
        x: (n_dim, 1), n维空间的一个样本点
        y: (n_dim, 1), n维空间的一个样本点
    Returns:
        两点的距离
    """
    return np.sqrt(np.sum((x - y) ** 2))


def manhattan_distance(x, y):
    """曼哈顿距离"""
    return np.sum(np.abs(x - y))


def chebyshev_distance(x, y):
    """切比雪夫距离"""
    return np.max(np.abs(x - y))


def minkowski_distance(x, y, p):
    """闵可夫斯基距离
    @param x: n维空间的一个样本点
    @param y: n维空间的一个样本点
    @param p: 闵可夫斯基距离其实是一类距离，当p=2时，等价于欧氏距离
    @return: 两点的闵可夫斯基距离
    """
    return (np.sum((np.abs(x - y)) ** p)) ** (1 / p)


def standard_euclidean_distance(x, y, sigma):
    """标准化欧氏距离
    @param x: n维空间的一个样本点
    @param y: n维空间的一个样本点
    @param sigma: x y所在样本集的标准差
    @return: 两点的标准化欧氏距离
    """
    return np.sqrt(np.sum(((x - y) / sigma) ** 2))


def mahalanobis_distance(x, y, covariance_matrix):
    """马氏距离
    Parameters
    ----------
        x, y: n维空间的两个样本点
        covariance_matrix: x和y所在样本集合的协方差矩阵
    Return
    ------
        返回点x和y的马氏距离
    """
    cm_ = np.linalg.inv(covariance_matrix)  # 协方差矩阵的逆
    x_12 = x - y
    x_12 = x_12.reshape((-1, 1))
    return np.sqrt(np.dot(np.dot(x_12.T, cm_), x_12))[0, 0]


def cosine_similarity(x, y):
    """余弦相似度

    Parameters
    ----------
    x, y: n维空间的两个样本点

    Return
    ------
        返回点x和y的余弦相似度
    """
    return np.dot(x, y) / np.sqrt(np.sum(x ** 2)) / np.sqrt(np.sum(y ** 2))


if __name__ == '__main__':
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    print(cosine_similarity(x, y))

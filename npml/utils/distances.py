"""距离、相似度
点x=(x1,x2,...,xn)和点y=(y1,y2,...,yn)之间的各种距离
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


def euclidean_distance(x1, x2):
    """欧氏距离
    @param x1: n维空间的一个样本点
    @param x2: n维空间的一个样本点
    @return: 两点的欧氏距离
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


def manhattan_distance(x1, x2):
    """曼哈顿距离"""
    return np.sum(np.abs(x1 - x2))


def chebyshev_distance(x1, x2):
    """切比雪夫距离"""
    return np.max(np.abs(x1 - x2))


def minkowski_distance(x1, x2, p):
    """闵可夫斯基距离
    @param x1: n维空间的一个样本点
    @param x2: n维空间的一个样本点
    @param p: 闵可夫斯基距离其实是一类距离，当p=2时，等价于欧氏距离
    @return: 两点的闵可夫斯基距离
    """
    return (np.sum((np.abs(x1 - x2)) ** p)) ** (1 / p)


def standard_euclidean_distance(x1, x2, sigma):
    """标准化欧氏距离
    @param x1: n维空间的一个样本点
    @param x2: n维空间的一个样本点
    @param sigma: x1 x2所在样本集的标准差
    @return: 两点的标准化欧氏距离
    """
    return np.sqrt(np.sum(((x1 - x2) / sigma) ** 2))


def mahalanobis_distance(x1, x2, covariance_matrix):
    """马氏距离
    Parameters
    ----------
        x1, x2: n维空间的两个样本点
        covariance_matrix: x1和x2所在样本集合的协方差矩阵
    Return
    ------
        返回点x1和x2的马氏距离
    """
    cm_ = np.linalg.inv(covariance_matrix)  # 协方差矩阵的逆
    x_12 = x1 - x2
    x_12 = x_12.reshape((-1, 1))
    return np.sqrt(np.dot(np.dot(x_12.T, cm_), x_12))[0, 0]


def cosine_similarity(x1, x2):
    """余弦相似度

    Parameters
    ----------
    x1, x2: n维空间的两个样本点

    Return
    ------
        返回点x1和x2的余弦相似度
    """
    return np.dot(x1, x2) / np.sqrt(np.sum(x1 ** 2)) / np.sqrt(np.sum(x2 ** 2))


if __name__ == '__main__':
    x1 = np.array([1, 2, 3])
    x2 = np.array([4, 5, 6])
    print(cosine_similarity(x1, x2))

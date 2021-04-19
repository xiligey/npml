"""回归模型的评价指标"""
import numpy as np
from numpy import ndarray


def r2(y_true: ndarray, y_pred: ndarray) -> float:
    """r2"""
    y_mean = y_true.mean()
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_mean)**2)


def explained_variance(y_true: ndarray, y_pred: ndarray) -> float:
    """可解释方差"""
    return 1 - np.var(y_true - y_pred) / np.var(y_true)


def max_error(y_true: ndarray, y_pred: ndarray) -> float:
    """最大误差"""
    return np.max(np.abs(y_true - y_pred))


def mae(y_true: ndarray, y_pred: ndarray) -> float:
    """平均绝对误差 median absolute error"""
    return np.average(np.abs(y_true - y_pred))


def mse(y_true: ndarray, y_pred: ndarray) -> float:
    """均方误差"""
    return np.average((y_true - y_pred) ** 2)


def msle(y_true: ndarray, y_pred: ndarray) -> float:
    """平均对数误差mean squared logarithmic error"""
    return np.average((np.log(1 + y_true) - np.log(1 + y_pred))**2)


def median_absolute_error(y_true: ndarray, y_pred: ndarray) -> float:
    """中位绝对误差"""
    abs_ = np.sort(np.abs(y_true - y_pred))
    length = len(abs_)
    if length % 2 == 0:
        mae = 0.5 * (abs_[int(length / 2)] + abs_[int(length / 2 - 1)])
    else:
        mae = abs_[int((length + 1) / 2)]
    return mae

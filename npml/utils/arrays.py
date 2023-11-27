"""常用数组操作"""
from numpy import ndarray as array
import numpy as np


def transform_array(x: array):
    """为x添加一列常数列"""
    n_samples = x.shape[0]
    return np.concatenate((np.ones((n_samples, 1)), x), axis=1)

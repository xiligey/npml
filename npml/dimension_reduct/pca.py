"""主成分分析"""
from __future__ import annotations
import numpy as np
from numpy import ndarray

from npml.model import DimensionReducer


class PCA(DimensionReducer):
    __model_type = "principle_component_analysis"

    def __init__(self, eigenvectors=None):
        super().__init__()
        self.eigenvectors = eigenvectors

    def fit(self, train: ndarray, dim: int = 2) -> PCA:
        """
        训练
        Args:
            train: 二维数组 (n_samples, n_features)
            dim: 将数据缩放为dim维
        Returns:

        """
        train = train - train.mean(axis=0)  # 均值归0
        cov = np.cov(train[:, 0], train[:, 1])  # 协方差矩阵
        eigenvalues, eigenvectors = np.linalg.eig(cov)  # 特征值，特征向量(每一列为1个特征向量)

        # 重新排列特征向量 从左往右按特征值大小降序；再取排序后的前n_dim个特征向量, 每一列为一个特征
        index = np.argsort(eigenvalues[::-1])
        self.eigenvectors = eigenvectors[:, index]

        return self

    def transform(self, train: ndarray) -> ndarray:
        return np.matmul(train, self.eigenvectors)

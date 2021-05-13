"""k最近邻
相关理论 👉 https://www.notion.so/chenxilin/K-404ba990650f4cfaa086da31bd1776d9
"""
from typing import Callable

import numpy as np
from numpy import ndarray

from npml.model import Classifier
from npml.utils.distances import euclidean_distance


class KNN(Classifier):
    __model_type = 'k_nearest_neighbors'

    def __init__(self, k: int) -> None:
        """初始化k最近邻分类器
        Parameters
            k: int, 最近邻分类选取的最近邻居数
        """
        super().__init__()
        self.k = k
        self.train_features = None
        self.train_labels = None

    def fit(self, train_features: ndarray, train_labels: ndarray) -> None:
        """训练
        Args:
            train_features: (n_samples, n_features), 训练集的features
            train_labels: (n_samples,), 训练集的labels
        """
        self.train_features = train_features
        self.train_labels = train_labels.astype(int)  # 标签需为整数类型

    def predict(self, pred_features: ndarray, distance_func: Callable = euclidean_distance) -> ndarray:
        """预测
        Args:
            pred_features: (n_samples, n_features), 待预测的数据集
            distance_func: 计算距离的函数
        Returns:
            (n_samples,), 预测结果
        """
        # 初始化预测的标签数组
        pred_labels = np.empty(pred_features.shape[0])

        # 循环待预测的数据集中的每条数据
        for i, pred_feature in enumerate(pred_features):
            # 计算这条数据和所有样本点的距离，得到一个列表
            distances = [distance_func(pred_feature, x) for x in self.train_features]
            # 对该距离列表排序，获取距离最小的前k个样本点的索引
            top_k_indexes = np.argsort(distances)[:self.k]
            # 在train_labels中找到这top_k个样本对应的标签出现次数最多的那个标签，即为预测标签
            train_labels_top_k = self.train_labels[top_k_indexes]
            bin_count = np.bincount(train_labels_top_k)  # 求出每个标签出现的次数
            pred_labels[i] = np.argmax(bin_count)

        return pred_labels

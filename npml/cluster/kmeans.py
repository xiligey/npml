"""kmeans聚类
相关理论 👉 https://github.com/xiligey/npml_theories/blob/master/cluster/kmeans.md
"""
from typing import Callable

import numpy as np
from numpy import ndarray

from npml.base import Clusterer
from npml.utils.distances import euclidean_distance
from npml.utils.exceptions import NotFittedError, NotSupportedError


class KMeans(Clusterer):

    def __init__(self) -> None:
        super().__init__()
        self.labels = None
        self.centroids = None

    def fit(self, train_features: ndarray, k: int, init_centroids_method: str = "random", random_seed: int = 1,
            max_iter: int = 100000, distance_func: Callable = euclidean_distance) -> None:
        """训练
        Args:
            train_features: (n_samples, n_features), 训练集
            k: 聚类数
            init_centroids_method: 初始化聚类中心的方法
                - random: 随机选择样本中k个点作为初始聚类中心
                - kmeans++: TODO
            random_seed: 随机种子
            max_iter: 训练最大迭代次数
            distance_func: 计算距离的方法
        """
        n_samples, n_features = train_features.shape

        # 初始化样本的标签
        self.labels = np.empty(n_samples)
        # 初始化质心
        self.centroids = self._init_centroids(train_features, k, method=init_centroids_method, random_seed=random_seed)

        # while循环判断条件：质心是否发生变化，是否达到最大迭代次数
        centroids_changed, max_iter_reached = True, False
        iter_count = 0

        # 循环迭代
        while centroids_changed and not max_iter_reached:
            # 对每个样本，计算其属于哪个类
            for i, train_feature in enumerate(train_features):
                # distances = 当前点x 和 所有质心 的距离
                distances = [distance_func(train_feature, self.centroids[i]) for i in range(k)]
                self.labels[i] = np.argmin(distances)
            # 更新每个类的质心
            updated_centroids = np.array([np.mean(train_features[self.labels == i], axis=0) for i in range(k)])
            if (updated_centroids == self.centroids).all():
                print('已收敛，停止迭代')
                centroids_changed = False
            else:
                self.centroids = updated_centroids
            iter_count += 1
            if iter_count == max_iter:
                print('达到最大迭代次数%s，停止迭代' % max_iter)
                max_iter_reached = True

    @staticmethod
    def _init_centroids(x_train: ndarray, k: int, method: str = "random", random_seed=1) -> ndarray:
        """初始化聚类中心"""
        if method == "random":
            np.random.seed(random_seed)
            # 从x_train中随机选k个值作为初始质心
            random_indices = np.random.choice(len(x_train), k, replace=False)
            return x_train[random_indices]
        elif method == 'kmeans++':  # TODO KMeans++
            # - 从输入的数据点集合中随机选择一个点作为第一个聚类中心
            # - 对于数据集中的每一个点xi，计算它与已选择的聚类中心中最近聚类中心的距离d，
            #   然后选择使得d最大的那个点xi作为下一个聚类中心
            # - 重复以上两步骤，直到选择了k个聚类中心
            centroids_indices = np.zeros(k)  # 初始化聚类中心点索引
            # 1、随机选择一个点作为第一个聚类中心
            # first_index = np.random.choice(len(a), 1)[0]
            return x_train[centroids_indices]
        else:
            raise NotSupportedError("Only supported methods for initializing centroids: [random, kmeans++]")

    @staticmethod
    def _get_nearest_class(sample, centers):
        """点sample离centers中哪个质心更近，返回哪个质心的索引"""
        return np.argmin(np.sqrt(np.sum((centers - sample) ** 2, axis=1)))

    def predict(self, pred_features):
        """预测
        Args:
            pred_features: (n_samples, n_features), 待预测的数据集
        Returns:
            (n_samples,), pred_features中每个点预测的类别
        """
        if self.labels is None and self.centroids is None:
            raise NotFittedError("Model is not fitted yet.")
        return np.array([self._get_nearest_class(x, self.centroids) for x in pred_features])

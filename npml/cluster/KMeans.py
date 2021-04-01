import numpy as np

from base import Clusterer
from utils.distances import euclidean_distance
from utils.exceptions import NotFittedError, NotSupportedError
from numpy import ndarray


class KMeans(Clusterer):

    def __init__(self):
        super(KMeans, self).__init__()
        self.labels = None
        self.centroids = None

    def fit(self, x_train, k=8, init_centroids_method="random", max_iter=100000,
            distance=euclidean_distance):
        """
        @param x_train: 训练集, (n_samples, n_features)
        @param k: 将x_train聚成k类
        @param init_centroids_method: 初始化聚类中心的方式
               - random 随机选择样本x_train中的k个点作为初始聚类中心
               - kmeans++
        @param max_iter: 最大迭代次数，到达max_iter则停止迭代
        @param distance: 距离函数，默认欧氏距离，支持多种距离函数
        @return: self
                 self.centroids:  质心, (k,)
                 self.labels:     存放每个点对应的类
        """
        n_samples, n_features = x_train.shape
        self.labels = np.zeros(n_samples)
        # 初始化质心
        self._init_centroids(x_train, k, method=init_centroids_method)
        centroids_changed = True
        count = 0
        # 当质心变化时进入循环
        while centroids_changed:
            # 对每个样本，计算其属于哪个类
            for i, x_train in enumerate(x_train):
                # distances = 当前点x 和 所有质心 的距离
                distances = np.array([distance(x_train, self.centroids[i]) for i in range(k)])
                self.labels[i] = distances.argmin()
            # 对每个类，更新其质心
            updated_centroids = np.array([np.mean(x_train[self.labels == i], axis=0) for i in range(k)])
            if updated_centroids.tolist() == self.centroids.tolist():
                print('已收敛，停止迭代')
                centroids_changed = False
            else:
                self.centroids = updated_centroids
            count += 1
            if count == max_iter:
                print('达到最大迭代次数%s，停止迭代' % max_iter)
                return self
        return self

    def _init_centroids(self, x_train: ndarray, k: int, method: str = "random"):
        """初始化聚类中心"""
        if method == "random":
            # 从x_train中随机选k个值作为初始质心
            random_indices = np.random.choice(len(x_train), k, replace=False)
            self.centroids = x_train[random_indices]
        elif method == 'kmeans++':  # TODO KMeans++
            # - 从输入的数据点集合中随机选择一个点作为第一个聚类中心
            # - 对于数据集中的每一个点xi，计算它与已选择的聚类中心中最近聚类中心的距离d，
            #   然后选择使得d最大的那个点xi作为下一个聚类中心
            # - 重复以上两步骤，直到选择了k个聚类中心
            centroids_indices = np.zeros(k)  # 初始化聚类中心点索引
            # 1、随机选择一个点作为第一个聚类中心
            # first_index = np.random.choice(len(a), 1)[0]
            self.centroids = x_train[centroids_indices]
        else:
            raise NotSupportedError("Only supported methods for initializing centroids: [random, kmeans++]")

    @staticmethod
    def _get_nearest_class(sample, centers):
        """点sample离centers中哪个质心更近，返回哪个质心的索引"""
        return np.argmin(np.sqrt(np.sum((centers - sample) ** 2, axis=1)))

    def predict(self, X):
        """预测

        :param X: 需要预测的数据集，(n_samples, n_features)
        :return: 数据集每个点分到的类, (n_samples)
        """
        # fit之后会添加属性labels和centroids，遂可根据这个判断是否模型已经训练
        if not hasattr(self, 'labels') or not hasattr(self, 'centroids'):
            raise NotFittedError('fit first, then predict')
        return np.array([self._get_nearest_class(x, self.centroids) for x in X])


if __name__ == "__main__":
    X = np.array([[1, 1], [1, 2], [2, 1], [1, 10],
                  [2, 10], [2, 9], [9, 9], [9, 10]])
    model = KMeans()
    model.fit(X, k=3)
    print(model.centroids)
    print(model.labels)
    print(model.predict(np.array([[1, 0], [11, 12]])))

"""层次聚类
样本空间有$N$个点$\{x_1,x_2,...,x_n\}$，层次聚类的过程如下：

1、将每个点都单独归为1类

2、计算各个类之间的相似度/距离

3、将相似度最大/距离最近的两个类合并为1类

4、重复步骤2和3，直到所有类归为1类

最开始有$N$个类，每循环一次便有两个类合并了(即类的总数-1)，直到减到你想要的k个类，则可以停止迭代，若想得到所有结果，则一直减，减到1为止

如何计算两个类之间的距离：

- Single Linkage：取两个类中最近的两个样本的距离
- Complete Linkage：取两个类中最远的两个样本的距离
- Average Linkage：把两个类中点两两求距离然后取均值
"""
import numpy as np

from base import Clusterer


class Cluster(object):
    """定义一个簇"""

    def __init__(self):
        pass


class Hierarchical(Clusterer):
    def __init__(self, vec, left=None, right=None, distance=0, id_=None):
        super(Hierarchical, self).__init__()
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id_
        self.distance = distance

    def fit(self, X):
        """"""
        # 1将每个点归为1类
        # 2计算各个类之间的距离
        # 3将距离最近的两个类合并为一个类
        # 4重复2和3 直到所有类归为一类
        X = np.array([[1, 1], [1, 10], [1, 11], [10, 1], [11, 2], [11, 11]])
        m, n = X.shape
        labels = dict(zip(range(m), [0] * m))  # 类
        dicts = {}  # 类和类的距离 {(class1,class2): dict1, (class1, class3): dict2, ...}

        for i in range(m):
            pass

    @staticmethod
    def _calc_classes_distance(class1, class2, method=""):
        """计算两个类之间的距离"""

    def predict(self, X):
        """"""

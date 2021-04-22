import operator

import numpy as np

from model import Classifier


class DecisionTree(Classifier):
    """决策树的三种实现 ID3 C4.5 CART"""

    def __init__(self):
        super(DecisionTree, self).__init__()
        self.model_type = "DecisionTree"

    def fit(self, X, y, method='id3'):
        """训练决策树
        Parameters
        ----------
            X: 训练数据集，(n_samples, n_features)
            y: X对应的标签, (n_samples)
            method: 分裂方法，可选'id3', 'c4.5', 'cart'
        """

        DecisionTree._create_tree()  # todo 将X和y合并为data_set
        self.is_fitted = True
        return self

    def predict(self, X):
        """预测
        Parameters
        ----------
            X: 需预测的数据集, (n_samples, n_features)
        Return
        ------
            预测的标签, (n_samples)
        """
        # TODO 预测

    @staticmethod
    def _calc_entropy(data_set):
        """给定数据集 计算其信息熵"""
        y = data_set[:, -1]  # [0, 0, 0, 0, 1]
        counts = np.array([(y == i).sum() for i in np.unique(y)])
        #     counts = np.bincount(b)  # [1, 4]  # 这个方法只能是整数才适用
        p = counts / sum(counts)  # [0.2, 0.8]
        log_p = np.log2(p)  # [log(0.2), log(0.8)]
        return -(p * log_p).sum()

    @staticmethod
    def _split_data_set(data_set, feature_index, feature_value):
        """分割数据集
        data_set: 待分割数据集
        feature_index: 要按哪个特征分割，feature_index为该特征的索引
        feature_value: 该分割特征会有多个值(如0、1)，此处选择的是哪个值(选0还是1，选0就返回所有该分割特征=0的数据)
        返回
        """
        result_with_feature_index = data_set[data_set[:,
                                             feature_index] == feature_value]  # 筛选出指定列，并带指定该列的值的array
        # 去掉result_with_feature_index的特征列
        result_without_feature_index = np.hstack((result_with_feature_index[:, 0:feature_index],
                                                  result_with_feature_index[:,
                                                  feature_index + 1:]))
        return result_without_feature_index

    @staticmethod
    def _choose_best_split_feature_id3(data_set):
        """给定数据集 按id3算法选择该数据集的最佳切分特征"""
        n_features = data_set.shape[1] - 1  # 数据集特征数
        n_samples = data_set.shape[0]  # 数据集样本数
        base_entropy = DecisionTree._calc_entropy(data_set)  # 原数据集的熵
        best_feature = -1  # 初始化最佳切分特征
        best_info_gain = 0  # 最佳信息增益 (增益最大为最佳)

        for i in range(n_features):
            feature_values = np.unique(data_set[:, i])  # 特征i的不重复值
            new_entropy = 0  # 初始化 按特征i切分后的数据集的熵
            for feature_value in feature_values:
                sub_data_set = DecisionTree._split_data_set(
                    data_set, i, feature_value)  # 按特征i，及feature_value分割数据集
                n_samples_sub = sub_data_set.shape[0]
                new_entropy += n_samples_sub / n_samples * \
                               DecisionTree._calc_entropy(
                                   sub_data_set)  # 计算sub_data_set的熵
            info_gain = base_entropy - new_entropy  # 计算信息增益
            if info_gain > best_info_gain:  # 选出信息增益最大的那个
                best_info_gain = info_gain
                best_feature = i
        return best_feature

    @staticmethod
    def _choose_best_split_feature_c45(data_set):
        """给定数据集 按c45算法选择该数据集的最佳切分特征 TODO"""

    @staticmethod
    def _choose_best_split_feature_cart(data_set):
        """给定数据集 按cart算法选择该数据集的最佳切分特征 TODO"""

    @staticmethod
    def _majority_vote(classes):
        """多数表决法 找出classes列表中出现次数最多的那一个  实际验证 用operator时间最快"""
        class_count = {}
        for vote in classes:
            if vote not in class_count.keys():
                class_count[vote] = 0
            class_count[vote] += 1

        sorted_class_count = sorted(
            class_count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]

    @staticmethod
    def _create_tree(data_set, method='id3'):
        """根据data_set递归创建决策树
        递归结束条件：
            - 程序遍历完所有划分数据集的属性
            - 或者每个分支下的所有样本都属于同一分类
        如果程序遍历完所有属性，有的叶子节点的类标签不唯一，则采取多数表决的方法定义该叶子节点的分类

        使用字典来存储决策树
            """
        method_dict = {'id3': DecisionTree._choose_best_split_feature_id3,
                       'c4.5': DecisionTree._choose_best_split_feature_c45,
                       'cart': DecisionTree._choose_best_split_feature_cart}

        tree = {}  # 初始化决策树

        m, n = data_set.shape
        classes = data_set[:, -1]  # 当前数据集的类标签 即y

        if len(np.unique(classes)) == 1:  # 检查数据集是否每个分支的所有样本都属于同一分类
            print('所有样本都属于同一类，停止遍历')
            return classes[0]

        if n == 1:  # 遍历完所有特征 则多数表决
            print('遍历完所有特征，多数表决')
            return DecisionTree._majority_vote(classes)
        best_feature = method_dict[method](data_set)
        # best_feature = DecisionTree._choose_best_split_feature_id3(data_set)  # 最好的切分特征
        tree[best_feature] = {}  # 将最好的切分特征存储到tree

        unique_feature_values = np.unique(
            data_set[:, best_feature])  # 最好的切分特征所在列的不重复的values
        for feature_value in unique_feature_values:
            sub_data_set = DecisionTree._split_data_set(
                data_set, best_feature, feature_value)
            tree[best_feature][feature_value] = DecisionTree._create_tree(
                sub_data_set)
        return tree

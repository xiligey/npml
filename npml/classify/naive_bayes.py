"""朴素贝叶斯算法
相关理论 👉 https://github.com/xiligey/npml_theories/blob/master/classify/naive_bayes.md
"""
import numpy as np
from numpy import ndarray

from npml.model import Classifier
from npml.utils.exceptions import FitError


class NaiveBayes(Classifier):
    def __init__(self):
        super().__init__()
        self.dict_p_label = {}  # 存放每个类别的概率
        self.dict_p_feature_based_label = {}  # 存放每个特征基于每个类别的概率

    def fit(self, train_features: ndarray, train_labels: ndarray) -> None:
        """训练
        Args:
            train_features: (n_samples, n_features), 训练集的features
            train_labels: (n_samples,), 训练集的labels 整数型
        """
        n_samples, n_features = train_features.shape
        distinct_labels = np.unique(train_labels)  # 计算有多少个类别
        if distinct_labels.shape[0] == 1:
            raise FitError("Values of labels of train data must > 1")
        for distinct_label in distinct_labels:
            label_count = np.sum(train_labels == distinct_label)  # 当前label在训练集中出现过的次数
            self.dict_p_label[distinct_label] = label_count / n_samples  # 将当前label发生的概率添加到dict_p_label
            if distinct_label not in self.dict_p_feature_based_label.keys():
                self.dict_p_feature_based_label[distinct_label] = {}
            for i in range(n_features):
                feature_i = train_features[i, :]  # 第i个样本
                distinct_feature_values = np.unique(feature_i)
                if i not in self.dict_p_feature_based_label[distinct_label].keys():
                    self.dict_p_feature_based_label[distinct_label][i] = {}
                for feature_value in distinct_feature_values:
                    a = train_features[train_labels == distinct_label]
                    x = a[a[:, feature_i == feature_value]]
                    self.dict_p_feature_based_label[distinct_label][i][feature_value] = len(x) / label_count

    def predict(self, pred_features: ndarray) -> ndarray:
        """预测
        Args:
            pred_features: (n_samples, n_features), 待预测的数据集
        Returns:
            (n_samples,), 预测结果
        """
        return np.array([self._predict_one_sample(i) for i in pred_features])

    def _predict_one_sample(self, sample: ndarray) -> int:
        """预测一个样本点的类别
        Args:
            sample: (n_features,), 带预测的一个样本
        Returns:
            int, 预测的类别
        """
        keys = self.dict_p_label.keys()
        labels = list(keys)
        probabilities = [self.dict_p_label[key] * np.prod([self.dict_p_feature_based_label[key][s] for s in sample]) for
                         key in keys]
        return labels[np.argmax(probabilities)]

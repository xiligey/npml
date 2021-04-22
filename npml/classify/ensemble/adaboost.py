"""Adaboost"""
import numpy as np

from npml.model import Classifier


class Adaboost(Classifier):

    def fit(self, X, y, max_iter=500, base_classifier='decision_tree'):
        """训练

        Arguments:
            a {二维数组} -- (n_samples, n_features)
            b {一维数组} -- (n_samples)

        Keyword Arguments:
            max_iter {int} -- 最大迭代次数 (default: {500})
            base_classifier {str} -- 默认弱分类器 (default: {'decision_tree'})

        Returns:
            self -- 返回最终的强分类器
        """

        n_samples = X.shape[0]
        weights = np.ones(n_samples) / n_samples  # 初始化样本权重
        split_points = Adaboost._get_all_split_points(X)  # 所有分割点

        best_weak_classifiers = []  # 所有最优弱分类器集合
        best_weak_classifier_weights = []  # 所有弱分类器对应权重集合

        for i in range(max_iter):
            best_weak_classifier = None  # 初始化本轮迭代的最优弱分类器
            min_error = 1  # 初始化本轮的最优弱分类器的错误率

            for point in split_points:  # 循环每个分割点，每个分割点生成一个弱分类器
                weak_classifier = Adaboost._generate_weak_classifier(point)  # 生成弱分类器
                predictions = np.array([weak_classifier(x)
                                        for x in X])  # 计算弱分类器的预测结果
                error = Adaboost._calc_weak_classifier_error(
                    predictions, X, weights)  # 计算弱分类器的错误率
                if error < min_error:
                    min_error = error
                    best_weak_classifier = weak_classifier
            best_weak_classifier_weight = Adaboost._calc_best_weak_classifier_weight(
                min_error)  # 计算最优弱分类器的权重
            best_weak_classifiers.append(
                best_weak_classifier)  # 将本轮迭代的最优弱分类器添加到最优弱分类器集合
            # 将本轮迭代的最优弱分类器对应的权重添加到最优弱分类器对应的权重集合
            best_weak_classifier_weights.append(best_weak_classifier_weight)

            predictions = np.array([best_weak_classifier(x)
                                    for x in X])  # 最优弱分类器的预测结果
            weights = Adaboost._calc_new_weight(
                weights, best_weak_classifier_weight, predictions, y)  # 更新样本权重

        self.final_strong_classifier = Adaboost._generate_final_strong_classifier(best_weak_classifiers,
                                                                                  best_weak_classifier_weights)
        return self

    def predict(self, X):
        return [self.final_strong_classifier(x) for x in X]

    @staticmethod
    def _calc_weak_classifier_error(predictions, y, weights):
        """计算弱分类器错误率
        predictions: 弱分类器的预测结果
        weights: 样本权重
        """
        error = np.dot(weights, predictions != y)
        return error

    @staticmethod
    def _calc_best_weak_classifier_weight(error):
        """计算最优弱分类器的权重
        error：最优弱分类器的错误率
        """
        return np.log((1 - error) / error) / 2

    @staticmethod
    def _calc_new_weight(weights, best_weak_classifier_weight, predictions, y):
        """计算新的样本权重"""
        z_ = weights * np.exp(-best_weak_classifier_weight * y * predictions)
        z = np.sum(z_)
        weights = z_ / z
        return weights

    @staticmethod
    def _generate_final_strong_classifier(best_weak_classifiers, best_weak_classifier_weights):
        """生成最后的强分类器
        best_weak_classifiers： 每轮迭代的最优弱分类器集合
        best_weak_classifier_weights：分类器对用的权重集合
        """
        C = best_weak_classifiers
        W = best_weak_classifier_weights

        def final_strong_classifier(x):
            C_ = [c(x) for c in C]
            result = sum([w * c_ for c_, w in zip(C_, W)])
            sign_result = 1 if result >= 0 else -1
            return sign_result

        return final_strong_classifier

    @staticmethod
    def _generate_weak_classifier():
        """生成弱分类器

        """

    @staticmethod
    def calc_weak_classifier_error():
        """计算弱分类器错误率

        """

    @staticmethod
    def _clac_weak_classifier_weight():
        """计算弱分类器的权重

        """

    @staticmethod
    def _calc_new_samples_weights():
        """根据之前的样本权重weights计算新的样本权重

        """

    @staticmethod
    def _generate_final_strong_classifier():
        """生成最后的强分类器

        """


'''


class Adaboost(Classifier):

    def fit(self, a, b, max_iter=500):
        """"""
        n_samples = a.shape[0]
        weights = np.ones(n_samples) / n_samples  # 初始化样本权重
        split_points = Adaboost._get_all_split_points(a)  # 所有分割点

        best_weak_classifiers = []  # 所有最优弱分类器集合
        best_weak_classifier_weights = []  # 所有弱分类器对应权重集合

        for i in range(max_iter):
            best_weak_classifier = None  # 初始化本轮迭代的最优弱分类器
            min_error = 1  # 初始化本轮的最优弱分类器的错误率

            for point in split_points:  # 循环每个分割点，每个分割点生成一个弱分类器
                weak_classifier = Adaboost._generate_weak_classifier(
                    point)  # 生成弱分类器
                predictions = np.array([weak_classifier(x_train)
                                        for x_train in a])  # 计算弱分类器的预测结果
                error = Adaboost._calc_weak_classifier_error(
                    predictions, a, weights)  # 计算弱分类器的错误率
                if error < min_error:
                    min_error = error
                    best_weak_classifier = weak_classifier
            best_weak_classifier_weight = Adaboost._calc_best_weak_classifier_weight(
                min_error)  # 计算最优弱分类器的权重
            best_weak_classifiers.append(
                best_weak_classifier)  # 将本轮迭代的最优弱分类器添加到最优弱分类器集合
            # 将本轮迭代的最优弱分类器对应的权重添加到最优弱分类器对应的权重集合
            best_weak_classifier_weights.append(best_weak_classifier_weight)

            predictions = np.array([best_weak_classifier(x_train)
                                    for x_train in a])  # 最优弱分类器的预测结果
            weights = Adaboost._calc_new_weight(
                weights, best_weak_classifier_weight, predictions, b)  # 更新样本权重

        self.final_strong_classifier = Adaboost._generate_final_strong_classifier(best_weak_classifiers,
                                                                                  best_weak_classifier_weights)
        return self

    def predict(self, a):
        return [self.final_strong_classifier(x_train) for x_train in a]

    @staticmethod
    def _get_all_split_points(a):
        """计算所有分割点"""
        return [(a[i] + a[i + 1]) / 2 for i in range(len(a) - 1)]

    @staticmethod
    def _generate_weak_classifier(split_point):
        """根据分割点生成弱分类器"""

        def classifier(x_train):
            return 1 if x_train <= split_point else -1

        return classifier

    @staticmethod
    def _calc_weak_classifier_error(predictions, b, weights):
        """计算弱分类器错误率
        predictions: 弱分类器的预测结果
        weights: 样本权重
        """
        error = np.dot(weights, predictions != b)
        return error

    @staticmethod
    def _calc_best_weak_classifier_weight(error):
        """计算最优弱分类器的权重
        error：最优弱分类器的错误率
        """
        return np.log((1 - error) / error) / 2

    @staticmethod
    def _calc_new_weight(weights, best_weak_classifier_weight, predictions, b):
        """计算新的样本权重"""
        z_ = weights * np.exp(-best_weak_classifier_weight * b * predictions)
        z = np.sum(z_)
        weights = z_ / z
        return weights

    @staticmethod
    def _generate_final_strong_classifier(best_weak_classifiers, best_weak_classifier_weights):
        """生成最后的强分类器
        best_weak_classifiers： 每轮迭代的最优弱分类器集合
        best_weak_classifier_weights：分类器对用的权重集合
        """
        C = best_weak_classifiers
        W = best_weak_classifier_weights

        def final_strong_classifier(x_train):
            C_ = [c(x_train) for c in C]
            result = sum([w * c_ for c_, w in zip(C_, W)])
            sign_result = 1 if result >= 0 else -1
            return sign_result

        return final_strong_classifier


'''

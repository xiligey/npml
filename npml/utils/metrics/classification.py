"""分类指标"""
from numpy import ndarray


def accuracy(y_true: ndarray, y_pred: ndarray) -> float:
    """准确率：accuracy = (TP + TN) / (TP + TN + FP + FN)"""
    return (y_true == y_pred).mean()


def precision(y_true: ndarray, y_pred: ndarray) -> float:
    """精确率：precision = TP / (TP + FP)"""
    T = y_pred.sum()  # 预测为正类的样本数：T = TP + FP
    TP = ((y_true + y_pred) == 2).sum()
    return TP / T


def recall(y_true: ndarray, y_pred: ndarray) -> float:
    """召回率：recall = TP / (TP + FN)"""
    P = y_pred.sum()  # 真实为正类的样本数：P = TP + FN
    TP = ((y_true + y_pred) == 2).sum()
    return TP / P


def f1(y_true: ndarray, y_pred: ndarray) -> float:
    """f1 = 2 * precision * recall / (precision + recall)"""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r)


def roc_auc(y_true: ndarray, y_pred: ndarray) -> float:
    """roc曲线、auc值
    TODO  https://zhuanlan.zhihu.com/p/25212301
    """


def hinge(y_true: ndarray, y_pred: ndarray) -> float:
    """hinge损失 TODO"""

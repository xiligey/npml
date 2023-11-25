"""分类指标
具体计算公式 👉 https://blog.csdn.net/xiligey1/article/details/86622309

- accuracy 准确率
- auc Area Under Curve
- average_precisioin 平均精度
- balanced_accuray 平衡精度
- brier_score_loss Brier分数损失
- class_likelihood_ratios 二元分类的正似然比和负似然比
- classification_report 主要分类指标报告
- cohen_kappa Cohen的kappa
- confusion_matrix 混淆矩阵
- dcg 贴现累积收益
- det_curve 不通概率阈值的错误率
- f1 F1分数
- f_beta F-beta分数
- hamming_loss 平均汉明损失
- hinge_loss 平均铰链损失（非正则化）
- jaccard jaccard相似系数得分
- log_loss 对数损失，又称为逻辑损失或交叉熵损失
- matthews_corrcoef 马修斯相关系数（MCC）
- ndcg 标准化贴现累积增益
- precision_recall_curve 计算不通概率阈值的精确召回对
- precision_recall_fscore_support 计算每个类别的精度、召回率、F score和支持度
- precision 精度
- recall 召回率
- roc_auc ROC曲线下的面积
- top_k_accuracy Top-k准确度分类分数
- zero_one_loss 零一分类损失

TODO: 支持多分类的指标计算（目前只支持二分类指标的计算）
"""
import numpy as np
from numpy import ndarray


def accuracy(y_true: ndarray, y_pred: ndarray) -> float:
    """准确率 = 正确分类的样本数 / 总样本数"""
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true: ndarray, y_pred: ndarray, labels=None) -> ndarray:
    r"""混淆矩阵

    真实\预测   A    B
    A         15    5
    B         2    18
    """
    # 若未提供标签，则 计算（合并真实和预测 + 去重 + 排序）
    labels = np.sort(np.unique(np.append(y_true, y_pred))) if labels is None else labels
    n = labels.shape[0]  # 标签个数
    # 标签索引字典，用于后续找到标签对应的索引位置
    label_index_dict = {k: v for k, v in zip(labels, range(n))}

    # 初始化混淆矩阵
    cm = np.zeros((n, n), dtype=np.int64)
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        index1 = label_index_dict[true]
        index2 = label_index_dict[pred]
        cm[index1, index2] += 1
    return cm


def precision(y_true: ndarray, y_pred: ndarray, positive_label=1) -> float:
    """精确率"""
    cm = confusion_matrix(y_true, y_pred)
    precisions_ = __precisions(cm)
    return float(precisions_[positive_label])


def __precisions(conf_mat: ndarray) -> ndarray:
    """
    计算所有类别的精确率
    :param conf_mat: 混淆矩阵
    """
    # 真阳 = 混淆矩阵对角线上的值
    true_positives = np.diag(conf_mat)
    # 所有真 = 混淆矩阵按列求和
    trues = np.sum(conf_mat, axis=0)
    # 所有类别的精确率
    return true_positives / trues


def recall(y_true: ndarray, y_pred: ndarray, postive_label=1) -> float:
    """
    召回率
    :param y_true: 真实类别
    :param y_pred: 预测类别
    :param postive_label: 阳性代表的类别
    :return:
    """
    cm = confusion_matrix(y_true, y_pred)
    recalls_ = _recalls(cm)
    # 最终阳类的召回率
    return float(recalls_[postive_label])


def _recalls(conf_mat: ndarray) -> ndarray:
    """
    计算所有类别的召回率
    :param conf_mat: 混淆矩阵
    """
    # 真阳 = 混淆矩阵对角线上的值
    true_positives = np.diag(conf_mat)
    # 所有阳 = 混淆矩阵按行求和
    positives = np.sum(conf_mat, axis=1)
    # 所有类别的召回率
    return true_positives / positives


def f1(y_true: ndarray, y_pred: ndarray, positive_label=1) -> float:
    """f1 = 2 * precision * recall / (precision + recall)"""
    p = precision(y_true, y_pred, positive_label)
    r = recall(y_true, y_pred, positive_label)
    return 2 * p * r / (p + r)


def roc_auc(y_true: ndarray, y_pred: ndarray) -> float:
    """roc曲线、auc值
    TODO  https://zhuanlan.zhihu.com/p/25212301
    """


def hinge(y_true: ndarray, y_pred: ndarray) -> float:
    """hinge损失 TODO"""

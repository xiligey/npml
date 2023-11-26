import logging
from typing import Optional

from matplotlib import pyplot as plt
from numpy import ndarray

from .utils.exceptions import NotFittedError
from .utils.metrics.classification import accuracy
from .utils.metrics.regression import r2


class Model:
    """所有模型的基类"""
    __model_type = "model"

    def fit(self, *args, **kwargs) -> None:
        """训练"""

    def predict(self, *args, **kwargs) -> ndarray:
        """预测"""

    def score(self, *args, **kwargs) -> ndarray:
        """评分"""

    def plot(self, *args, **kwargs) -> None:
        """绘图"""

    @property
    def model_type(self) -> str:
        """获取模型类型"""
        return self.__model_type


class SupervisedModel(Model):
    """监督模型"""
    __model_type = "supervised_model"


class UnsupervisedModel(Model):
    """无监督模型"""
    __model_type = "unsupervised_model"


class SemiSupervisedModel(Model):
    """半监督模型"""
    __model_type = "semi_supervised_model"


class Regressor(SupervisedModel):
    """回归模型"""
    __model_type = "regressor"

    def __init__(self):
        super().__init__()
        self.intercept: Optional[float] = None  # 截距项
        self.coeffs: Optional[ndarray] = None  # 系数

    def score(self, y_true: ndarray, y_pred: ndarray) -> float:
        return r2(y_true, y_pred)

    def plot(self, features: ndarray, values: ndarray) -> None:
        """绘图(目前只支持特征维度为1)
        Args:
            features: (n_samples, n_features), 用于绘图的样本数据
            values: (n_samples,), 样本数据对应的值
        """
        if self.coeffs is not None and self.intercept is not None:
            if len(self.coeffs) > 1:
                logging.warning("Number of feature dimension > 1, plotting is not supported yet")
            else:
                predicted = self.predict(features)
                features = features.flatten()
                plt.plot(features, values, label='original')
                plt.plot(features, predicted, label='predicted')
                plt.legend()
                plt.show()
        else:
            raise NotFittedError("Model is not fitted yet.")


class Classifier(SupervisedModel):
    """分类模型"""
    __model_type = "classifier"

    def score(self, y_true: ndarray, y_pred: ndarray) -> float:
        return accuracy(y_true, y_pred)


class Clusterer(UnsupervisedModel):
    """聚类模型"""
    __model_type = "clusterer"

    def score(self, y_true: ndarray, y_pred: ndarray) -> float:
        # TODO 评估聚类效果
        pass


class DimensionReducer(UnsupervisedModel):
    """降维模型"""
    __model_type = "dimension_reducer"


class AnomalyDetector(UnsupervisedModel):
    """异常检测模型"""
    __model_type = "anomaly_detector"

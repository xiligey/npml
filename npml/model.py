from typing import Callable

from numpy import ndarray

from metrics.classification import accuracy
from metrics.regression import r2


class Model:
    """所有模型的基类"""
    __model_type = "model"

    def predict(self, *args, **kwargs) -> ndarray:
        """预测"""

    def score(self, *args, **kwargs) -> float:
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

    def __init__(self, scoring: Callable):
        super().__init__()
        self.scoring = scoring

    def fit(self, *args, **kwargs) -> None:
        """训练"""

    def score(self, x_pred: ndarray, y_true: ndarray) -> float:
        """
        回归模型的默认评估方法是决定系数R方
        Args:
            x_pred: (n_samples, n_features) 待预测的数据集
            y_true: (n_samples,) 对应的真实值
        Returns:
            得分
        """
        y_pred = self.predict(x_pred)
        return self.scoring(y_true, y_pred)


class UnsupervisedModel(Model):
    """无监督模型"""
    __model_type = "unsupervised_model"


class SemiSupervisedModel(Model):
    """半监督模型"""
    __model_type = "semi_supervised_model"


class Regressor(SupervisedModel):
    """回归模型"""
    __model_type = "regressor"

    def __init__(self, scoring: Callable = r2):
        super().__init__(scoring)
        self.intercept = None  # 截距项
        self.coefficient = None  # 系数


class Classifier(SupervisedModel):
    """分类模型"""
    __model_type = "classifier"

    def __init__(self, scoring: Callable = accuracy):
        super().__init__(scoring)


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

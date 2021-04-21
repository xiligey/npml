"""base模块"""
from numpy import ndarray

from npml.utils.metrics.classification import accuracy
from npml.utils.metrics.regression import r2


class Model:
    """所有模型的基类"""
    __model_type = "model"

    def fit(self) -> None:
        """训练"""

    def predict(self) -> ndarray:
        """预测"""

    def score(self) -> ndarray:
        """评分"""

    def plot(self) -> None:
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

    def score(self, y_true: ndarray, y_pred: ndarray) -> float:
        return r2(y_true, y_pred)


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

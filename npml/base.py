from numpy import ndarray

from metrics.classification import accuracy
from metrics.regression import r2
from python.npml.utils.decorators import check_params_type


class Model(object):
    """所有模型的基类"""
    __model_type = "model"

    def fit(self, *args, **kwargs):
        """训练"""

    def predict(self, *args, **kwargs):
        """预测"""

    def score(self, *args, **kwargs):
        """评分"""

    def plot(self, *args, **kwargs):
        """绘图"""

    @property
    def model_type(self):
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

    @check_params_type()
    def score(self, y_true: ndarray, y_pred: ndarray) -> float:
        return r2(y_true, y_pred)


class LinearRegressor(Regressor):
    """线性回归模型"""
    __model_type = "linear_regressor"

    def __init__(self):
        self.intercept = None  # 截距项
        self.coefficient = None  # 系数

    @property
    def is_fitted(self):
        """判断模型是否已训练，如果coefficient和intercept属性皆不为空则表示模型已训练"""
        return True if self.coefficient is not None and self.intercept is not None else True


class Classifier(SupervisedModel):
    """分类模型"""
    __model_type = "classifier"

    @check_params_type()
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

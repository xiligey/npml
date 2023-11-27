class NpmlError(Exception):
    """npml异常"""


class ParametersError(NpmlError):
    """参数错误"""


class NotFittedError(NpmlError):
    """模型未训练"""


class NotFittedWithGradientDescentError(NpmlError):
    """模型未通过梯度下降法训练"""


class FitError(NpmlError):
    """模型训练失败"""


class NotSupportedError(NpmlError):
    """不支持该功能"""

class NPMLError(Exception):
    """npml异常"""


class ParametersError(NPMLError):
    """参数错误"""


class NotFittedError(NPMLError):
    """模型未训练"""


class FitError(NPMLError):
    """模型训练失败"""


class NotSupportedError(NPMLError):
    """不支持该功能"""

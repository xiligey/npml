class NPMLError(Exception):
    """npml异常"""


class ParametersError(NPMLError):
    """参数错误"""


class NotFittedError(NPMLError):
    """模型未训练或训练失败"""


class NotSupportedError(NPMLError):
    """不支持该功能"""


if __name__ == '__main__':
    a = ParametersError("123")
    raise a

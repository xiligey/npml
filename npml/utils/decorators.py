"""npml装饰器"""

from functools import wraps
from typing import Callable

import numpy as np
from numpy import ndarray

from npml.utils.exceptions import ParametersError


def check_params_type(type_=ndarray, except_first=True):
    """检查参数类型

    Args:
        type_ ([type], optional): [description]. Defaults to ndarray.
        except_first (bool, optional): [description]. Defaults to True.

    Raises:
        ParametersError: [description]

    Returns:
        [type]: [description]
    """    """"""
    """
    检查参数类型是否为type_，如果不是，报ParametersError
    @param type_: 参数需满足的类型
    @param except_first: 是否不检测第一个参数，默认为True，因为第一个参数一般为self
    """

    def check_type(func):
        @wraps(func)
        def check(*args, **kwargs):
            filtered_args = args[1:] if except_first else args
            for arg in filtered_args:
                if not isinstance(arg, type_):
                    raise ParametersError("%s must be %s" % (arg, type_.__name__))
            return func(*args, **kwargs)

        return check

    return check_type


def check_array_dimension(ndim: int) -> Callable:
    """检查数组参数的维度
    Args:
        ndim: 指定的维度
    Returns:
        如果该装饰器应用在某函数上，该函数的参数的维度不等于ndim则引发ParametersError
    """

    def check_dimension(func):
        @wraps(func)
        def check(*args, **kwargs):
            for arg in args:
                if arg.ndim != ndim:
                    raise ParametersError("%s must be %sd array" % (arg, ndim))
            return func(*args, **kwargs)

        return check

    return check_dimension

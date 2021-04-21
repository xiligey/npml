"""npml绘图"""
from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt

from npml.utils.activation_functions import *


def plot_activation_functions(
    x: ndarray,
    func: Callable[[ndarray], ndarray],
    func_equation: Optional[str] = None,
    func_equation_coordinate: Optional[Tuple[float, float]] = None,
    xticks: Optional[range] = None,
    yticks: Optional[range] = None,
    save_fig=False
):
    """绘制激活函数
    Args:
        x: 要绘制的点的横坐标数组
        func: [[npml.utils.activation_functions]]里的激活函数
        func_equation: 激活函数数学公式的mathjax格式
        func_equation_coordinate: 图中放置数学公式的坐标
        xticks: x轴的刻度
        yticks: y轴的刻度
        save_fig: 是否保存为图片
    """
    """绘制激活函数"""
    y = func(x)
    plt.plot(x, y)
    plt.title('%s激活函数' % func.__name__, loc='left')  # 标题放在左侧
    plt.axis('scaled')  # 让x轴和y轴的单位度量一样长
    xticks = range(x.min() - 1, x.max() + 1, 1) if xticks is None else xticks
    yticks = range(y.min() - 1, y.max() + 1, 1) if xticks is None else xticks
    plt.xticks(xticks)
    plt.yticks(yticks)
    if func_equation is not None:
        coordinate = (x.min() - 1, y.max() - 1) if func_equation_coordinate is None else func_equation_coordinate
        plt.text(coordinate[0], coordinate[1], r"" + func_equation, color='green')  # 添加公式
    ax = plt.gca()
    ax.spines['right'].set_color('none')  # 隐藏右方竖线
    ax.spines['top'].set_color('none')  # 隐藏上方横线
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))  # y轴居中
    if save_fig:
        plt.savefig('../images/activation_functions/relu激活函数.png')
    plt.show()

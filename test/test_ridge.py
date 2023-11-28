import pandas as pd

from npml.linear_model.ridge import Ridge as Ridge_npml
from sklearn.linear_model import Ridge as Ridge_sk


def test_ordinary_least_squares():
    train_data = pd.read_csv('..\\data\\world-happiness-report-2017.csv')
    input_param_name = 'Economy..GDP.per.Capita.'
    output_param_name = 'Happiness.Score'
    x = train_data[[input_param_name]].values
    y = train_data[[output_param_name]].values

    ridge = Ridge_npml()
    l2 = 0.1
    # 矩阵法
    ridge.fit(x, y)
    print(f"矩阵法系数：{ridge.theta}")
    # 批量梯度下降法
    ridge.fit(x, y, method='gd', alpha=0.1, l2=l2, num_iters=10000)
    print(f"批量梯度下降法系数：{ridge.theta}")
    ridge.plot_losses()
    # 小批量梯度下降法
    ridge.fit(x, y, method='gdb', alpha=0.1, l2=l2, num_iters=10000, batch_size=2)
    print(f"小批量梯度下降法系数：{ridge.theta}")
    ridge.plot_losses()
    # 随机梯度下降法
    ridge.fit(x, y, method='sgd', alpha=0.1, l2=l2, num_iters=10000)
    print(f"随机梯度下降法系数：{ridge.theta}")
    ridge.plot_losses()
    # sklearn的岭回归
    ridge_sk = Ridge_sk(alpha=l2)
    ridge_sk.fit(x, y)
    print(f"sklearn的岭回归回归系数：{ridge_sk.coef_, ridge_sk.intercept_}")

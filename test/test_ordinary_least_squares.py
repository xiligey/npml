import pandas as pd

from npml.linear_model.ordinary_least_squares import OrdinaryLeastSquares
from sklearn.linear_model import LinearRegression


def test_ordinary_least_squares():
    data = pd.read_csv(r'C:\Users\chenxilin\BaiduSyncdisk\Python\npml\data\world-happiness-report-2017.csv')

    # 得到训练和测试数据
    train_data = data

    input_param_name = 'Economy..GDP.per.Capita.'
    output_param_name = 'Happiness.Score'

    x = train_data[[input_param_name]].values
    y = train_data[[output_param_name]].values

    ols = OrdinaryLeastSquares()
    # 矩阵法
    ols.fit(x, y)
    print(f"矩阵法系数：{ols.theta}")
    # 批量梯度下降法
    ols.fit(x, y, method='gd', alpha=0.1, num_iters=10000)
    print(f"批量梯度下降法系数：{ols.theta}")
    ols.plot_losses()
    # 小批量梯度下降法
    ols.fit(x, y, method='gdb', alpha=0.1, num_iters=10000, batch_size=2)
    print(f"小批量梯度下降法系数：{ols.theta}")
    ols.plot_losses()
    # 随机梯度下降法
    ols.fit(x, y, method='sgd', alpha=0.1, num_iters=10000)
    print(f"随机梯度下降法系数：{ols.theta}")
    ols.plot_losses()
    # sklearn的线性回归
    ols_sk = LinearRegression()
    ols_sk.fit(x, y)
    print(f"sklearn的线性回归系数：{ols_sk.coef_, ols_sk.intercept_}")

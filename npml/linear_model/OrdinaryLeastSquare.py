import numpy as np
from numpy.linalg import inv
from numpy import ndarray
from base import LinearRegressor
from utils.decorators import check_params_type
from utils.exceptions import NotFittedError


class OrdinaryLeastSquare(LinearRegressor):
    """普通最小二乘法"""

    @check_params_type()
    def fit(self, x_train: ndarray, y_train: ndarray) -> None:
        """
        @param x_train: 训练数据，二维数组
        @param y_train: 训练数据的真实值，一维数组
        @return: self
        """
        n_samples, n_features = x_train.shape

        # 给X添加一列1， 将y转换成(n_samples, 1)
        x_train = np.concatenate(
            (np.ones(n_samples).reshape((n_samples, 1)), x_train), axis=1)
        y_train = y_train.reshape((n_samples, 1))

        # A @ B == np.dot(A, B)
        theta = inv(x_train.T @ x_train) @ x_train.T @ y_train

        self.intercept = theta[0, 0]  # 截距项
        self.coefficient = theta[1:, 0]  # 系数

    @check_params_type()
    def predict(self, x_test):
        """
        @param x_test: 测试数据，二维数组
        @return: 测试数据的预测值，一维数组
        """
        return x_test @ self.coefficient + self.intercept

    @check_params_type()
    def plot(self, x_train, y_train):
        if not self.is_fitted:
            raise NotFittedError("Only fitted model are supported to plot.")
        pass


if __name__ == '__main__':
    # 对比npml和sklearn的结果
    print('----npml----')
    ols = OrdinaryLeastSquare()
    x = np.array([[1], [2], [3]])
    y = np.array([10, 12, 14])
    ols.fit(x, y)
    print('b = %sx + %s' % (ols.coefficient[0], ols.intercept))
    prediction = ols.predict(np.array([[10]]))
    print('x_train = 10 时, b = %s' % prediction[0])

    print('----sklearn----')
    from sklearn.linear_model import LinearRegression

    ols_sk = LinearRegression()
    ols_sk.fit(x, y)
    print('b = %sx + %s' % (ols_sk.coef_[0], ols_sk.intercept_))
    prediction_sk = ols_sk.predict(np.array([[10]]))
    print('x_train = 10 时, b = %s' % prediction_sk[0])

    # 断言npml和sklearn的结果误差不超过1e-6
    assert abs(ols_sk.coef_ - ols.coefficient) < 1e-6
    assert abs(ols_sk.intercept_ - ols.intercept) < 1e-6

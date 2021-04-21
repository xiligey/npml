
import numpy as np
from sklearn.linear_model import LinearRegression

from npml.regress.ordinary_least_squares import OrdinaryLeastSquares


class TestOLS:
    def test_ols(self):
        # 对比npml和sklearn的结果
        print('----npml----')
        ols = OrdinaryLeastSquares()
        x = np.array([[1], [2], [3]])
        y = np.array([10, 12, 14])
        ols.fit(x, y)
        print('y = %sx + %s' % (ols.coefficient[0], ols.intercept))
        prediction = ols.predict(np.array([[10]]))
        print('train_features = 10 时, y = %s' % prediction[0])

        print('----sklearn----')
        ols_sk = LinearRegression()
        ols_sk.fit(x, y)
        print('y = %sx + %s' % (ols_sk.coef_[0], ols_sk.intercept_))
        prediction_sk = ols_sk.predict(np.array([[10]]))
        print('train_features = 10 时, y = %s' % prediction_sk[0])

        # 断言npml和sklearn的结果误差不超过1e-6
        assert abs(ols_sk.coef_ - ols.coefficient) < 1e-6
        assert abs(ols_sk.intercept_ - ols.intercept) < 1e-6
        assert abs(prediction[0] - prediction_sk[0]) < 1e-6

        # 测试绘图
        try:
            ols.plot(x, y)
        except Exception as exc:
            print(exc)
            assert False

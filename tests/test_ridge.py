import numpy as np

from npml.regress.ridge import Ridge
from sklearn.linear_model import Ridge as Ridge_SK


class TestRidge:
    def test_ridge(self):
        # 对比npml和sklearn的结果
        print('----npml----')
        ridge = Ridge()
        x = np.array([[1], [2], [3], [4]])
        y = np.array([10, 12, 14, 16])
        ridge.fit(x, y, alpha=0.1)
        print('y = %sx + %s' % (ridge.coefficient[0], ridge.intercept))
        prediction = ridge.predict(np.array([[10]]))
        print('train_features = 10 时, y = %s' % prediction[0])

        print('----sklearn----')
        ridge_sk = Ridge_SK(alpha=0.1)
        ridge_sk.fit(x, y)
        print('y = %sx + %s' % (ridge_sk.coef_[0], ridge_sk.intercept_))
        prediction_sk = ridge_sk.predict(np.array([[10]]))
        print('train_features = 10 时, y = %s' % prediction_sk[0])

        # 测试绘图
        try:
            ridge.plot(x, y)
        except Exception as exc:
            print(exc)
            assert False

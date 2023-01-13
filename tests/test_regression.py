import numpy as np
from sklearn.metrics import r2_score

from npml.model_evaluation.metrics.regression import r2

x = np.arange(100)
y = np.random.randn(100) + x


def test_r2():
    assert r2_score(x, y) == r2(x, y)

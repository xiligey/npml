import sklearn.metrics as sk_metrics
import npml.utils.metrics.classification as np_metrics
import numpy as np

y_true = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
y_pred = np.array([1, 1, 1, 2, 2, 2, 2, 2, 1, 1])
labels = np.array([1, 2])


def test_accuracy():
    ac1 = np_metrics.accuracy(y_true, y_pred)
    ac2 = sk_metrics.accuracy_score(y_true, y_pred)
    assert np.all(ac1 == ac2)


def test_precision():
    pr1 = np_metrics.precision(y_true, y_pred)
    pr2 = sk_metrics.precision_score(y_true, y_pred)
    assert np.all(pr1 == pr2)


def test_recall():
    re1 = np_metrics.recall(y_true, y_pred)
    re2 = sk_metrics.recall_score(y_true, y_pred)
    assert re1 == re2


def test_f1():
    f11 = np_metrics.f1(y_true, y_pred)
    f12 = sk_metrics.f1_score(y_true, y_pred)
    assert f11 == f12


def test_confusion_matrix():
    cm1 = np_metrics.confusion_matrix(y_true, y_pred, labels=labels)
    cm2 = sk_metrics.confusion_matrix(y_true, y_pred, labels=labels)
    assert np.all(cm1 == cm2)

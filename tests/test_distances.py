import numpy as np

from npml.utils.distances import euclidean_distance, manhattan_distance


def test_euclidean_distance():
    x1 = np.array([1, 2, 3, 4, 5])
    x2 = x1 + 1
    distance = euclidean_distance(x1, x2)
    assert distance == np.sqrt(5)

    x3 = np.array([[1], [2], [3], [4], [5]])
    x4 = x3 + 1
    distance = euclidean_distance(x3, x4)
    print(distance)


def test_manhattan_distance():
    x1 = np.array([1, 2, 3, 4, 5])
    x2 = x1 + 1
    assert manhattan_distance(x1, x2) == 5


def test_chebyshev_distance():
    assert False


def test_minkowski_distance():
    assert False


def test_standard_euclidean_distance():
    assert False


def test_mahalanobis_distance():
    assert False


def test_cosine_similarity():
    assert False

from npml.utils.distances import *

x1 = np.array([1, 2, 3, 4])
y1 = np.array([2, 3, 4, 5])
x2 = np.array([1, 1, 1, 1, 1, 1])
y2 = np.array([2, 3, 4, 5, 6, 4])
stds1 = np.array([0.1, 0.2, 0.1, 0.3])
stds2 = np.array([0.1, 0.2, 0.1, 0.3, 0.6, 0.7])
covariance_matrix1 = np.array([[1, 2, 3, 4], [2, 3, 2, 4], [1, 2, 1, 2], [2, 2, 3, 2]])


def test_euclidean_distance():
    assert euclidean_distance(x1, y1) == 2
    assert euclidean_distance(x2, y2) == 8


def test_manhattan_distance():
    assert manhattan_distance(x1, y1) == 4
    assert manhattan_distance(x2, y2) == 18


def test_chebyshev_distance():
    assert chebyshev_distance(x1, y1) == 1
    assert chebyshev_distance(x2, y2) == 5


def test_minkowski_distance():
    assert minkowski_distance(x1, y1, 1) == manhattan_distance(x1, y1)
    assert minkowski_distance(x2, y2, 1) == manhattan_distance(x2, y2)
    assert minkowski_distance(x1, y1, 2) == euclidean_distance(x1, y1)
    assert minkowski_distance(x2, y2, 2) == euclidean_distance(x2, y2)


def test_standard_euclidean_distance():
    assert standard_euclidean_distance(x1, y1, stds1) == 15.36590742882148
    assert standard_euclidean_distance(x2, y2, stds2) == 36.95388435822407


def test_mahalanobis_distance():
    assert mahalanobis_distance(x1, y1, covariance_matrix1) == 0.5773502691896257


def test_cosine_similarity():
    assert cosine_similarity(x1, y1) == 0.9938079899999065

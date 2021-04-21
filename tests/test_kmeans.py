import numpy as np

from npml.cluster.kmeans import KMeans


class TestKMeans:
    def test_kmeans(self):
        train_features = np.array([
            [1, 1], [1, 2], [2, 1], [2, 2],
            [1, 10], [2, 10], [2, 9], [2, 10],
            [10, 1], [10, 2], [11, 1], [11, 2],
            [9, 9], [9, 10], [10, 9], [10, 10]
        ])
        model = KMeans()
        model.fit(train_features, k=4)
        print(model.centroids)
        print(model.labels)
        print(model.predict(np.array([[1, 0], [11, 12]])))

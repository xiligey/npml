import numpy as np

from npml.classify.k_nearest_neighbors import KNN


class TestKNN:

    def test_knn(self):
        k = 3
        train_features = np.array([
            [1, 1], [1, 2], [2, 1], [2, 2],
            [1, 10], [1, 11], [2, 10], [2, 11],
            [10, 1], [10, 2], [11, 1], [11, 2],
            [10, 10], [10, 11], [11, 10], [11, 11]
        ])
        train_labels = np.array([
            0, 0, 0, 0,
            1, 1, 1, 1,
            2, 2, 2, 2,
            3, 3, 3, 3
        ])
        pred_features = train_features
        knn = KNN(k=k)
        knn.fit(train_features, train_labels)
        pred_labels = knn.predict(pred_features)
        print('训练集labels: %s' % train_labels)
        print('预测集labels %s' % pred_labels)
        # 断言 对训练集进行预测 预测的结果和训练集的标签完全一致
        assert (train_labels == pred_labels).all()

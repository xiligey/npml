import numpy as np

from npml.classify.naive_bayes import NaiveBayes


class TestNaiveBayes:
    def test_naive_bayes(self):
        train_features = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0]
        ])
        train_labels = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0])
        test_features = np.array([
            [0, 0, 0, 1],
            [0, 1, 0, 0]
        ])
        model = NaiveBayes()
        model.fit(train_features, train_labels)
        print(model.predict(test_features))

"""逻辑回归"""
import numpy as np


class LogisticRegression(object):
    def fit(self, X, y, max_iter=1e6, tol=1e-4, alpha=0.1):
        """训练
        Parameters
        ----------
            X: 训练样本 (n_samples, n_features)
            y: 标签 (n_samples)
            max_iter: 最大迭代次数
            tol: 参数变化最大误差
            alpha: 学习率
        """
        n_samples, n_features = X.shape
        self.theta = np.zeros((n_features + 1, 1))  # 初始化theta

        X = np.concatenate(
            (np.ones(n_samples).reshape((n_samples, 1)), X), axis=1)
        y = y.reshape((n_samples, 1))
        for i in range(int(max_iter)):
            theta_next = self.theta - alpha * \
                         X.T @ (1 / (1 + np.exp(-X @ self.theta)) - y) / n_samples
            print(theta_next)
            if np.abs(self.theta - theta_next).sum() < tol:
                self.theta = theta_next
                print("merge")
                break
            self.theta = theta_next
        else:
            print("get the max_iter, stop iter.")

        return self

    def predict(self, X):
        theta = self.theta.reshape(-1, 1)
        return 1 / (1 + np.exp(np.dot(-theta.T, X)))

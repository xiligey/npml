from sklearn.linear_model import LinearRegression

from linear_model.OrdinaryLeastSquare import OrdinaryLeastSquare

if __name__ == '__main__':
    a = LinearRegression()
    print(a._estimator_type)

    b = OrdinaryLeastSquare()

    print(b.model_type)


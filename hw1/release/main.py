import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger


class LinearRegressionBase:
    def __init__(self):
        self.weights = None
        self.intercept = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        # Add one column to represent the intercept
        X = np.c_[np.ones(X.shape[0]), X]
        # Calculate weights using the Close-form Equation
        self.weights = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.intercept = self.weights[0]
        self.weights = self.weights[1:]

    def predict(self, X):
        return np.dot(X, self.weights) + self.intercept


class LinearRegressionGradientdescent(LinearRegressionBase):
    def __init__(self):
        super().__init__()
        self.loss_history = []

    def fit(self, X, y, learning_rate=0.0001, epochs=800000, regularization_param=0.1):
        # Initialize weights and intercept
        m, n = X.shape
        self.weights = np.zeros(n)
        self.intercept = 0.0

        # Reshape y to ensure that the shape of y is one dimension
        y = y.reshape(-1)
        for epoch in range(epochs):
            y_pred = self.predict(X)

            gradient_weights = -(2 / m) * np.dot(X.T, (y - y_pred)) + regularization_param * np.sign(self.weights)
            gradient_intercept = -(2 / m) * np.sum(y - y_pred)

            self.weights -= learning_rate * gradient_weights
            self.intercept -= learning_rate * gradient_intercept

            loss = np.mean((y_pred - y) ** 2) + regularization_param * np.sum(np.abs(self.weights))

            self.loss_history.append(loss)

            if (epoch % (epochs / 10) == 0):
                logger.info(f'{epoch}/{epochs}: loss={loss}')

        return self.loss_history

    def predict(self, X):
        return np.dot(X, self.weights) + self.intercept

    def plot_learning_curve(self, losses):
        plt.plot(losses)
        plt.title('Learning Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig("learning_curve.png")


def compute_mse(prediction, ground_truth):
    return np.mean((prediction - ground_truth) ** 2)


def main():
    train_df = pd.read_csv('./train.csv')
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()
    train_y = train_df["Performance Index"].to_numpy()
    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)
    logger.info(f'{LR_CF.weights=}, {LR_CF.intercept=:.4f}')

    LR_GD = LinearRegressionGradientdescent()
    losses = LR_GD.fit(train_x, train_y, learning_rate=0.00015, epochs=700000)
    LR_GD.plot_learning_curve(losses)
    logger.info(f'{LR_GD.weights=}, {LR_GD.intercept=:.4f}')

    test_df = pd.read_csv('./test.csv')
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).sum()
    logger.info(f'Prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = ((mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')


if __name__ == '__main__':
    main()

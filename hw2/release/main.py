import typing as t

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.metrics import roc_auc_score


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        """
        Implement your fitting function here.
        The weights and intercept should be kept in self.weights and self.intercept.
        """
        n_samples, n_features = inputs.shape
        self.weights = np.zeros(n_features)
        self.intercept = 0
        # Gradient descent
        for epoch in range(self.num_iterations):
            y_predicted = np.dot(inputs, self.weights) + self.intercept
            y_predicted = self.sigmoid(y_predicted)
            # Compute gradients
            dw = (1 / n_samples) * np.dot(inputs.T, (y_predicted - targets))
            db = (1 / n_samples) * np.sum(y_predicted - targets)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db

            loss = self.compute_loss(targets, y_predicted)

            if (epoch % (self.num_iterations / 10) == 0):
                logger.info(f'{epoch}/{self.num_iterations}: loss={loss}')

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Tuple[t.Sequence[np.float_], t.Sequence[int]]:
        """
        Implement your prediction function here.
        The return should contains
        1. sample probabilty of being class_1
        2. sample predicted class
        """
        linear_model = np.dot(inputs, self.weights) + self.intercept
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted, y_predicted_cls

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y_true, y_pred):
        # Implementing the cross-entropy loss
        return -np.mean(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))


class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        # Separate data into classes
        class0 = inputs[targets == 0]
        class1 = inputs[targets == 1]

        # Calculate means
        self.m0 = np.mean(class0, axis=0)
        self.m1 = np.mean(class1, axis=0)

        # Calculate within-class covariance matrix
        s0 = np.dot((class0 - self.m0).T, (class0 - self.m0))
        s1 = np.dot((class1 - self.m1).T, (class1 - self.m1))
        self.sw = s0 + s1

        # Calculate between-class covariance matrix
        mean_diff = (self.m0 - self.m1).reshape(-1, 1)
        self.sb = np.dot(mean_diff, mean_diff.T)

        # Calculate the Fisher's linear discriminant
        self.w = np.linalg.inv(self.sw).dot(self.m0 - self.m1)

        # Calculate slope and intercept for the projection line
        self.slope = self.w[1] / self.w[0]
        self.intercept = -self.slope * np.mean(inputs, axis=0)[0]

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Sequence[t.Union[int, bool]]:
        projected = inputs.dot(self.w)
        class0_mean = self.m0.dot(self.w)
        class1_mean = self.m1.dot(self.w)
        return np.array([0 if np.abs(p - class0_mean) < np.abs(p - class1_mean) else 1 for p in projected])

    def plot_projection(self, inputs: npt.NDArray[float]):
        plt.figure(figsize=(8, 8))
        predicts = self.predict(inputs)

        # Projection line
        x_values = np.linspace(-0.2, 1, 100)
        y_values = self.slope * x_values + self.intercept
        plt.plot(x_values, y_values)

        for i in range(len(predicts)):
            if predicts[i] == 0:
                # Input point
                plt.scatter(inputs[i][0], inputs[i][1], c='blue', alpha=0.5)

                # Projection point
                x_proj = (inputs[i][0] + self.slope * inputs[i][1] - self.slope * self.intercept) / (self.slope**2 + 1)
                y_proj = self.slope * x_proj + self.intercept
                plt.plot([inputs[i][0], x_proj], [inputs[i][1], y_proj], color='gray', linewidth=1)
                plt.scatter(x_proj, y_proj, c='blue', alpha=0.5)

            elif predicts[i] == 1:
                # Input point
                plt.scatter(inputs[i][0], inputs[i][1], c='red', alpha=0.5)

                # Projection point
                x_proj = (inputs[i][0] + self.slope * inputs[i][1] - self.slope * self.intercept) / (self.slope**2 + 1)
                y_proj = self.slope * x_proj + self.intercept
                plt.plot([inputs[i][0], x_proj], [inputs[i][1], y_proj], color='gray', linewidth=1)
                plt.scatter(x_proj, y_proj, c='red', alpha=0.5)

        plt.title(f'Projection line slope: {self.slope}, Intercept: {self.intercept}')
        plt.axis('equal')
        plt.savefig("test.png")
        plt.close()


def compute_auc(y_trues, y_preds) -> float:
    return roc_auc_score(y_trues, y_preds)


def accuracy_score(y_trues, y_preds) -> float:
    correct_count = np.sum(y_trues == y_preds)
    accuracy = correct_count / len(y_trues)
    return accuracy


def main():
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features)
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=1e-3,  # You can modify the parameters as you want
        num_iterations=70000,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['27', '30']  # Dont modify
    x_train = train_df[cols].to_numpy()
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()

    FLD_ = FLD()
    FLD_.fit(x_train, y_train)
    y_preds = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_preds)
    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')
    FLD_.plot_projection(x_test)


if __name__ == '__main__':
    main()

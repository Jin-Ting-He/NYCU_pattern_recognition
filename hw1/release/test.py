import numpy as np
from loguru import logger
from main import LinearRegressionCloseform, LinearRegressionGradientdescent
def sample_data():
    slope, intercept = 3, 4
    n_datapoints = 100
    xs = np.linspace(-100, 100, n_datapoints).reshape((n_datapoints, 1))
    ys = slope * xs + intercept
    return xs, ys

def test_regression_gd(sample_data):
    x, y = sample_data
    print(x.shape, y.shape)
    model = LinearRegressionGradientdescent()
    model.fit(x, y, learning_rate=1e-4, epochs=70000)

    logger.info(f'{model.weights=}, {model.intercept=}')

data = sample_data()
test_regression_gd(data)
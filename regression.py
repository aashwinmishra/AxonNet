"""
Nonlinear Regression example as a basic check.
"""
import numpy as np

from AxonNet.nn import NeuralNet
from AxonNet.layers import Linear, Tanh
from AxonNet.train import train

n = 128
W = np.array([[3.2, -0.5]]).T
x1 = np.linspace(-1, 1, n).reshape(-1, 1)
x2 = np.square(x1)
X = np.concatenate([x1, x2], axis=1)
noise = 0.25*np.random.randn(n, 1)
Y = X@W + 0.5 + noise


net = NeuralNet([
    Linear(input_dim=2, output_dim=16),
    Tanh(),
    Linear(input_dim=16, output_dim=1)
])

train(net, X, Y)

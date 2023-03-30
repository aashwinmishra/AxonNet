"""
Simple Regression example as a basic check.
"""
import numpy as np

from AxonNet.nn import NeuralNet
from AxonNet.layers import Linear, Tanh
from AxonNet.train import train

n = 128
W = 2.2
b = -0.5
x = np.linspace(-1, 1, n).reshape(-1, 1)
noise = np.random.randn(n, 1)
y = W*x + b + noise


net = NeuralNet([
    Linear(input_dim=1, output_dim=1)
])

train(net, x, y)
for layer in net.layers:
    print(layer.params["w"], layer.params["b"])


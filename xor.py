"""
Simple implementation of xor for a test
"""
import numpy as np

from AxonNet.nn import NeuralNet
from AxonNet.layers import Linear, Tanh
from AxonNet.train import train

inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])
targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

net = NeuralNet([
    Linear(input_dim=2, output_dim=8),
    Tanh(),
    Linear(input_dim=8, output_dim=2)
])

train(net, inputs, targets, num_epochs=10000)

for x, y in zip(inputs, targets):
    pred = net.forward(x)
    print(y, pred)

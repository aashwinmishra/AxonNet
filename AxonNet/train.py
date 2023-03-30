"""
Functions to train the NeuralNet instances
"""
import numpy as np
from AxonNet.tensor import Tensor
from AxonNet.nn import NeuralNet
from AxonNet.optim import Optimizer, SGD
from AxonNet.loss import Loss, MSE
from AxonNet.data import DataIterator, BatchIterator


def train(net: NeuralNet,
          inputs: Tensor,
          targets: Tensor,
          num_epochs: int = 500,
          iterator: DataIterator = BatchIterator(),
          loss: Loss = MSE(),
          optim: Optimizer = SGD(),
          lr: float = 0.001,
          verbose: bool = True) -> None:
    epoch_losses = []
    for epoch in range(num_epochs):
        batch_losses = []
        for batch in iterator(inputs, targets):
            xb, yb = batch.inputs, batch.targets
            yhat = net.forward(xb)
            batch_losses.append(loss.loss(yhat, yb))
            grad_upstream = loss.grad(yhat, yb)
            grad_downstream = net.backward(grad_upstream)
            optim.step(net)
        epoch_loss = np.mean(batch_losses).item()
        epoch_losses.append(epoch_loss)
        if verbose:
            print(f"Epoch: {epoch} \t Loss: {epoch_loss}")









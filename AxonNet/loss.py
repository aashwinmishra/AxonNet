"""
A metric evaluates the quality of a set of predictions.
A loss function is a differentiable metric.
"""
import numpy as np
from AxonNet.tensor import Tensor


class Loss:
    """Base Class"""
    def loss(self, predictions: Tensor, targets: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predictions: Tensor, targets: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    """Mean Squared Error"""
    def loss(self, predictions: Tensor, targets: Tensor) -> float:
        return 0.5*np.mean(np.square(predictions - targets)).item()

    def grad(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return (predictions - targets)/len(targets)



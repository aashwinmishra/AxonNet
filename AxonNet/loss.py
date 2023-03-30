"""
Loss: a [smooth] mapping from (Predictions, True Labels) to the Real Line.
In the calls, the default order is Predictions, and then True Labels.
"""
from AxonNet.tensor import Tensor
import numpy as np


class Loss:
    """Abstract base Loss class."""
    def loss(self, predictions: Tensor, truth: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predictions: Tensor, truth: Tensor) -> Tensor:
        raise NotImplementedError


class MSE(Loss):
    """Mean Squared Error between the predictions and the labels."""
    def loss(self, predictions: Tensor, truth: Tensor) -> float:
        return np.mean(np.square(predictions - truth)).item()

    def grad(self, predictions: Tensor, truth: Tensor) -> Tensor:
        return 2.0*(predictions - truth)/len(predictions)

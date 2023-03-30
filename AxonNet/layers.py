"""
Layer: Mapping defining a forward and a backward operation.
"""
from AxonNet.tensor import Tensor
import numpy as np
from typing import Dict, Callable


class Layer:
    """Abstract base Layer class"""
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}
        self.inputs: Tensor = None

    def forward(self, inputs: Tensor) -> Tensor:
        """Forward Mapping"""
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """BackPropagate the inputed gradient through the Layer"""
        raise NotImplementedError


class Linear(Layer):
    """ Computes output = x @ w + b.
    x is inputs of Tensor type and shape [batch size, input size]
    The output is a Tensor of shape [batch size, output size]
    """
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.zeros(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        """
        Let Y = f(X) and X = A @ B + C
        dY/dA = f'(X) @ B.T,
        dY/dB = A.T @ f'(X),
        dY/dC = f'(X) with summation across the batch axis.
        """
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T


F = Callable[[Tensor], Tensor]


class Activation(Layer):
    """
    Applies a function to the inputs in an element-wise manner.
    """
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        return self.f_prime(self.inputs) * grad


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)


def tanh_prime(x: Tensor) -> Tensor:
    return 1.0 - np.tanh(x)**2


class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)


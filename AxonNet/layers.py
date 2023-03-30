"""
A layer is a map.
The layer objects have a forward and a backward pass.
"""
import numpy as np
from typing import Dict, Callable
from AxonNet.tensor import Tensor


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}
        self.inputs: Tensor = None

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Layer):
    """
    y = x@W + b
    x = [batch_dim, input_dim]; y = [batch_dim, output_dim]
    """
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.params["w"] = np.random.randn(input_dim, output_dim)
        self.params["b"] = np.zeros(output_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return inputs@self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T


F = Callable[[Tensor], Tensor]


class Activation(Layer):
    """ A [non-linear] function applied element-wise to the input"""
    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad: Tensor) -> Tensor:
        return grad*self.f_prime(self.inputs)


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)


def tanh_prime(x: Tensor) -> Tensor:
    return 1.0 - np.square(np.tanh(x))


class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)



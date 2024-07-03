# Author: Mohammad Senan Ali

import numpy as np


class Activation:
    def __init__(self, activation, activation_prime):
        self.input = None
        self.output = None
        self.activation = activation
        self.activation_derivative = activation_prime

    def forward(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_gradient):
        return self.activation_derivative(self.input) * output_gradient


class Linear(Activation):
    def __init__(self) -> None:
        super().__init__(self.linear, self.linear_prime)

    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def linear_prime(x):
        return np.ones_like(x)


class Tanh(Activation):
    def __init__(self) -> None:
        super().__init__(self.tanh, self.tanh_prime)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_prime(x):
        return 1.0 - np.tanh(x) ** 2


class Sigmoid(Activation):
    def __init__(self) -> None:
        super().__init__(self.sigmoid, self.sigmoid_prime)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_prime(x):
        return Sigmoid.sigmoid(x) * (1 - Sigmoid.sigmoid(x))


class ReLU(Activation):
    def __init__(self) -> None:
        super().__init__(self.relu, self.relu_prime)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_prime(x):
        return np.where(x > 0, 1, 0)


class Softmax(Activation):
    def __init__(self):
        super().__init__(self.softmax, self.softmax_prime)

    def backward(self, output_gradient):
        return output_gradient @ self.activation_derivative(self.input)

    @staticmethod
    def softmax(x):
        exp_values = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=-1, keepdims=True)

    @staticmethod
    def softmax_prime(x):
        s = Softmax.softmax(x)
        return s * (np.eye(len(s[0])) - s.T)

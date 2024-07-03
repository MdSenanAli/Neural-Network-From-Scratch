# Author: Mohammad Senan Ali

import numpy as np
from activations import *
from loss import *
import matplotlib.pyplot as plt

np.random.seed(99)
# Generate a random 28x28 image with pixel values between 0 and 255
sample_image = np.random.randint(0, 256, size=(5, 5))

# Normalize the pixel values to be between 0 and 1
sample_image_normalized = sample_image / 255.0


class Convolutional:
    def __init__(
        self, input_shape, activation=Linear, kernel_size=3, output_channels=1
    ):
        input_depth, input_height, input_width = input_shape
        self.depth = output_channels
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (
            output_channels,
            input_height - kernel_size + 1,
            input_width - kernel_size + 1,
        )
        self.flatten = False
        self.kernels_shape = (output_channels, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

        self.activation = activation()

    def forward(self, input_forward):
        self.input = input_forward
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                # print("Printing: ", self.input.shape)
                self.output[i] += self.my_correlate(self.input[j], self.kernels[i, j])
                # self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j])

        if self.flatten:
            self.output, shape = self.flatten_func(self.output)

        self.output = self.activation.forward(self.output)

        return self.output

    def backward(self, output_gradient, learning_rate=0.01):
        output_gradient = self.activation.backward(output_gradient)

        if self.flatten:
            output_gradient = self.unflatten(output_gradient, self.output_shape)

        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = self.my_correlate(
                    self.input[j], output_gradient[i]
                )
                input_gradient[j] += self.my_convolve(
                    output_gradient[i], self.kernels[i, j], "full"
                )

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient

    def my_correlate(self, input, kernel, mode="valid"):
        """
        2D correlation function.
        """
        input_height, input_width = input.shape
        kernel_height, kernel_width = kernel.shape

        if mode == "valid":
            result_height = input_height - kernel_height + 1
            result_width = input_width - kernel_width + 1
        elif mode == "same":
            result_height = input_height
            result_width = input_width
        elif mode == "full":
            result_height = input_height + kernel_height - 1
            result_width = input_width + kernel_width - 1
        else:
            raise ValueError("Invalid mode. Use 'valid', 'same', or 'full'.")

        result = np.zeros((result_height, result_width))

        for i in range(result_height):
            for j in range(result_width):
                if mode == "valid":
                    result[i, j] = np.sum(
                        input[i : i + kernel_height, j : j + kernel_width] * kernel
                    )
                elif mode == "same":
                    i_start = max(0, i - kernel_height // 2)
                    i_end = min(input_height, i + kernel_height // 2 + 1)
                    j_start = max(0, j - kernel_width // 2)
                    j_end = min(input_width, j + kernel_width // 2 + 1)
                    result[i, j] = np.sum(
                        input[i_start:i_end, j_start:j_end]
                        * kernel[: i_end - i_start, : j_end - j_start]
                    )
                elif mode == "full":
                    i_start = max(0, i - kernel_height + 1)
                    i_end = min(input_height, i + 1)
                    j_start = max(0, j - kernel_width + 1)
                    j_end = min(input_width, j + 1)
                    result[i, j] = np.sum(
                        input[i_start:i_end, j_start:j_end]
                        * kernel[i_start - i : i_end - i, j_start - j : j_end - j]
                    )

        return result

    def my_convolve(self, output_gradient, kernel, mode="full"):
        """
        2D convolution function.

        Parameters:
            output_gradient (numpy.ndarray): Output gradient array.
            kernel (numpy.ndarray): Kernel or filter.
            mode (str): Mode of convolution. 'valid', 'same', or 'full'.

        Returns:
            numpy.ndarray: Convolution result.
        """
        # kernel = np.flipud(np.fliplr(kernel))
        output_height, output_width = output_gradient.shape
        kernel_height, kernel_width = kernel.shape

        if mode == "valid":
            result_height = output_height - kernel_height + 1
            result_width = output_width - kernel_width + 1
        elif mode == "same":
            result_height = output_height
            result_width = output_width
        elif mode == "full":
            result_height = output_height + kernel_height - 1
            result_width = output_width + kernel_width - 1
        else:
            raise ValueError("Invalid mode. Use 'valid', 'same', or 'full'.")

        result = np.zeros((result_height, result_width))

        for i in range(result_height):
            for j in range(result_width):
                if mode == "valid":
                    result[i, j] = np.sum(
                        output_gradient[i : i + kernel_height, j : j + kernel_width]
                        * kernel[::-1, ::-1]
                    )

                elif mode == "same":
                    i_start = max(0, i - kernel_height // 2)
                    i_end = min(output_height, i + kernel_height // 2 + 1)
                    j_start = max(0, j - kernel_width // 2)
                    j_end = min(output_width, j + kernel_width // 2 + 1)
                    result[i, j] = np.sum(
                        output_gradient[i_start:i_end, j_start:j_end]
                        * kernel[: i_end - i_start, : j_end - j_start]
                    )
                elif mode == "full":
                    i_start = max(0, i - kernel_height + 1)
                    i_end = min(output_height, i + 1)
                    j_start = max(0, j - kernel_width + 1)
                    j_end = min(output_width, j + 1)
                    result[i, j] = np.sum(
                        output_gradient[i_start:i_end, j_start:j_end]
                        * kernel[: i_end - i_start, : j_end - j_start]
                    )

        return result

    def flatten_func(self, input_conv):
        flattened_input = input_conv.flatten()
        return np.array(flattened_input), input_conv.shape

    def unflatten(self, flattened_input, shape):
        unflattend_output = flattened_input.reshape(shape)
        return unflattend_output

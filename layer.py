# Author: Mohammad Senan Ali

import numpy as np
from activations import *
from loss import *
from conv import *

# np.random.seed(42)


class NeuralNetwork:
    def __init__(self, layers, loss, seed=99) -> None:
        np.random.seed(seed)
        self.layers = layers
        self.loss = loss()

        self.convolution_layers = list()
        self.dense_layers = list()

    def initialise_temp(self, shape, learning_rate=0.001):
        for layer in self.layers:
            if isinstance(layer, Convolutional):
                self.convolution_layers.append(layer)
            else:
                self.dense_layers.append(layer)

        num_inputs = self.initialise_conv_layers(shape)
        for layer in self.dense_layers:
            layer.initialise(num_inputs, learning_rate)
            num_inputs = layer.output_size

        self.loss.initialise(num_inputs)

    def initialise_conv_layers(self, shape):
        if len(self.convolution_layers) != 0:
            self.convolution_layers[-1].flatten = True
            num_inputs = np.prod(self.convolution_layers[-1].output_shape)
            return num_inputs

        return shape[1]

    def initialise(self, num_inputs, learning_rate=0.001):
        for layer in self.layers:
            layer.initialise(num_inputs, learning_rate)
            num_inputs = layer.output_size
        self.loss.initialise(num_inputs)

    def fit_convolution(self, X, y, learning_rate=0.01, epochs=1000):
        self.initialise_temp(X[0], learning_rate)

        for epoch in range(epochs):

            # Num of Samples
            samples = len(X)
            # Variable to count epoch loss
            epoch_loss = 0

            # Running over all training points
            for i in range(samples):

                # Input to the next layer
                layer_output = X[i]
                # Forward pass
                for layer in self.layers:
                    # Store the layer output
                    layer_output = layer.forward(layer_output)
                    # print("Done")

                # Check for loss
                epoch_loss += self.loss.forward(layer_output, y[i])
                gradient_input = self.loss.backward()
                # print(gradient_input)
                for layer in reversed(self.layers):
                    gradient_output = layer.backward(gradient_input)
                    # print("Here Again")
                    gradient_input = gradient_output

            # if epoch % 1 == 0:
            print(f"Average Loss at epoch {epoch}: {epoch_loss/samples}")

    def fit(self, X, y, learning_rate=0.01, epochs=1000, reshape=True):
        # self.initialise(X.shape, learning_rate)
        self.initialise(X.shape[1], learning_rate)

        # Reshape X if required
        if reshape:
            X = X.reshape(X.shape[0], 1, X.shape[1])

        # Run the SGD Algorithm
        for epoch in range(epochs):

            # Num of Samples
            samples = len(X)
            # Variable to count epoch loss
            epoch_loss = 0

            # Running over all training points
            for i in range(samples):

                # Input to the next layer
                layer_output = X[i]
                # Forward pass
                for layer in self.layers:
                    # Store the layer output
                    layer_output = layer.forward(layer_output)

                # Check for loss
                epoch_loss += self.loss.forward(layer_output, y[i])
                gradient_input = self.loss.backward()

                for layer in reversed(self.layers):
                    gradient_output = layer.backward(gradient_input)
                    gradient_input = gradient_output

            # if epoch % 1 == 0:
            print(f"Average Loss at epoch {epoch}: {epoch_loss/samples}")

    def predict(self, X):
        output = []
        for i in range(len(X)):
            layer_output = X[i]
            # Forward pass
            for layer in self.layers:
                # Store the layer output
                layer_output = layer.forward(layer_output)

            output.append(layer_output)

        return np.vstack(output)

    def predict_conv(self, X):
        output = []
        for i in range(len(X)):
            layer_output = X[i]
            # Forward pass
            for layer in self.layers:
                # Store the layer output
                layer_output = layer.forward(layer_output)

            output.append(layer_output)

        return np.vstack(output)


class Layer:
    def __init__(self, output_size, activation=Linear):
        # Layer Input and Output
        self.input = None
        self.output = None

        # Trainable Parameters
        self.weights = None
        self.bias = None

        # Layer properties
        self.input_size = None
        self.output_size = output_size

        # Activation
        self.activation = activation()

        # Learning Rate (can be updated by the neural network class)
        self.learning_rate = None

    def initialise(self, input_size, learning_rate=0.01):
        # Storing the number of neurons from the previous layer
        self.input_size = input_size

        # Adding the learning rate
        self.learning_rate = learning_rate

        # Initialising the weights and biases
        self.weights = np.random.rand(input_size, self.output_size) - 0.5
        self.bias = np.random.rand(1, self.output_size) - 0.5

    def forward(self, input_data):
        # Storing the input data
        # self.input = input_data
        self.input = input_data.reshape(1, -1)

        # Getting the intermediate value XW + B
        intermediate = np.dot(self.input, self.weights) + self.bias

        # Getting the output
        self.output = self.activation.forward(intermediate)

        # Returning the output
        return self.output

    def backward(self, output_gradient):
        # Propagating the gradient back from the activation
        intermediate_gradient = self.activation.backward(output_gradient)
        # print("Intermediate Grad", intermediate_gradient)
        # Calculating the gradient of error w.r.t.  the Inputs
        input_gradient = np.dot(intermediate_gradient, self.weights.T)
        # print("Input Shape", self.input.shape)
        weights_gradient = np.dot(self.input.T, intermediate_gradient)
        # print("Weight Grad: ", weights_gradient)
        self.weights -= self.learning_rate * weights_gradient
        self.bias -= self.learning_rate * intermediate_gradient

        # Gradient w.r.t. Inputs for the previous layer
        return input_gradient


if __name__ == "__main__":
    pass

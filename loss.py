# Author: Mohammad Senan Ali

import numpy as np


class MeanSquaredLoss:
    def __init__(self) -> None:
        # True labels
        self.truth = None

        # Loss Output
        self.output = None

        # For storing the prediction
        self.prediction = None

        # Count of input neurons
        self.input_neurons = None

    def initialise(self, num_inputs):
        self.input_neurons = num_inputs

    def forward(self, y_pred, y_true):
        # Store the predictions
        self.truth = y_true
        self.prediction = y_pred

        # Calculate Loss
        self.output = np.mean(np.power(self.prediction - self.truth, 2))

        # Return Loss
        return self.output

    def backward(self):
        return 2 * (self.prediction - self.truth) / self.truth.size


class CrossEntropyLoss:
    def __init__(self):
        # True labels
        self.truth = None

        # Loss Output
        self.output = None

        # For storing the prediction
        self.prediction = None

        self.input_neurons = None

    def convert_one_hot(self, number):
        one_hot = np.zeros(self.input_neurons)
        one_hot[number] = 1
        return one_hot

    def initialise(self, num_inputs):
        self.input_neurons = num_inputs

    def forward(self, y_pred, y_true):
        # Storing the prediction
        self.prediction = y_pred

        # One hot conversion
        self.truth = self.convert_one_hot(y_true)

        # Calculate total loss
        self.output = np.sum(-self.truth * np.log(self.prediction))

        # Return Result
        return self.output

    def backward(self):
        return -self.truth / self.prediction

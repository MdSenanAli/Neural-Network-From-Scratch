# Author: Mohammad Senan Ali

from layer import *

import tensorflow as tf
from tensorflow.keras import datasets

# Load the MNIST dataset
(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

tr_shape = X_train.shape
te_shape = X_test.shape
X_train = X_train.reshape(tr_shape[0], 1, tr_shape[1], tr_shape[2])
X_test = X_test.reshape(te_shape[0], 1, te_shape[1], te_shape[2])
# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train, X_test = X_train[:100], X_test[:100]
y_train, y_test = y_train[:100], y_test[:100]
print(X_train.shape)
print(X_test.shape)

nn = NeuralNetwork(
    [
        Convolutional((1, 28, 28), ReLU, output_channels=1),
        Layer(10, Softmax),
    ],
    CrossEntropyLoss,
)
nn.fit_convolution(X_train, y_train, epochs=10)

output = nn.predict_conv(X_test)
pred = np.argmax(output, axis=1)
print(f"Accuracy: {np.sum(pred==y_test)/len(y_test)*100}")

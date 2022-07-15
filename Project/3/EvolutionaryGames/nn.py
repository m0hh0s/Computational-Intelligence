import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.w1 = np.random.randn(layer_sizes[1], layer_sizes[0])
        self.w2 = np.random.randn(layer_sizes[2], layer_sizes[1])
        self.b1 = np.zeros((layer_sizes[1], 1))
        self.b2 = np.zeros((layer_sizes[2], 1))

    def activation(self, x):
        return np.divide(1, np.add(1, np.exp(-x)))

    def forward(self, x):
        hidden_layer = self.activation(np.add(np.dot(self.w1, x), self.b1))
        output = self.activation(np.add(np.dot(self.w2, hidden_layer), self.b2))
        if output < 0.5:
            return -1
        else:
            return 1

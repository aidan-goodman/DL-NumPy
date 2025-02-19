import numpy as np
from funcs import *


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def perdict(self, x):
        w1, w2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        sigmoid = sigmoid_layer()
        softmax = softmax_layer()

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid.forward(a1)
        a2 = np.dot(z1, w2) + b2
        y = softmax.forward(a2)

        return y

    def loss(self, x, t):
        y = self.perdict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.perdict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads

    def gradient(self, x, t):
        w1, w2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        sigmoid = sigmoid_layer()
        softmax = softmax_layer()

        grads = {}

        batch_num = x.shape[0]

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid.forward(a1)
        a2 = np.dot(z1, w2) + b2
        y = softmax.forward(a2)

        dy = (y - t) / batch_num
        grads["W2"] = np.dot(z1.T, dy)
        grads["b2"] = np.sum(dy, axis=0)

        da1 = np.dot(dy, w2.T)
        dz1 = sigmoid.backward(da1)
        grads["W1"] = np.dot(x.T, dz1)
        grads["b1"] = np.sum(dz1, axis=0)

        return grads

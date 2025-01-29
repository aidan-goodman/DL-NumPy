import numpy as np


class sigmoid_layer:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        if self.out is None:
            raise Exception("Please forward first")
        dx = dout * (1.0 - self.out) * self.out
        return dx


class softmax_layer:
    def __init__(self):
        self.out = None

    def forward(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            out = y.T
        else:
            x = x - np.max(x)
            out = np.exp(x) / np.sum(np.exp(x))

        self.out = out
        return out

    def backward(self, dout):
        if self.out is None:
            raise Exception("Please forward first")
        dx = dout * (self.out - self.out.sum(axis=1, keepdims=True))
        return dx


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        # 计算正向步长的偏导数
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        # 计算反向步长的偏导数
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad

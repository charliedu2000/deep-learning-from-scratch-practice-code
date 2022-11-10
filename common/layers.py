import numpy as np

import sys, os

sys.path.append(os.curdir)
from common.func import cross_entropy_error, softmax


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)  # mask 是由 bool 值组成的 numpy 数组
        out = x.copy()
        out[self.mask] = 0  # 小于等于0的项置0
        return out

    def backward(self, dout):
        dout[self.mask] = 0  # x 小于等于0的项对应导数是0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 -
                     self.out) * self.out  # type: ignore 根据正向传播的输出计算反向传播的结果
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 损失
        self.y = None  # softmax 的输出
        self.t = None  # 监督数据（one-hot vector）

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size  # 单个数据的误差
        return dx
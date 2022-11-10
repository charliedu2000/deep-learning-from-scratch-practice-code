import sys, os
import numpy as np

sys.path.append(os.curdir)

from common.func import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self) -> None:
        self.W = np.random.randn(2, 3)  # 利用高斯分布随机生成0到1之间的数，填充指定形状的多维数组

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


net = simpleNet()
print('net.W:\n', net.W)  # net 的权重参数

x = np.array([0.6, 0.9])
p = net.predict(x)
print('predict x:\n', p)  # 输出
print('index of max value:\n', np.argmax(p))  # 最大值索引

t = np.array([0, 0, 1])
print('net.loss:\n', net.loss(x, t))  # 计算交叉熵损失

# W 是伪参数，计算梯度时会执行 f，用到 net 内的 W
# def f(W):
#     return net.loss(x, t)
# 也可以用 lambda 表达式
f = lambda w: net.loss(x, t)

dW = numerical_gradient(f, net.W)  # 求关于权重的梯度，权重自然是 f 的参数
print('dW:\n', dW)
import numpy as np
import matplotlib.pylab as plt


# 阶跃函数，支持 numpy 数组的实现
def step_function(x):
    y = x > 0  # 对 numpy 数组进行不等号运算后，数组中的各个元素都会进行相应运算，生成一个布尔型数组
    return y.astype(np.int32)  # 将布尔型数组转换为 int32 （np.int 已被弃用）


# 实现 sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # 由于 numpy 的广播1功能，该实现可以支持 numpy 数组


def relu(x):
    return np.maximum(0, x)

def identity_function(x):
    return x

# 把权重记为大写字母，其他如偏置和中间结果等记为小写字母
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

# 将输入信号转化为输出信号的过程（前向传播？）
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y) # [ 0.31682708 0.69627909]

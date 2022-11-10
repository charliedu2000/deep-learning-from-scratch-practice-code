import numpy as np


# 阶跃函数，支持 numpy 数组的实现
def step_function(x):
    y = x > 0  # 对 numpy 数组进行不等号运算后，数组中的各个元素都会进行相应运算，生成一个布尔型数组
    return y.astype(np.int32)  # 将布尔型数组转换为 int32 （np.int 已被弃用）


# 实现 sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # 由于 numpy 的广播功能，该实现可以支持 numpy 数组


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


# ReLU
def relu(x):
    return np.maximum(0, x)


# 恒等函数
def identity_function(x):
    return x


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)  # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


# def softmax(a):
#     c = np.max(a)
#     exp_a = np.exp(a - c)  # 溢出对策
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
#     return y


# 均方误差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)


# # 交叉熵误差
# def cross_entropy_error(y, t):
#     delta = 1e-7
#     return -np.sum(t * np.log(y + delta))  # 防止出现 log(0) 无限大

# 交叉熵误差 mini-batch
# def cross_entropy_error(y, t):
# if y.ndim == 1:
#     t = t.reshape(1, t.size)
#     y = y.reshape(1, y.size)


# batch_size = y.shape[0]
# return -np.sum(t * np.log(y + 1e-7)) / batch_size
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 将独热编码转换为对应的正确标签
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# 交叉熵误差 mini-batch，监督数据不是独热编码形式
# def cross_entropy_error_t_no_one_hot(y, t):
#     if y.ndim == 1:
#         t = t.reshape(1, t.size)
#         y = y.reshape(1, y.size)

#     batch_size = y.shape[0]
#     return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# 数值微分求导
def numerical_diff(f, x):
    h = 1e-4  # 0.0001，避免使用过小的值
    return (f(x + h) - f(x - h)) / (2 * h)  # 使用中心差分来计算


# 求梯度
def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # 生成和 x 形状相同、元素都为0的数组

    for idx in range(x.size):
        tmp_val = x[idx]
        # 计算 f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # 计算 f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        # 计算偏导，还原 x
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad
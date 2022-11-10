import numpy as np
import matplotlib.pylab as plt


# 阶跃函数，支持 numpy 数组的实现
def step_function(x):
    y = x > 0  # 对 numpy 数组进行不等号运算后，数组中的各个元素都会进行相应运算，生成一个布尔型数组
    return y.astype(np.int32)  # 将布尔型数组转换为 int32 （np.int 已被弃用）


# 实现 sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # 由于 numpy 的广播功能，该实现可以支持 numpy 数组

# ReLU
def relu(x):
    return np.maximum(0, x)

# 恒等函数
def identity_function(x):
    return x


x = np.array([-1, 1, 0])
print(x.shape)  # shape 的结果是元组
print(np.ndim(x))
print(step_function(x))

# 矩阵乘法，A 的列数应等于 B 的行数
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(np.dot(A, B))
A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([7, 8])
print(np.dot(A, B))

# 实现神经网络
x = np.array([1, 2])  # x1 x2
w = np.array([[1, 3, 5], [2, 4, 6]])  # 权重
print(x)
print(x.shape)
print(w)
print(w.shape)
print(np.dot(x, w))

# x = np.arange(-5.0, 5.0, 0.1) # 在 -5.0 到 5.0 的范围内，以 0.1 为单位生成 numpy 数组
# y = step_function(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1) # y 轴范围
# plt.show()

x = np.arange(-5.0, 5.0, 0.1)  # 在 -5.0 到 5.0 的范围内，以 0.1 为单位生成 numpy 数组
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # y 轴范围
plt.show()

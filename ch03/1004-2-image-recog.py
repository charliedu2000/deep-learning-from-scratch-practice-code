import sys, os
sys.path.append(os.curdir)
# print(sys.path)
from dataset.mnist import load_mnist
import pickle
import numpy as np

# 实现 sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # 由于 numpy 的广播功能，该实现可以支持 numpy 数组

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

# 获取测试数据
# 对数据进行了正规化处理（属于预处理），使数据的值在0.0～1.0的范围内
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

# 读入学习到的权重参数
def init_network():
    # rb, read as text, binary
    with open("ch0203/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

# 前向传播推理过程
def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)
    return y

x, t = get_data()
network = init_network()
accuracy_cnt = 0  # 识别精度
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y) #  取概率最高的元素的索引
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy: " + str(float(accuracy_cnt) / len(x)))
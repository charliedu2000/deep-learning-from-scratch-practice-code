import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.curdir)

from ch04.two_layer_net import TwoLayerNet
from dataset.mnist import load_mnist

# print('Load mnist dataset...')
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True,
                                                  one_hot_label=True)

# print('Initialize network...')
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 超参数
iters_num = 10000  # 梯度法更新次数
train_size = x_train.shape[0]  # 训练集大小
batch_size = 100  # batch 大小
learning_rate = 0.1  # 学习率

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 平均每个 epoch 的重复次数
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 获取 mini-batch
    # print(i, ': choose mini-batch...')
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    # print(i, ': calculate grads...')
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)  # 快速版本

    # 更新参数
    # print(i, ': update params...')
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 记录学习过程
    loss = network.loss(x_batch, t_batch)
    # print(i, 'loss: ', loss)
    train_loss_list.append(loss)

    # 每个 epoch 完成后计算识别精度
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('Train acc, test acc | ' + str(train_acc) + ', ' + str(test_acc))

# 画出图像
# markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))  # 生成 x 坐标
plt.plot(x, train_acc_list, label='train acc')  # 训练集识别精度
plt.plot(x, test_acc_list, label='test acc', linestyle='--')  # 测试集识别精度，虚线
plt.xlabel("epochs")  # 横轴单位
plt.ylabel("accuracy")  # 纵轴单位
plt.ylim(0, 1.0)
plt.legend(loc='lower right')  # 右下角图例
plt.show()

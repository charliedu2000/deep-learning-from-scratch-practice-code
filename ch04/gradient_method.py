import numpy as np
import sys
import os

sys.path.append(os.curdir)

from ch04.gradient_2d import numerical_gradient


def function_2(x):
    return x[0]**2 + x[1]**2


# f: 函数
# init_x: 初始值
# lr: 学习率
# step_num： 梯度法重复次数
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):  # 反复执行更新
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))

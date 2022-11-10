import numpy as np


def _numerical_gradient_no_batch(f, x):
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

        # 算出偏导，还原 x
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)  # 1维输出，直接计算
    else:
        grad = np.zeros_like(X)

        # 对于一组输出（2维），对每个输出计算梯度，组成梯度数组
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=[['readwrite']])
    while not it.finished:  # 遍历
        idx = it.multi_index  # 多维数组中的索引？
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原元素
        it.iternext()  # 继续

    return grad
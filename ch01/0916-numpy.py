import numpy as np

x = np.array([1.0, 2.0, 3.0])  # 一维数组
print(x)
print(x.shape)  # 形状
print(type(x))  # 类型
print(x.dtype)  # 元素类型
print(x + x)  # 对应元素的加法
print(x * 4)  # 数组与标量运算，标量会被“广播”成数字的形状

x = np.array([[51, 55],
              [14, 19],
              [0, 4]])  # 矩阵
print(x)
print(x.shape)
print(x[0])  # 首行
print(x[0][1])  # (0, 1) 位置的元素
for row in x:  # 遍历
    for ele in row:
        print(ele)
    print(row)

x = x.flatten()  # 将 x 转化为一维数组
print(x)
print(x[np.array([0, 2, 4])])  # 获取下标为 0 2 4 的元素
print(x > 15)
print(x[x > 15])

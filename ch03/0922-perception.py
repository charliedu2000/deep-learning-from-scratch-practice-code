import numpy as np
# 感知机实现与门
# def AND(x1, x2):
#     w1, w2, theta = 0.5, 0.5, 0.7 # 权重和阈值
#     tmp = x1*w1 + x2*w2
#     if tmp <= theta:
#         return 0
#     else:
#         return 1
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7 # 将阈值的相反数设置为偏置
    temp = np.sum(w*x) + b
    if temp <= 0:
        return 0
    else:
        return 1

# 感知机实现与非门
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    temp = np.sum(w*x) + b
    if temp <= 0:
        return 0
    else:
        return 1

# 感知机实现或门
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    temp = np.sum(w*x) + b
    if temp <= 0:
        return 0
    else:
        return 1

# 异或门，2层感知机
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print(AND(0, 0))
print(AND(0, 1))
print(AND(1, 0))
print(AND(1, 1))
print(NAND(0, 0))
print(NAND(0, 1))
print(NAND(1, 0))
print(NAND(1, 1))
print(OR(0, 0))
print(OR(0, 1))
print(OR(1, 0))
print(OR(1, 1))
print(XOR(0, 0))
print(XOR(0, 1))
print(XOR(1, 0))
print(XOR(1, 1))
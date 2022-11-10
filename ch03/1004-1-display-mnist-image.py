import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
# Python Image Library
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))  # 从 numpy 数组读取
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# 训练图像的第一张
img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)  # 更改为原来的尺寸
print(img.shape)

img_show(img)

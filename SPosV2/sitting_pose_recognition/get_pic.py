from PIL import Image
import numpy as np

# 打开图片
img = Image.open('./pic1.png')

# 将图片转换为numpy数组
img_array = np.array(img)
img_array1 = img_array.copy()

gap = 35

# for i in range(len(img_array) - gap):
#     for j in range(len(img_array[i])):
#         img_array1[i + gap][j] = img_array[i][j]

# for i in range(gap):
#     for j in range(len(img_array[i])):
#         img_array1[i][j] = img_array[0][0]

for i in range(len(img_array)):
    for j in range(0, len(img_array[0]) - gap):
        img_array1[i][j] = img_array[i][j + gap]

for i in range(len(img_array)):
    for j in range(len(img_array[0]) - gap, len(img_array[0])):
        img_array1[i][j] = img_array[0][0]

img = Image.fromarray(img_array1)
img.save('pic2.png')

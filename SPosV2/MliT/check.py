# import argparse
# import cv2
# import matplotlib.pyplot as plt
#
# parser = argparse.ArgumentParser(description='Run keypoint detection')
# parser.add_argument("--device", default="cpu", help="Device to inference on")
# parser.add_argument("--image_file", default="E:\desktop\classify_sitting_three\OpenPose//1.jpg", help="Input image")
#
# args = parser.parse_args()
#
# image_raw = cv2.imread(args.image_file)
#
# # cv2.namedWindow('IMG')
# # cv2.imshow("IMG", image_raw)
# # cv2.waitKey()
# # cv2.destroyAllWindows()
#
# print(image_raw.shape)


# import visdom
# import numpy as np
# viz = visdom.Visdom(server='http://localhost', port=8097, env='liyunfei')
# # 一张图片
# viz.image(
#     np.random.rand(3, 512, 256),
#     opts=dict(title='Random!', caption='How random.'),
# )
# # 多张图片
# viz.images(
#     np.random.randn(20, 3, 64, 64),
#     nrow=5,
#     opts=dict(title='Random images', caption='How random.')
# )

from tqdm import tqdm
import time

d = {'loss': 0.2, 'learn': 0.8}
for i in tqdm(range(50), desc='进行中', ncols=10, postfix=d):  # desc设置名称,ncols设置进度条长度.postfix以字典形式传入详细信息
    time.sleep(0.1)
    pass

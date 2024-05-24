import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

epochs = 80
lowess = sm.nonparametric.lowess

with open('./graph/whole_VIT_CosineAnnealingLR(1.12)_1_90.0_file_train_acc.txt', 'r') as file:
    lines = file.readlines()
VIT_S_acc = [float(line.strip()) for line in lines]

with open('./graph/whole_VIT_CosineAnnealingLR(1.12)_2_89.16666666666667_file_train_acc.txt', 'r') as file:
    lines = file.readlines()
VIT_acc = [float(line.strip()) for line in lines]

with open('./graph/whole_resnet101_CosineAnnealingLR(1.12)_2_86.66666666666667_file_train_acc.txt', 'r') as file:
    lines = file.readlines()
Resnet18_acc = [float(line.strip()) for line in lines]

with open('./graph/whole_resnet101_CosineAnnealingLR(1.12)_1_91.38888888888889_file_train_acc.txt', 'r') as file:
    lines = file.readlines()
Resnet101_acc = [float(line.strip()) for line in lines]

with open('./graph/whole_vgg16_CosineAnnealingLR(1.12)_2_89.44444444444444_file_train_acc.txt', 'r') as file:
    lines = file.readlines()
VGG16_acc = [float(line.strip()) for line in lines]

with open('./graph/whole_vgg19_CosineAnnealingLR(1.12)_2_83.88888888888889_file_train_acc.txt', 'r') as file:
    lines = file.readlines()
VGG19_acc = [float(line.strip()) for line in lines]

with open('./graph/whole_pit_CosineAnnealingLR(1.12)_2_83.05555555555556_file_train_acc.txt', 'r') as file:
    lines = file.readlines()
pit_acc = [float(line.strip()) for line in lines]

with open('./graph/whole_VIT-S_CosineAnnealingLR_24(1.12)_2_88.61111111111111_file_train_acc.txt', 'r') as file:
    lines = file.readlines()
cait_acc = [float(line.strip()) for line in lines]

with open('./graph/whole_VIT-S_CosineAnnealingLR_24(1.12)_3_86.38888888888889_file_train_acc.txt', 'r') as file:
    lines = file.readlines()
mobile_acc = [float(line.strip()) for line in lines]
x = []
for i in range(1, epochs + 1):
    x.append(i)

# plt.figure(1)
plt.title('val acc')
plt.axis([1, epochs, 0, 100])
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.ylim(70, 100)
VIT_S_smooth_acc = lowess(VIT_S_acc, x, frac=1. / 3.)[:, 1]
VIT_smooth_acc = lowess(VIT_acc, x, frac=1. / 3.)[:, 1]
VGG16_smooth_acc = lowess(VGG16_acc, x, frac=1. / 3.)[:, 1]
VGG19_smooth_acc = lowess(VGG19_acc, x, frac=1. / 3.)[:, 1]
pit_smooth_acc = lowess(pit_acc, x, frac=1. / 3.)[:, 1]
cait_smooth_acc = lowess(cait_acc, x, frac=1. / 3.)[:, 1]
Resnet101_smooth_acc = lowess(Resnet101_acc, x, frac=1. / 3.)[:, 1]
Resnet18_smooth_acc = lowess(Resnet18_acc, x, frac=1. / 3.)[:, 1]
MobileNet_smooth_acc = lowess(mobile_acc, x, frac=1. / 3.)[:, 1]

plt.plot(x, VIT_S_smooth_acc, label="MliT", color="red", linewidth=2)
plt.plot(x, VIT_smooth_acc, label="ViT", color="blue", linewidth=2)
plt.plot(x, cait_smooth_acc, label="CaiT", color="orange", linewidth=2)
plt.plot(x, pit_smooth_acc, label="PiT", color="brown", linewidth=2)
plt.plot(x, Resnet101_smooth_acc, label="ResNet101", color="pink", linewidth=2)
plt.plot(x, Resnet18_smooth_acc, label="ResNet18", color="gray", linewidth=2)
plt.plot(x, VGG16_smooth_acc, label="VGG16", color="green", linewidth=2)
plt.plot(x, VGG19_smooth_acc, label="VGG19", color="yellow", linewidth=2)
plt.plot(x, MobileNet_smooth_acc, label="MobileNet", color="purple", linewidth=2)

plt.legend(loc='lower right')
plt.savefig('./graph/' + 'accuracy.png')

# plt.figure(2)
# plt.xlabel("epoch")
# plt.ylabel("loss")
# # plt.plot(x, train_loss, label="train loss", color="r", linestyle="-", linewidth=1)
# smooth_train_loss = lowess(train_loss, x, frac=1. / 3.)[:, 1]
# plt.plot(x, smooth_train_loss, label="smooth train loss", color="g", linestyle="dashed", linewidth=3)
#
# # plt.plot(x, test_loss, label="val loss", color="b", linestyle="-", linewidth=1)
# smooth_test_loss = lowess(val_loss, x, frac=1. / 3.)[:, 1]
# plt.plot(x, smooth_test_loss, label="smooth val loss", color="fuchsia", linestyle="dashed", linewidth=3)

# plt.legend(loc=1)
# plt.savefig('./graph/' + model_name + '_acc.png')
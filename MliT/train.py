import numpy as np
import torch
import torchvision
import os
from config import config
import Model
from torch import optim, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datasets import *
from test import *
from utils.utils import *
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import datetime
from test import test, get_files
from torch.optim.lr_scheduler import *

lowess = sm.nonparametric.lowess

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == '__main__':
    date_start = pd.to_datetime(datetime.datetime.now())

    # 1. 定义模型
    # 'vgg16' or 'resnet18' or 'resnet101' or 'GoogLeNet' or 'VIT' or 'MliT_whole' or 'cait' or'MobilNet'
    model = Model.get_net('MliT_whole')
    if torch.cuda.is_available():
        model = model.cuda()

    # 2.创建文件夹
    if not os.path.exists(config.example_folder):
        os.mkdir(config.example_folder)
    if not os.path.exists(config.weights):
        os.mkdir(config.weights)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)

    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
    # optimizer = optim.Adagrad(model.parameters(), lr=config.lr, initial_accumulator_value=0, eps=1e-10)
    # optimizer = optim.RMSprop(model.parameters(), lr=config.lr, alpha=0.99, eps=1e-08, momentum=0.9, centered=False)
    # optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False, maximize=False)
    criterion = nn.CrossEntropyLoss().cuda()

    # 3.是否需要加载checkpoints 训练
    start_epoch = 0
    current_accuracy = 0
    resume = False  # false不加载模型
    if resume:
        checkpoint = torch.load('./checkpoints/half_colored_VIT_1e-3_640(2023.6.9)(93.2).pth')
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    # 4. 定义训练集 测试集
    crop = config.img_width * 0.9, config.img_height * 0.9
    transform = transforms.Compose([
        transforms.RandomResizedCrop((config.img_width, config.img_height)),  # 随即裁剪至指定大小
        # transforms.ColorJitter(0.05, 0.05, 0.05),  # 对比度
        transforms.RandomRotation(10),  # 随即旋转++
        # transforms. RandomGrayscale(p = 0.5),   # 转灰度图
        # transforms.Resize((config.img_width, config.img_height)),  # 改为随机裁剪++
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.07392877, 0.07392877, 0.07392877],  # four
        #                      std=[0.24807487, 0.24807487, 0.24807487])])

        transforms.Normalize(mean=[0.2675, 0.354, 0.2407],
                             std=[0.3253, 0.4063, 0.2736])])  # 全身

        # transforms.Normalize(mean=[0.359, 0.3125, 0.2261],
        #                      std=[0.3723, 0.3791, 0.2471])])  # 半身

    # transform = transforms.Compose([transforms.ToTensor()])
    # _, train_list = divide_data(config.data_folder,config.ratio)
    _, train_list = get_files(config.data_folder, config.ratio)
    # train_data = DataLoader(input_data)
    train_loader = DataLoader(datasets(train_list, transform=transform), batch_size=config.batch_size, shuffle=True,
                              collate_fn=collate_fn,
                              pin_memory=False, num_workers=6)

    # 测试集 不要数据增强 transform = None
    # test_list, _=divide_data(config.data_folder,config.ratio)
    test_list, _ = get_files(config.data_folder, config.ratio)
    test_loader = DataLoader(datasets(test_list, transform=None), batch_size=config.batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=6)

    # 设置动态变换的学习率 lr每经过50个epoch 就变为原来的0.4倍
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.3)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    train_loss = []
    acc = []
    test_loss = []

    len_train = len(train_loader.dataset)

    # 5. 开始训练
    print("------ Start Training ------\n")
    for epoch in range(start_epoch, config.epochs):
        model.train()

        # current_lr = lr_step(epoch, config.lr)

        # optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
        # optimizer = optim.Adagrad(model.parameters(), lr=config.lr, initial_accumulator_value=0, eps=1e-10)
        # optimizer = optim.RMSprop(model.parameters(), lr=config.lr, alpha=0.99, eps=1e-08, momentum=0.9, centered=False)
        # optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False, maximize=False)

        criterion = nn.CrossEntropyLoss().cuda()
        loss_epoch = 0
        scheduler.step()
        train_times = 0
        for index, (input, target) in enumerate(train_loader):
            model.train()
            train_times += 1
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            output = model(input)

            loss = criterion(output, target)

            # train_loss.append(loss)  # no
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            if (index + 1) % 10 == 0:
                print("Epoch: {} [{:>3d}/{}]\t Loss: {:.6f} ".format(epoch + 1, index * config.batch_size,
                                                                     len_train, loss.item()))
        if (epoch + 1) % 1 == 0:
            print("\n------ Evaluate ------")
            model.eval()
            # evaluate the model on the test data
            test_loss1, accTop1 = evaluate(test_loader, model, criterion)
            acc.append(accTop1)

            test_loss.append(test_loss1)

            train_loss.append(loss_epoch / len_train)  # 训练损失=总损失/训练loader

            print("Test_epoch: {} Test_accuracy: {:.4}% Test_Loss: {:.6f}".format(epoch + 1, accTop1, test_loss1))
            print("\r\n")

            # accuracy_y.append(accTop1)
            # loss_y.append(test_loss1)
            # x.append(epoch + 1)

            save_model = accTop1 > current_accuracy  # 测试的准确率大于当前准确率为True
            accTop1 = max(current_accuracy, accTop1)
            current_accuracy = accTop1
            # print(current_accuracy)
            if save_model:
                save_checkpoint({
                    "epoch": epoch + 1,
                    "model_name": config.model_name,
                    "state_dict": model.state_dict(),
                    "accTop1": current_accuracy,
                    "optimizer": optimizer.state_dict(),
                }, save_model)

    file_train_acc = open("./graph/" + config.model_name + "_file_train_acc.txt", 'w+')
    # file_train_loss = open("./graph/" + config.model_name + "_file_train_loss.txt", 'w+')
    # file_val_loss = open("./graph/" + config.model_name + "_file_val_loss.txt", 'w+')


    for num in acc:
        file_train_acc.write(str(num) + '\n')

    # for num in test_loss:
    #     file_val_loss.write(str(num) + '\n')
    #
    # for num in train_loss:
    #     file_train_loss.write(str(num) + '\n')

    # # 准确率可视化作图
    # x = []
    # for i in range(1, config.epochs + 1):
    #     x.append(i)
    #
    # plt.figure(1)
    # plt.axis([1, config.epochs, 0, 100])
    # plt.xlabel("epoch")
    # plt.ylabel("accuracy")
    # plt.plot(x, acc, label="accuracy", color="r", linestyle="-", linewidth=1)  # 画图
    # smooth_acc = acc
    # smooth_acc = lowess(smooth_acc, x, frac=1. / 3.)[:, 1]
    # plt.plot(x, smooth_acc, label="accuracy", color="g", linestyle="dashed", linewidth=3)
    # plt.legend(loc=1)
    # plt.savefig('./graph/' + config.model_name + '_accuracy.png')
    #
    # plt.figure(2)
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # # plt.plot(x, train_loss, label="train loss", color="r", linestyle="-", linewidth=1)
    # smooth_train_loss = lowess(train_loss, x, frac=1. / 3.)[:, 1]
    # plt.plot(x, smooth_train_loss, label="smooth train loss", color="g", linestyle="dashed", linewidth=3)
    #
    # # plt.plot(x, test_loss, label="val loss", color="b", linestyle="-", linewidth=1)
    # smooth_test_loss = lowess(test_loss, x, frac=1. / 3.)[:, 1]
    # plt.plot(x, smooth_test_loss, label="smooth test loss", color="fuchsia", linestyle="dashed", linewidth=3)
    #
    # plt.legend(loc=1)
    # plt.savefig('./graph/' + config.model_name + '_loss.png')

    file_train_acc.close()
    # file_train_loss.close()
    # file_val_loss.close()
    date_end = pd.to_datetime(datetime.datetime.now())  # Timestamp('2021-05-19 08:06:08.683355')
    date_start_end = date_end - date_start
    print(date_start_end)

    #  测试
    test_list, _ = get_files(config.test_data_folder, 1)
    test_loader = DataLoader(datasets(test_list, transform=None, test=True), batch_size=1, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)  # 测试时这里的batch_size = 1
    checkpoint = torch.load('./checkpoints/' + config.model_name + '.pth')
    model.load_state_dict(checkpoint["state_dict"])
    print("Start Test.......")
    res = test(test_loader, model)
    os.rename('./checkpoints/' + config.model_name + '.pth', './checkpoints/' + config.model_name + res + '.pth')
    os.rename('./graph/' + config.model_name + '_file_train_acc.txt', './graph/' + config.model_name + res + '_file_train_acc.txt')
    # os.rename('./graph/' + config.model_name + '_file_train_loss.txt', './graph/' + config.model_name + res + '_file_train_loss.txt')
    # os.rename('./graph/' + config.model_name + '_file_val_loss.txt', './graph/' + config.model_name + res + '_file_val_loss.txt')
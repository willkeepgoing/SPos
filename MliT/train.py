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

    # 'vgg16' or 'resnet18' or 'resnet101' or 'GoogLeNet' or 'VIT' or 'MliT_whole' or 'cait' or'MobilNet'
    model = Model.get_net('MliT_whole')
    if torch.cuda.is_available():
        model = model.cuda()

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

    # necessary or not to load checkpoints training
    start_epoch = 0
    current_accuracy = 0
    resume = False
    if resume:
        checkpoint = torch.load('./checkpoints/half_colored_VIT_1e-3_640(2023.6.9)(93.2).pth')
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    crop = config.img_width * 0.9, config.img_height * 0.9
    transform = transforms.Compose([
        transforms.RandomResizedCrop((config.img_width, config.img_height)),  # Random cropping
        # transforms.ColorJitter(0.05, 0.05, 0.05),  # contrast ratio
        transforms.RandomRotation(10),  # Random rotation
        # transforms. RandomGrayscale(p = 0.5),   # Convert to grayscale image
        # transforms.Resize((config.img_width, config.img_height)),  # Random cropping
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2675, 0.354, 0.2407],
                             std=[0.3253, 0.4063, 0.2736])])
    # transform = transforms.Compose([transforms.ToTensor()])
    # _, train_list = divide_data(config.data_folder,config.ratio)
    _, train_list = get_files(config.data_folder, config.ratio)
    # train_data = DataLoader(input_data)
    train_loader = DataLoader(datasets(train_list, transform=transform), batch_size=config.batch_size, shuffle=True,
                              collate_fn=collate_fn,
                              pin_memory=False, num_workers=6)

    # Test set transform = None
    # test_list, _=divide_data(config.data_folder,config.ratio)
    test_list, _ = get_files(config.data_folder, config.ratio)
    test_loader = DataLoader(datasets(test_list, transform=None), batch_size=config.batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=6)

    # Set the learning rate for dynamic transformations
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.3)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    train_loss = []
    acc = []
    test_loss = []

    len_train = len(train_loader.dataset)

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

            train_loss.append(loss_epoch / len_train)  # Training loss=total loss/training loader

            print("Test_epoch: {} Test_accuracy: {:.4}% Test_Loss: {:.6f}".format(epoch + 1, accTop1, test_loss1))
            print("\r\n")

            save_model = accTop1 > current_accuracy  # The accuracy of the test is greater than the current accuracy, which is True
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

    file_train_acc.close()
    # file_train_loss.close()
    # file_val_loss.close()
    date_end = pd.to_datetime(datetime.datetime.now())  # Timestamp('2021-05-19 08:06:08.683355')
    date_start_end = date_end - date_start
    print(date_start_end)

    #  test
    test_list, _ = get_files(config.test_data_folder, 1)
    test_loader = DataLoader(datasets(test_list, transform=None, test=True), batch_size=1, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)
    checkpoint = torch.load('./checkpoints/' + config.model_name + '.pth')
    model.load_state_dict(checkpoint["state_dict"])
    print("Start Test.......")
    res = test(test_loader, model)
    os.rename('./checkpoints/' + config.model_name + '.pth', './checkpoints/' + config.model_name + res + '.pth')
    os.rename('./graph/' + config.model_name + '_file_train_acc.txt', './graph/' + config.model_name + res + '_file_train_acc.txt')
    # os.rename('./graph/' + config.model_name + '_file_train_loss.txt', './graph/' + config.model_name + res + '_file_train_loss.txt')
    # os.rename('./graph/' + config.model_name + '_file_val_loss.txt', './graph/' + config.model_name + res + '_file_val_loss.txt')
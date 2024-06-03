import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os
from tqdm import tqdm
from config import config
from glob import glob
import os
from torchvision import transforms
import numpy as np
import random
from shutil import copy
from PIL import Image
import math

random.seed(666)  # Set random seeds


#  Dataset parsing
def get_files(file_dir, ratio):
    mid = []
    labels_mid = []
    stand = []
    labels_stand = []
    head = []
    labels_head = []
    down = []
    labels_down = []
    for file in os.listdir(file_dir + 'mid'):
        mid.append(file_dir + 'mid' + '/' + file)
        labels_mid.append(0)
    for file in os.listdir(file_dir + 'stand'):
        stand.append(file_dir + 'stand' + '/' + file)
        labels_stand.append(1)
    for file in os.listdir(file_dir + 'head'):
        head.append(file_dir + 'head' + '/' + file)
        labels_head.append(2)
    for file in os.listdir(file_dir + 'down'):
        down.append(file_dir + 'down' + '/' + file)
        labels_down.append(3)
    image_list = np.hstack((mid, stand, head, down))
    labels_list = np.hstack((labels_mid, labels_stand, labels_head, labels_down))

    temp = np.array([image_list, labels_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])
    all_label_list = [int(float(i)) for i in all_label_list]
    length = len(all_image_list)
    n_test = int(math.ceil(length * ratio))
    n_train = length - n_test

    tra_image = all_image_list[0:n_train]
    tra_label = all_label_list[0:n_train]

    test_image = all_image_list[n_train:-1]
    test_label = all_label_list[n_train:-1]

    train_data = [(tra_image[i], tra_label[i]) for i in range(len(tra_image))]
    test_data = [(test_image[i], test_label[i]) for i in range(len(test_image))]
    # print("train_data = ",test_image)
    # print("test_data = " , test_label)
    return test_data, train_data


# The purpose of a dataset class is to load training and testing data
class datasets(Dataset):
    def __init__(self, data, transform=None, test=False):
        imgs = []
        labels = []
        self.test = test
        self.len = len(data)
        self.data = data
        self.transform = transform
        for i in self.data:
            imgs.append(i[0])
            self.imgs = imgs
            labels.append(int(i[1]))  # ross entropy in pytorch needs to start from 0
            self.labels = labels

    def __getitem__(self, index):
        if self.test:
            filename = self.imgs[index]
            filename = filename
            img_path = self.imgs[index]
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (config.img_width, config.img_height))
            img = transforms.ToTensor()(img)
            return img, filename
        else:
            img_path = self.imgs[index]
            label = self.labels[index]
            # label = int(label)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (config.img_width, config.img_height))
            # img = transforms.ToTensor()(img)

            if self.transform is not None:
                img = Image.fromarray(img)
                img = self.transform(img)

            else:
                img = transforms.ToTensor()(img)
            return img, label

    def __len__(self):
        return len(self.data)  # self.len


def collate_fn(batch): # Splicing multiple samples into one batch
    imgs = []
    label = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), label


if __name__ == '__main__':
    test_data, _ = get_files(config.data_folder, config.ratio)
    _, train_data= get_files(config.data_folder, config.ratio)
    for i in (train_data ):
        print(i)
    print(len(train_data ),len(test_data))

    # transform = transforms.Compose([transforms.ToTensor()])
    # data = datasets(test_data, transform=transform)
    # # print(data[0])
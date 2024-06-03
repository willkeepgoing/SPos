import os
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from config import config
import pickle

"""
Run this function to obtain the mean and standard deviation of the data before network training
"""


class Dataloader():
    def __init__(self, dataroot):
        self.dataroot = dataroot
        self.dirs = ['three_train']

        self.means = [0, 0, 0]
        self.std = [0, 0, 0]

        self.transform = transforms.Compose([
            # transforms.Resize((250, 500)),
            transforms.Resize((config.img_width, config.img_height)),
            transforms.ToTensor()  # The data value ranges from [0255] to [0,1], which is equivalent to dividing by 255
        ])

        # Because ImageFolder is used here, data is classified by folder, and each folder is classified into one category, The label will be automatically labeled
        self.dataset = {x: ImageFolder(os.path.join(dataroot, x), self.transform) for x in self.dirs}

    def get_mean_std(self):
        """
        Calculate the mean and standard deviation of the dataset
        """
        num_imgs = len(self.dataset['three_train'])  # train_half
        for data in self.dataset['three_train']:
            img = data[0]
            for i in range(3):
                # Calculate the mean and standard deviation for each channel
                self.means[i] += img[i, :, :].mean()
                self.std[i] += img[i, :, :].std()

        self.means = np.asarray(self.means) / num_imgs
        self.std = np.asarray(self.std) / num_imgs

        print("{}: normMean = {}".format(type, self.means))
        print("{}: normstd = {}".format(type, self.std))

        # Write the obtained mean and standard deviation into a file, and then read from it
        # with open(mean_std_path, 'wb') as f:
        #     pickle.dump(self.means, f)
        #     pickle.dump(self.stdevs, f)
        #     print('pickle done')


if __name__ == '__main__':
    dataroot = os.getcwd() + '/'
    dataloader = Dataloader(dataroot)
    dataloader.get_mean_std()

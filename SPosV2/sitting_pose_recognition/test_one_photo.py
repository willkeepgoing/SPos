import torch
import cv2
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.autograd import Variable
from PIL import ImageFont, ImageDraw, Image
from config import config
from datasets import *
import Model
import sys
from utils.utils import accuracy

classes = {0: "mid", 1: "stand", 2: "head", 3: "down"}


def test_one_image_0(image, model, device):  # 四分类
    model.eval()
    result = ''
    res = ''
    image = cv2.resize(image, (config.img_height, config.img_width))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = transforms.ToTensor()(image)
    img = img.unsqueeze(0)  # 增加一个维度
    # img = Variable(img)
    img = Variable(img).cuda(device)
    y_pred = model(img)
    smax = nn.Softmax(1)
    smax_out = smax(y_pred)
    pred_label = np.argmax(smax_out.cpu().data.numpy())
    confidence = round(smax_out.cpu().data.numpy()[0][pred_label], 4)
    if confidence < 0.35:
        return ' '
    # print(classes[pred_label] + '\n')
    if pred_label == 0:
        res = '端坐'
        result = '端坐,置信度:' + str(confidence * 100) + '%'
    elif pred_label == 1:
        res = '站立'
        result = '站立,置信度:' + str(confidence * 100) + '%'
    elif pred_label == 2:
        res = '托头'
        result = '托头,置信度:' + str(confidence * 100) + '%'
    elif pred_label == 3:
        res = '趴着'
        result = '趴着,置信度:' + str(confidence * 100) + '%'
    print(result)
    return res


def test_one_image(image, model):
    model.eval()
    image = cv2.resize(image, (config.img_height, config.img_width))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = transforms.ToTensor()(image)
    img = img.unsqueeze(0)  # 增加一个维度
    img = Variable(img)
    # img = Variable(img).cuda()
    y_pred = model(img)
    smax = nn.Softmax(1)
    smax_out = smax(y_pred)
    pred_label = np.argmax(smax_out.cpu().data.numpy())
    confidence = round(smax_out.cpu().data.numpy()[0][pred_label], 4)
    if confidence < 0.35:
        return ' '
    # print(classes[pred_label] + '\n')
    if pred_label == 0:
        res = '端坐'
        result = '这是端坐,置信度为:' + str(confidence * 100) + '%'
    elif pred_label == 1:
        res = '面向左'
        result = '这是面向左,置信度为:' + str(confidence * 100) + '%'
    elif pred_label == 2:
        res = '面向右'
        result = '这是面向右,置信度为:' + str(confidence * 100) + '%'
    elif pred_label == 3:
        res = '站立'
        result = '这是站立,置信度为:' + str(confidence * 100) + '%'
    elif pred_label == 4:
        res = '托头'
        result = '这是托头,置信度为:' + str(confidence * 100) + '%'
    elif pred_label == 5:
        res = '趴着'
        result = '这是趴着,置信度为:' + str(confidence * 100) + '%'
    print(result)
    return res


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


def test_one_image_0(image, model, device):
    model.eval()
    result = ''
    res = ''
    image = cv2.resize(image, (config.img_height, config.img_width))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = transforms.ToTensor()(image)
    img = img.unsqueeze(0)  # Add a dimension
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
        res = 'mid'
        result = 'mid:' + str(confidence * 100) + '%'
    elif pred_label == 1:
        res = 'stand'
        result = 'stand:' + str(confidence * 100) + '%'
    elif pred_label == 2:
        res = 'head'
        result = 'head:' + str(confidence * 100) + '%'
    elif pred_label == 3:
        res = 'down'
        result = 'down:' + str(confidence * 100) + '%'
    print(result)
    return res



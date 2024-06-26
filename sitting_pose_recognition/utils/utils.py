#This file include some functions
from config import config
import torch
import os
import shutil


def save_checkpoint(state, save_model):
    filename = config.weights + config.model_name + ".pth"
    torch.save(state, filename, _use_new_zipfile_serialization=False)
    if save_model:
        message = config.weights + config.model_name+ '.pth'
        print("Get Better top1 : %s saving weights to %s"%(state["accTop1"],message))
        with open("./logs/%s.txt"%config.model_name,"a") as f:
            print("Get Better top1 : %s saving weights to %s"%(state["accTop1"],message),file=f)


def accuracy(output,target,topk = (1, 5)):
    '''计算模型的precision top1 and top5'''
    maxk = max(topk)
    batch_size = target.size(0)  # size(0) = batch_size  size(1) = num_classes
    _, pred = output.topk(maxk, 1, True, True)  # 1 是dim维度
    pred = pred.t() #转置
    correct = pred.eq(target.view(1,-1).expand_as(pred))

    res =[]
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0,keepdim =True)
        '''.float()转换成float类型，False = 0,True = 1'''
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def lr_step(epoch, config_lr):
    if epoch < 30:
        lr = config_lr
    elif epoch < 80:
        lr = config_lr * 0.2
    elif epoch < 120:
        lr = config_lr * 0.04
    else:
        lr = config_lr * 0.01
    return lr

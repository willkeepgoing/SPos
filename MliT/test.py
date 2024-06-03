import os
import time

import torch
import cv2
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.autograd import Variable
from config import config
from datasets import *
import Model
from utils.utils import accuracy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, precision_recall_fscore_support

# classes = {0: "mid", 1: "left", 2: "right"}
classes = {0: "mid", 1: "stand", 2: "head", 3: "down"}
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 用于评估模型
def evaluate(test_loader, model, criterion):
    sum = 0
    test_loss_sum = 0
    test_top1_sum = 0
    model.eval()

    for ims, label in test_loader:
        input_test = Variable(ims).cuda()
        target_test = Variable(torch.from_numpy(np.array(label)).long()).cuda()
        output_test = model(input_test)
        loss = criterion(output_test, target_test)
        top1_test = accuracy(output_test, target_test, topk=(1,))
        sum += 1
        test_loss_sum += loss.data.cpu().numpy()
        test_top1_sum += top1_test[0].cpu().numpy()[0]
    avg_loss = test_loss_sum / sum
    avg_top1 = test_top1_sum / sum
    return avg_loss, avg_top1


def test_one_image(image, model):
    model.eval()
    image = cv2.resize(image, (config.img_height, config.img_width))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = transforms.ToTensor()(image)
    img = img.unsqueeze(0)  # Add a dimension
    # img = Variable(img)
    img = Variable(img).cuda()
    y_pred = model(img)
    smax = nn.Softmax(1)
    smax_out = smax(y_pred)
    pred_label = np.argmax(smax_out.cpu().data.numpy())
    confidence = round(smax_out.cpu().data.numpy()[0][pred_label], 4)
    if confidence < 0.5:
        return 'None'
    # print(classes[pred_label] + '\n')
    if pred_label == 0:
        res = 'mid'
        result = 'mid:' + str(confidence * 100) + '%'
    # elif pred_label == 1:
    #     res = 'left'
    #     result = 'left:' + str(confidence * 100) + '%'
    # elif pred_label == 2:
    #     res = 'right'
    #     result = 'right:' + str(confidence * 100) + '%'
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


def test(test_loader, model):
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    t0 = time.time()
    # predict_file = open("%s_test_output.txt" % config.model_name, 'w')
    num = 0
    right = 0
    yy_true = []
    yy_pred = []
    for i, (input, filename) in enumerate(tqdm(test_loader)):
        num += 1
        if torch.cuda.is_available():
            input = Variable(input).cuda()
        else:
            input = Variable(input)
        # print("input.size = ", input.data.shape)
        y_pred = model(input)
        smax = nn.Softmax(1)
        smax_out = smax(y_pred)
        pred_label = np.argmax(smax_out.cpu().data.numpy())
        x = filename[0].split("/")

        yy_true.append(x[2])
        yy_pred.append(classes[pred_label])
        if x[2] == classes[pred_label]:
            right += 1
        # else:
        #     predict_file.write(filename[0] + ',   ' + classes[pred_label] + '\n')
        #     predict_file.write(filename[0] + ',   ' + '\n')
        #     predict_file.write('\n')

        # print(filename[0] + ',   ' + classes[pred_label] + '\n')
        pred_label = smax_out.cpu().data.numpy()
        # print(pred_label)
        # print('\n')
    # predict_file.write("Acc:" + str(right / num * 100) + "%")
    # print("Acc:" + str(right / num * 100) + "%")
    print("time:", time.time() - t0)
    # print(yy_pred)
    # print(yy_true)
    report = classification_report(yy_true, yy_pred, digits=5)
    print(report)
    # print("Acc:", accuracy_score(yy_true, yy_pred))
    # print("Prec:", precision_score(yy_true, yy_pred))
    # print("Recall:", recall_score(yy_true, yy_pred))
    # print("F1:", f1_score(yy_true, yy_pred))
    return str(right / num * 100)


if __name__ == '__main__':
    test_list, _ = get_files(config.test_data_folder, 1)
    print(len(test_list))
    test_loader = DataLoader(datasets(test_list, transform=None, test=True), batch_size=1, shuffle=False,
                             collate_fn=collate_fn, num_workers=0)
    model = Model.get_net('MliT_half')
    # 'vgg16' or 'resnet18' or 'resnet101' or 'GoogLeNet' or 'VIT' or 'MliT'
    checkpoint = torch.load(config.weights + 'half_MliT_CosineAnnealingLR_SGD_40_6(1.12)_1_91.8918918918919.pth')
    # checkpoint = torch.load(config.weights + config.model_name+'.pth')
    model.load_state_dict(checkpoint["state_dict"])
    print("Start Test.......")
    test(test_loader, model)

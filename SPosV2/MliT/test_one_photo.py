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

classes = {0: "正面", 1: "左面", 2: "右面"}
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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


def test(test_loader, model):
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    predict_file = open("%s.txt" % config.model_name, 'w')
    num = 0
    right = 0
    for i, (input, filename) in enumerate(tqdm(test_loader)):
        num += 1
        if torch.cuda.is_available():
            input = Variable(input).cuda()
        else:
            input = Variable(input)
        # print("input.size = ",input.data.shape)
        y_pred = model(input)
        smax = nn.Softmax(1)
        smax_out = smax(y_pred)
        pred_label = np.argmax(smax_out.cpu().data.numpy())
        x = filename[0].split("/")
        if x[2] == classes[pred_label]:
            right += 1
        print(filename[0] + ',   ' + classes[pred_label] + '\n')
        predict_file.write(filename[0] + ',   ' + classes[pred_label] + '\n')
        pred_label = smax_out.cpu().data.numpy()
        print(filename[0] + ',   ' + '\n')
        predict_file.write(filename[0] + ',   ' + '\n')
        print(pred_label)
        print('\n')
        # predict_file.write(pred_label)
        predict_file.write('\n')
    predict_file.write("准确率:" + str(right / num * 100) + "%")
    print("准确率:" + str(right / num * 100) + "%")


def test_one_image(image, model):
    model.eval()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (config.img_height, config.img_width))
    img = transforms.ToTensor()(image)
    img = img.unsqueeze(0)  # 增加一个维度
    img = Variable(img)
    y_pred = model(img)
    smax = nn.Softmax(1)
    smax_out = smax(y_pred)

    pred_label = np.argmax(smax_out.cpu().data.numpy())
    print(classes[pred_label] + '\n')
    # print(pred_label)
    # print(smax_out.cpu().data.numpy()[0][pred_label])
    if pred_label == 0:
        result = '这是面向前方的坐姿,置信度为：' + str(round(smax_out.cpu().data.numpy()[0][pred_label], 4) * 100) + '%'
    elif pred_label == 1:
        result = '这是面向左侧的坐姿,置信度为：' + str(round(smax_out.cpu().data.numpy()[0][pred_label], 4) * 100) + '%'
    elif pred_label == 2:
        result = '这是面向右侧的坐姿,置信度为：' + str(round(smax_out.cpu().data.numpy()[0][pred_label], 4) * 100) + '%'
    # return result
    print(result)
    return classes[pred_label]


if __name__ == '__main__':
    model = Model.get_net()
    checkpoint = torch.load('./checkpoints/VGG16_20-6(2022.11.10)60epoh.pth')
    model.load_state_dict(checkpoint["state_dict"])
    image = cv2.imread("./24.jpg")

    res = test_one_image(image, model)

    fontpath = "font/simsun.ttc"
    font = ImageFont.truetype(fontpath, 32)
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    # 绘制文字信息
    draw.text((0, 0), res, font=font, fill=(0, 0, 255))
    bk_img = np.array(img_pil)

    cv2.namedWindow("result", 0)
    cv2.resizeWindow("result", 250, 500)
    cv2.imshow("result", bk_img)
    cv2.waitKey(0)

    print(res)

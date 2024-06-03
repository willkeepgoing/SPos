import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy
from PIL import Image
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import division
from cut import cut_person
import shutil
import Model
from config import config
from test_one_photo import test_one_image_0
from tf_bodypix.api import load_model
from queue import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TF_ENABLE_ONEDN_OPTS'] = '0'
wwhole = 0
hhalf = 0


def clean_output_folder():
    single_person_image = './single_person_image/'
    single_person_binary_image_path = './single_person_binary_image/'
    if not os.path.exists(single_person_image):
        os.mkdir(single_person_image)
    else:
        shutil.rmtree(single_person_image)
        os.mkdir(single_person_image)

    if not os.path.exists(single_person_binary_image_path):
        os.mkdir(single_person_binary_image_path)
    else:
        shutil.rmtree(single_person_binary_image_path)
        os.mkdir(single_person_binary_image_path)


def one_image_detect(image, imgsz, names, model, net_model_half, net_model_whole, bodypix_model, device, colored, save_img, pic_num):
    global person_num
    person_num = 1
    with torch.no_grad():
        return detect_one(image, imgsz, names, model, net_model_half, net_model_whole, bodypix_model, device, colored, save_img, pic_num)


def detect_init(imgsz, weights_path, bodypix_model_path, checkpoint_path_half, checkpoint_path_whole, net_half, net_whole):
    # Initialize
    set_logging()
    global device
    # device = select_device('1')
    device = torch.device('cuda:0')
    print(device)

    # Load model
    bodypix_model = load_model(bodypix_model_path)

    model = attempt_load(weights_path, map_location=device)
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names

    net_model_half = Model.get_net(net_half)
    net_model_whole = Model.get_net(net_whole)
    checkpoint_half = torch.load(checkpoint_path_half, map_location=device)  # Image classification model
    checkpoint_whole = torch.load(checkpoint_path_whole, map_location=device)
    net_model_half.load_state_dict(checkpoint_half["state_dict"])
    net_model_whole.load_state_dict(checkpoint_whole["state_dict"])
    net_model_half.to(device)
    net_model_whole.to(device)

    return imgsz, names, model, net_model_half, net_model_whole, bodypix_model, device


def detect_one(image, imgsz, names, model, net_model_half, net_model_whole, bodypix_model, device, colored, save_img, pic_num):
    t0 = time.time()
    location_x_all = []
    location_y_all = []
    res_all = []
    global person_num, wwhole, hhalf
    # Set Dataloader
    dataset = LoadImages(image, img_size=imgsz)

    for img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img)[0]
        # NMS
        pred = non_max_suppression(pred, 0.8, 0.8, classes=[0])

        im0 = im0s
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization

        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                k = 0
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    label = '%s %.2f' % (names[int(cls)], conf)
                    # print(xywh)
                    # print(label)
                    single_person_binary_image_path = './single_person_binary_image/'
                    if save_img:
                        k += 1
                        single_person_image = './single_person_image/{}_{}.jpg'.format(pic_num, k)
                        location_x, location_y, one_img = cut_person(xywh, im0, single_person_image)
                        one_img = cv2.resize(one_img, (int(one_img.shape[1]), int(one_img.shape[0])))
                        image_path, body, direction = division.func(one_img, single_person_binary_image_path, str(person_num), bodypix_model, colored, True, pic_num)
                    else:
                        location_x, location_y, one_img = cut_person(xywh, im0, "")
                        one_img = cv2.resize(one_img, (int(one_img.shape[1]), int(one_img.shape[0])))
                        image_path, body, direction = division.func(one_img, single_person_binary_image_path, str(person_num), bodypix_model, colored, False, pic_num)

                    image = cv2.imread(image_path)
                    # image = numpy.uint8(image)
                    # Identify individual picture
                    if body == 'whole':
                        # wwhole += 1
                        res = test_one_image_0(image, net_model_whole, device) + '(' + direction + ')'
                    else:
                        # hhalf += 1
                        res = test_one_image_0(image, net_model_half, device) + '(' + direction + ')'
                    if res != 'None':
                        res_all.append(res)
                        location_x_all.append(location_x)
                        location_y_all.append(location_y)
                    person_num += 1

    # print(hhalf)
    # print(wwhole)
    print('Done. (%.3fs)' % (time.time() - t0))
    return location_x_all, location_y_all, res_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5x.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.65, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.75, help='IOU threshold for NMS')
    parser.add_argument('--device', default='1,2', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, default=0,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    person_num = 1
    clean_output_folder()
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            if opt.source.endswith(".jpg") or opt.source.endswith(".png") or opt.source.endswith(".mp4"):
                detect()
            else:
                folder = opt.source
                for dirpath, dirnames, filenames in os.walk(folder):
                    for filepath in filenames:
                        print(os.path.join(dirpath, filepath))
                        opt.source = os.path.join(dirpath, filepath)
                        detect()

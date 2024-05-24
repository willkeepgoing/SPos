import os
import cv2
import argparse
from detect import detect_init, one_image_detect
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import time
import shutil

''' 
抽帧
'''


# 每x帧抽取一帧
def get_frames(video_name, frame_gap):
    work_path = os.getcwd()
    os.chdir(work_path)  # 例如f:/video
    v_path = work_path + video_name
    cap = cv2.VideoCapture(v_path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    j = 0
    for i in range(int(frame_count)):
        _, img = cap.read()
        if i % frame_gap == 0:
            j += 1
            cv2.imwrite('./frames/two_{}.jpg'.format(j), img)
            print(j)


# 处理单张图片
def get_image(image_path, output_path):
    imgsz, names, model, net_model_half, net_model_whole, bodypix_model, device = init()
    image = cv2.imread(image_path)
    width = len(image[0])
    font = ImageFont.truetype("./SourceHanSerif-Heavy.ttc", int(width / 20))  # 字体根据原始视频大小
    # location_x_all, location_y_all是写字的位置
    location_x_all, location_y_all, res_all = one_image_detect(image, imgsz, names, model, net_model_half, net_model_whole,
                                                               bodypix_model, device, colored, save_img, 1)

    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    for i in range(len(location_x_all)):
        draw.text((location_x_all[i], location_y_all[i]), res_all[i], font=font, fill=(0, 0, 255))
    img = np.array(img_pil)
    cv2.imwrite(output_path, img)


def clean_output_folder():
    ingle_person_image = './single_person_image/'
    single_person_binary_image_path = './single_person_binary_image/'
    frames_path = './frames/'

    if not os.path.exists(ingle_person_image):
        os.mkdir(ingle_person_image)
    else:
        shutil.rmtree(ingle_person_image)
        os.mkdir(ingle_person_image)

    if not os.path.exists(single_person_binary_image_path):
        os.mkdir(single_person_binary_image_path)
    else:
        shutil.rmtree(single_person_binary_image_path)
        os.mkdir(single_person_binary_image_path)

    if not os.path.exists(frames_path):
        os.mkdir(frames_path)
    else:
        shutil.rmtree(frames_path)
        os.mkdir(frames_path)


# 每x帧抽取一帧处理，并合成视频
def get_video(video_name, frame_gap, output_name):
    imgsz, names, model, net_model_half, net_model_whole, bodypix_model, device = init()
    work_path = os.getcwd()
    os.chdir(work_path)
    v_path = work_path + video_name
    cap = cv2.VideoCapture(v_path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), True)
    font = ImageFont.truetype("./SourceHanSerif-Heavy.ttc", int(width / 20))  # 字体根据原始视频大小

    j = 0
    for i in range(int(frame_count)):
        return_value, img = cap.read()
        if return_value:
            if i % frame_gap == 0:
                j += 1
                if save_frame:
                    cv2.imwrite('./frames/{}.jpg'.format(j), img)
                location_x_all, location_y_all, res_all = one_image_detect(img, imgsz, names, model, net_model_half, net_model_whole,
                                                                           bodypix_model, device, colored, save_img, j)
            img_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(img_pil)
            for i in range(len(location_x_all)):
                draw.text((location_x_all[i], location_y_all[i]), res_all[i], font=font, fill=(0, 0, 255))
            img = np.array(img_pil)
            out.write(img)
    out.release()


def init():
    clean_output_folder()
    imgsz, names, model, net_model_half, net_model_whole, bodypix_model, device = detect_init(imagesz, weights_path, bodypix_model_path,
                                                                        checkpoint_path_half, checkpoint_path_whole, net_half, net_whole)
    global t0
    print('初始化用时 (%.3fs)' % (time.time() - t0))
    return imgsz, names, model, net_model_half, net_model_whole, bodypix_model, device


if __name__ == '__main__':
    global t0
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, default='video', help='get_video or get_frames')
    parser.add_argument('--frame_gap', type=str, default='5', help='frame_gap')
    parser.add_argument('--video_path', type=str, default='/original_video/jiaoshuyang/mid.mp4', help='video_path')
    opt = parser.parse_args()
    opt.frame_gap = int(opt.frame_gap)

    weights_path = 'yolo-R.pt'
    imagesz = 640
    bodypix_model_path = './.keras/ESbody/8ba301b16e59fd7bda330880a9d70e58--tfjs-models-savedmodel-bodypix-mobilenet-float-050-model-stride16'
    checkpoint_path_half = './checkpoints/half_MliT_CosineAnnealingLR_SGD_40_6(1.12)_1_91.8918918918919.pth'
    checkpoint_path_whole = './checkpoints/whole_MliT_CosineAnnealingLR_SGD_0._30_30(1.12)_1_92.77777777777779.pth'
    # 'vgg16' or 'resnet18' or 'resnet101' or 'GoogLeNet' or 'VIT' or 'MliT_half' or 'MliT_whole'
    net_half = 'MliT_half'
    net_whole = 'MliT_whole'
    save_frame = False  # 保存帧
    save_img = False
    colored = True

    if opt.func == 'get_frame':  # 只抽帧
        get_frames(opt.video_path, opt.frame_gap)
    elif opt.func == 'get_image':  # 处理图片
        out_path = 'output/' + opt.video_path.split('/')[-1].split('.')[0] + '_out.jpg'
        get_image(opt.video_path, out_path)
    else:  # 处理视频
        out_path = 'output/' + opt.video_path.split('/')[-1].split('.')[0] + str(time.time()) + '_out.mp4'
        get_video(opt.video_path, opt.frame_gap, out_path)
    print('共计处理用时 (%.3fs)' % (time.time() - t0))
    print('处理完成！')

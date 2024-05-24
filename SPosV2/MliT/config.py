# 可以根据自己的情况进行修改
class MyConfigs:
    data_folder = './train_half/'
    test_data_folder = "./colored_val_half/"
    # data_folder = './train_whole/'
    # test_data_folder = "./colored_val_whole/"

    # OpenPose
    # data_folder = './four_train/'
    # test_data_folder = "./four_val/"
    # data_folder = './three_train/'
    # test_data_folder = "./three_val/"

    # model_name = "openpose_three_VIT_1e-3_640(2023.6.9)(3)"
    model_name = "whole_MliT_whole_CosineAnnealingLR(1.12)_0_"
    weights = "./checkpoints/"
    logs = "./logs/"
    example_folder = "./example/"
    freeze = False
    epochs = 80
    batch_size = 8
    img_height = 224  # 网络输入的高和宽
    img_width = 224
    num_classes = 4
    # num_classes = 3
    lr = 1e-3  # 学习率
    lr_decay = 0.1
    weight_decay = 0.1
    ratio = 0.2


config = MyConfigs()

# Training parameter configuration file
class MyConfigs:
    data_folder = './train_half/'  # training set
    test_data_folder = "./colored_val_half/"  # testing set
    # data_folder = './train_whole/'
    # test_data_folder = "./colored_val_whole/"

    # model_name = "openpose_three_VIT_1e-3_640(2023.6.9)(3)"
    model_name = "whole_MliT_whole_CosineAnnealingLR(1.12)_0_"  # saved model
    weights = "./checkpoints/"  # model file save location
    logs = "./logs/"
    example_folder = "./example/"
    freeze = False
    epochs = 80
    batch_size = 8
    img_height = 224  # height and width of network input
    img_width = 224
    num_classes = 4
    # num_classes = 3
    lr = 1e-3  # learning rate
    lr_decay = 0.1
    weight_decay = 0.1
    ratio = 0.2


config = MyConfigs()

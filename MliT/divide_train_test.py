# coding=utf-8
import os, random, shutil


# Split the image into training set train (0.8) and validation set val (0.2)

def moveFile(classes, dir, train_ratio=0.8, val_ratio=0.2):
    # if not os.path.exists(os.path.join(Dir, 'train')):
    #     os.makedirs(os.path.join(Dir, 'train'))
    #
    # if not os.path.exists(os.path.join(Dir, 'test')):
    #     os.makedirs(os.path.join(Dir, 'test'))

    filenames = []
    for root, dirs, files in os.walk(dir + '/' + classes):
        for name in files:
            filenames.append(name)
        break

    filenum = len(filenames)

    num_train = int(filenum * train_ratio)
    sample_train = random.sample(filenames, num_train)

    if not os.path.exists(os.path.join(dir, 'train', classes)):
        os.makedirs(os.path.join(dir, 'train', classes))

    if not os.path.exists(os.path.join(dir, 'test', classes)):
        os.makedirs(os.path.join(dir, 'test', classes))

    for name in sample_train:
        shutil.move(os.path.join(dir + '/' + classes, name), os.path.join(dir, 'train', classes, name))

    sample_val = list(set(filenames).difference(set(sample_train)))

    for name in sample_val:
        shutil.move(os.path.join(dir + '/' + classes, name), os.path.join(dir, 'test', classes, name))


if __name__ == '__main__':
    Dir = './data'  # +
    if not os.path.exists(os.path.join(Dir, 'train')):
        os.makedirs(os.path.join(Dir, 'train'))

    if not os.path.exists(os.path.join(Dir, 'test')):
        os.makedirs(os.path.join(Dir, 'test'))

    for root, dirs, files in os.walk(Dir):
        for name in dirs:
            folder = os.path.join(root, name)
            print("Processing:" + folder)
            moveFile(name, root)
        break



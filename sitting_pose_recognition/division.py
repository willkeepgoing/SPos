import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from pathlib import Path
import tensorflow as tf
import time
from itertools import chain
import collections


def func(img, output_path, file_name, bodypix_model, colored, save_img, pic_num):
    np.set_printoptions(threshold=10000)
    body = ''
    direction = ''

    output_path = Path(output_path)

    image_array = tf.keras.preprocessing.image.img_to_array(img)
    result = bodypix_model.predict_single(image_array)
    mask = result.get_mask(threshold=0.75)
    if colored:
        coloured_mask = result.get_colored_part_mask(mask)

        # Reno: removing non current individuals
        colored_mask = Reno(coloured_mask)

        color_pic = coloured_mask[:, :, 1]
        color_pic = list(chain.from_iterable(color_pic))

        # PAF: judgment of facial orientation and lower body occlusion
        body, direction =APF(color_pic, colored_mask)

        if save_img:
            file_path = f'{output_path}/mask_{pic_num}_{file_name}.jpg'
            tf.keras.preprocessing.image.save_img(file_path, colored_mask)
        else:
            file_path = f'{output_path}/mask_{file_name}.jpg'
            tf.keras.preprocessing.image.save_img(file_path, coloured_mask)
    else:
        if save_img:
            file_path = f'{output_path}/{pic_num}_{file_name}.jpg'
            tf.keras.preprocessing.image.save_img(file_path, mask)
        else:
            file_path = f'{output_path}/{file_name}.jpg'
            tf.keras.preprocessing.image.save_img(file_path, mask)

    return file_path, body, direction


def APF(color_pic, colored_mask):
    d = collections.Counter(color_pic)
    left = d[61]
    right = d[64]
    down = d[101] + d[150] + d[199] + d[234] + d[247] + d[81]

    # If the proportion of the lower body exceeds 0.1
    if down > len(colored_mask) * len(colored_mask[0]) / 10:
        body = 'whole'
    else:
        body = 'half'

    # print(str(left) + '  ' + str(right))
    if left > right * 1.5:
        direction = ' right'
    elif right > left * 1.5:
        direction = ' left'
    else:
        direction = ' mid'
    return body, direction


def Reno(colored_mask):
    b = colored_mask
    # Initialize visited array
    m, n = len(b), len(b[0])
    visited = [[0] * n for _ in range(m)]

    # Traverse Up
    i = 0
    contour = []
    for j in range(n):
        if b[i][j].all == 0 and visited[i][j] == 0:
            _, contour = dfs_top(b, visited, i, j, contour)
            length = len(contour)
            if length < m + n:
                visited, b = breadthFirstSearch(b, visited, i, j)
    # Traverse Down
    i = m - 1
    contour = []
    for j in range(n):
        if b[i][j].all == 0 and visited[i][j] == 0:
            _, contour = dfs_bottom(b, visited, i, j, contour)
            length = len(contour)
            if length < m + n:
                visited, b = breadthFirstSearch(b, visited, i, j)
    # Traverse Left
    contour = []
    j = 0
    for i in range(m):
        if b[i][j].all == 0 and visited[i][j] == 0:
            _, contour = dfs_left(b, visited, i, j, contour)
            length = len(contour)
            if length < m + n:
                visited, b = breadthFirstSearch(b, visited, i, j)
    # Traverse Right
    contour = []
    j = n - 1
    for i in range(m):
        if b[i][j].all == 0 and visited[i][j] == 0:
            _, contour = dfs_right(b, visited, i, j, contour)
            length = len(contour)
            if length < m + n:
                visited, b = breadthFirstSearch(b, visited, i, j)
    return b


def breadthFirstSearch(b, visited, i, j):
    m, n = len(b), len(b[0])
    queue = deque([(i, j)])

    while queue:
        x, y = queue.popleft()

        if x < 0 or x >= m or y < 0 or y >= n or visited[x][y] or b[x][y] == 0:
            continue

        visited[x][y] = 1
        b[x][y] = 0

        # Breadth priority search in four directions: up, down, left, and right
        queue.append((x - 1, y))  # 上
        queue.append((x + 1, y))  # 下
        queue.append((x, y - 1))  # 左
        queue.append((x, y + 1))  # 右
    return visited, b


def dfs_top(b, visited, i, j, contour):
    m, n = len(b), len(b[0])
    if visited[i][j] or b[i][j] == 0:
        return False, contour
    contour.append((i, j))
    visited[i][j] = 1

    if len(contour) > 0 and (i == 0 or j == 0 or i == m-1 or j == n-1):
        return True, contour

    # The depth first search order is: bottom, right, top, left
    flag, contour = dfs_top(b, visited, i + 1, j, contour)  # 下
    if flag:
        return flag, contour
    flag, contour = dfs_top(b, visited, i, j + 1, contour)  # 右
    if flag:
        return flag, contour
    flag, contour = dfs_top(b, visited, i - 1, j, contour)  # 上
    if flag:
        return flag, contour
    flag, contour = dfs_top(b, visited, i, j - 1, contour)  # 左
    return flag, contour


def dfs_bottom(b, visited, i, j, contour):
    m, n = len(b), len(b[0])
    if visited[i][j] or b[i][j] == 0:
        return False, contour
    contour.append((i, j))
    visited[i][j] = 1

    if len(contour) > 0 and (i == 0 or j == 0 or i == m-1 or j == n-1):
        return True, contour

    # The depth first search order is: top, right, bottom, left
    flag, contour = dfs_bottom(b, visited, i - 1, j, contour)  # 上
    if flag:
        return flag, contour
    flag, contour = dfs_bottom(b, visited, i, j + 1, contour)  # 右
    if flag:
        return flag, contour
    flag, contour = dfs_bottom(b, visited, i + 1, j, contour)  # 下
    if flag:
        return flag, contour
    flag, contour = dfs_bottom(b, visited, i, j - 1, contour)  # 左
    return flag, contour


def dfs_left(b, visited, i, j, contour):
    m, n = len(b), len(b[0])
    if visited[i][j] or b[i][j] == 0:
        return False, contour
    contour.append((i, j))
    visited[i][j] = 1

    if len(contour) > 0 and (i == 0 or j == 0 or i == m-1 or j == n-1):
        return True, contour

    # The depth first search order is: right, bottom, left, top
    flag, contour = dfs_left(b, visited, i, j + 1, contour)  # 右
    if flag:
        return flag, contour
    flag, contour = dfs_left(b, visited, i + 1, j, contour)  # 下
    if flag:
        return flag, contour
    flag, contour = dfs_left(b, visited, i, j - 1, contour)  # 左
    if flag:
        return flag, contour
    flag, contour = dfs_left(b, visited, i - 1, j, contour)  # 上
    return flag, contour


def dfs_right(b, visited, i, j, contour):
    m, n = len(b), len(b[0])
    if visited[i][j] or b[i][j] == 0:
        return False, contour
    contour.append((i, j))
    visited[i][j] = 1

    if len(contour) > 0 and (i == 0 or j == 0 or i == m-1 or j == n-1):
        return True, contour

    # The depth first search order is: left, bottom, right, top
    flag, contour = dfs_right(b, visited, i, j - 1, contour)  # 左
    if flag:
        return flag, contour
    flag, contour = dfs_right(b, visited, i + 1, j, contour)  # 下
    if flag:
        return flag, contour
    flag, contour = dfs_right(b, visited, i, j + 1, contour)  # 右
    if flag:
        return flag, contour
    flag, contour = dfs_right(b, visited, i - 1, j, contour)  # 上
    return flag, contour


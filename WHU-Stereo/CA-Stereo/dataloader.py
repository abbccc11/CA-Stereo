import cv2
import numpy as np
from scipy.signal import convolve2d
import random
# Sobel 核
kx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])  # 水平方向 Sobel 核
ky = np.array([[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]])  # 垂直方向 Sobel 核


def read_left(filename):

    # 读取图像
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

    # 图像标准化
    img = (img - np.mean(img)) / np.std(img)

    # 使用 Sobel 核计算梯度
    dx = convolve2d(img, kx, mode='same', boundary='symm')  # x 方向梯度
    dy = convolve2d(img, ky, mode='same', boundary='symm')  # y 方向梯度

    # 扩展维度以匹配模型输入格式
    img = np.expand_dims(img.astype('float32'), axis=0)
    dx = np.expand_dims(dx.astype('float32'), axis=0)
    dy = np.expand_dims(dy.astype('float32'), axis=0)

    return img, dx, dy

def read_right(filename):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    img = (img - np.mean(img)) / np.std(img)
    return np.expand_dims(img.astype('float32'), 0)

def read_disp(filename):
    disp = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    disp_16x = cv2.resize(disp, (64, 64)) / 16.0
    disp_8x = cv2.resize(disp, (128, 128)) / 8.0
    disp_4x = cv2.resize(disp, (256, 256)) / 4.0
    disp = np.expand_dims(disp, 0)
    disp_16x = np.expand_dims(disp_16x, 0)
    disp_8x = np.expand_dims(disp_8x, 0)
    disp_4x = np.expand_dims(disp_4x, 0)
    return disp_16x, disp_8x, disp_4x, disp


def read_batch(left_paths, right_paths, disp_paths):
    lefts, dxs, dys, rights, d16s, d8s, d4s, ds = [], [], [], [], [], [], [], []
    for left_path, right_path, disp_path in zip(left_paths, right_paths, disp_paths):
        left, dx, dy = read_left(left_path)
        right = read_right(right_path)
        d16, d8, d4, d = read_disp(disp_path)
        lefts.append(left)
        dxs.append(dx)
        dys.append(dy)
        rights.append(right)
        d16s.append(d16)
        d8s.append(d8)
        d4s.append(d4)
        ds.append(d)
    return np.array(lefts), np.array(rights), np.array(dxs), np.array(dys),\
           np.array(d16s), np.array(d8s), np.array(d4s), np.array(ds)


def load_batch(all_left_paths, all_right_paths, all_disp_paths, batch_size=4, reshuffle=False):
    
    assert len(all_left_paths) == len(all_disp_paths)
    assert len(all_right_paths) == len(all_disp_paths)

    i = 0
    while True:
        lefts, rights, dxs, dys, d16s, d8s, d4s, ds = read_batch(
            left_paths=all_left_paths[i * batch_size:(i + 1) * batch_size],
            right_paths=all_right_paths[i * batch_size:(i + 1) * batch_size],
            disp_paths=all_disp_paths[i * batch_size:(i + 1) * batch_size])
        yield [lefts, rights, dxs, dys], [d16s, d8s, d4s, ds]
        i = (i + 1) % (len(all_left_paths) // batch_size)
        if reshuffle:
            if i == 0:
                paths = list(zip(all_left_paths, all_right_paths, all_disp_paths))
                random.shuffle(paths)
                all_left_paths, all_right_paths, all_disp_paths = zip(*paths)

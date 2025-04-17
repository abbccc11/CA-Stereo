import cv2
import numpy as np
from scipy.signal import convolve2d
import random

# Sobel 核
kx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float32)  # 水平方向 Sobel
ky = np.array([[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]], dtype=np.float32)  # 垂直方向 Sobel


def read_left(filename):
    """
    读取彩色左图，对R/G/B三个通道分别计算Sobel梯度，
    并做 (x / 127.5 - 1.0) 形式的归一化，返回:
      - rgb: shape=[3, H, W], 通道顺序为(B, G, R)或(R, G, B)取决于你的需求
      - dx:  shape=[3, H, W], 三通道梯度
      - dy:  shape=[3, H, W], 三通道梯度
    """
    rgb = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if rgb is None:
        raise FileNotFoundError(f"Left image not found: {filename}")

    # 计算每个通道的Sobel梯度
    # 注意：cv2.imread读取顺序是B=rgb[:,:,0], G=rgb[:,:,1], R=rgb[:,:,2]
    bdx = convolve2d(rgb[:, :, 0], kx, mode='same', boundary='symm')
    bdy = convolve2d(rgb[:, :, 0], ky, mode='same', boundary='symm')
    gdx = convolve2d(rgb[:, :, 1], kx, mode='same', boundary='symm')
    gdy = convolve2d(rgb[:, :, 1], ky, mode='same', boundary='symm')
    rdx = convolve2d(rgb[:, :, 2], kx, mode='same', boundary='symm')
    rdy = convolve2d(rgb[:, :, 2], ky, mode='same', boundary='symm')

    # merge后dx, dy均为 (H, W, 3)；按B, G, R顺序拼回去
    dx = cv2.merge([bdx, gdx, rdx])  # shape: [H, W, 3]
    dy = cv2.merge([bdy, gdy, rdy])  # shape: [H, W, 3]

    # 按你的需求做归一化
    # rgb -> float32, scale到[-1,1]
    rgb = rgb.astype(np.float32) / 127.5 - 1.0
    # dx, dy -> float32, scale到[-(some_value), +(some_value)]，这里仅除以127.5
    dx = dx.astype(np.float32) / 127.5
    dy = dy.astype(np.float32) / 127.5

    # 转置到 [C, H, W]
    # 注意：如果你需要(R, G, B)顺序，就把下面的dx, dy也保持相同顺序；目前合并顺序是(B, G, R)
    rgb = np.transpose(rgb, (2, 0, 1))  # [3, H, W]
    dx = np.transpose(dx, (2, 0, 1))    # [3, H, W]
    dy = np.transpose(dy, (2, 0, 1))    # [3, H, W]

    return rgb, dx, dy


def read_right(filename):
    """
    读取彩色右图，只做 (x / 127.5 - 1.0) 归一化，不计算梯度。
    返回 shape=[3, H, W]。
    """
    image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Right image not found: {filename}")

    image = image.astype(np.float32) / 127.5 - 1.0
    # [H, W, 3] -> [3, H, W]
    image = np.transpose(image, (2, 0, 1))
    return image


def read_disp(filename):
    """
    读取视差图并在 1/16、1/8、1/4，以及原分辨率上分别进行缩放，保持原逻辑不变。
    返回的 shape均为 [1, H, W]。
    """
    disp = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if disp is None:
        raise FileNotFoundError(f"Disparity map not found: {filename}")

    disp_16x = cv2.resize(disp, (64, 64)) / 16.0
    disp_8x = cv2.resize(disp, (128, 128)) / 8.0
    disp_4x = cv2.resize(disp, (256, 256)) / 4.0

    disp = np.expand_dims(disp, 0)      # [1, H, W]
    disp_16x = np.expand_dims(disp_16x, 0)
    disp_8x = np.expand_dims(disp_8x, 0)
    disp_4x = np.expand_dims(disp_4x, 0)
    return disp_16x, disp_8x, disp_4x, disp


def read_batch(left_paths, right_paths, disp_paths):
    """
    批量读取左图、右图和视差图。
    返回值:
      lefts:  [batch_size, 3, H, W]
      rights: [batch_size, 3, H, W]
      dxs:    [batch_size, 3, H, W]
      dys:    [batch_size, 3, H, W]
      d16s:   [batch_size, 1, H/16, W/16]
      d8s:    ...
      d4s:    ...
      ds:     [batch_size, 1, H, W]
    """
    lefts, dxs, dys = [], [], []
    rights = []
    d16s, d8s, d4s, ds = [], [], [], []

    for left_path, right_path, disp_path in zip(left_paths, right_paths, disp_paths):
        rgb, dx, dy = read_left(left_path)
        right = read_right(right_path)
        d16, d8, d4, d = read_disp(disp_path)

        lefts.append(rgb)
        dxs.append(dx)
        dys.append(dy)
        rights.append(right)
        d16s.append(d16)
        d8s.append(d8)
        d4s.append(d4)
        ds.append(d)

    return (
        np.array(lefts),   # [B, 3, H, W]
        np.array(rights),  # [B, 3, H, W]
        np.array(dxs),     # [B, 3, H, W]
        np.array(dys),     # [B, 3, H, W]
        np.array(d16s),    # [B, 1, H/16, W/16]
        np.array(d8s),
        np.array(d4s),
        np.array(ds)       # [B, 1, H, W]
    )


def load_batch(all_left_paths, all_right_paths, all_disp_paths,
               batch_size=4, reshuffle=False):
    """
    按批次迭代加载数据。可以指定是否在每个epoch结束时 reshuffle。
    """
    assert len(all_left_paths) == len(all_disp_paths)
    assert len(all_right_paths) == len(all_disp_paths)

    i = 0
    while True:
        # 读取一个批次
        lefts, rights, dxs, dys, d16s, d8s, d4s, ds = read_batch(
            left_paths=all_left_paths[i * batch_size:(i + 1) * batch_size],
            right_paths=all_right_paths[i * batch_size:(i + 1) * batch_size],
            disp_paths=all_disp_paths[i * batch_size:(i + 1) * batch_size]
        )

        # 这里返回格式可根据你的网络需要自行调整
        yield [lefts, rights, dxs, dys], [d16s, d8s, d4s, ds]

        i = (i + 1) % (len(all_left_paths) // batch_size)

        if reshuffle and i == 0:
            # 一个epoch结束后，打乱数据列表
            combined = list(zip(all_left_paths, all_right_paths, all_disp_paths))
            random.shuffle(combined)
            all_left_paths, all_right_paths, all_disp_paths = zip(*combined)
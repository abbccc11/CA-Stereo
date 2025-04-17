import torch
import os
from tqdm import tqdm
from mynet import CAStereo
from torch.utils.data import Dataset, DataLoader  # 假设您有自定义的数据加载函数
import glob
import numpy as np
import cv2
from scipy.signal import convolve2d
import random
import glob
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

kx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float32)  # Horizontal Sobel kernel
ky = np.array([[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]], dtype=np.float32)  # Vertical Sobel kernel



class StereoDataset(Dataset):
    """
    PyTorch Dataset for Stereo Image Data with Asymmetric Intensity Augmentation and Random Vertical Flip,
    and channel-wise gradient computation.
    """
    def __init__(self, left_paths, right_paths, disp_paths,
                 min_disp=-64, max_disp=64, transform=None, augment=True):
        """
        Initializes the dataset with paths to left images, right images, and disparity maps.

        Args:
            left_paths (list): List of file paths for left images.
            right_paths (list): List of file paths for right images.
            disp_paths (list): List of file paths for disparity maps.
            min_disp (int): Minimum disparity value.
            max_disp (int): Maximum disparity value.
            transform (callable, optional): Optional transform to be applied on a sample.
            augment (bool): Whether to apply data augmentation.
        """
        assert len(left_paths) == len(right_paths) == len(disp_paths), (
            "Mismatch in number of left, right, and disparity images."
        )
        self.left_paths = left_paths
        self.right_paths = right_paths
        self.disp_paths = disp_paths
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.left_paths)

    # ------------------------------ #
    #  Reading Methods
    # ------------------------------ #

    def read_left(self, filename):
        """
        Reads the left image in UNCHANGED mode.
        Assumes it's a 3-channel 8-bit image (BGR).
        """
        img_bgr = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            raise FileNotFoundError(f"Left image not found: {filename}")
        
        return img_bgr

    def read_right(self, filename):
        """
        Reads the right image in UNCHANGED mode.
        Assumes it's also a 3-channel 8-bit image.
        """
        img_bgr = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if img_bgr is None:
            raise FileNotFoundError(f"Right image not found: {filename}")
        
        return img_bgr

    def read_disp(self, filename):
        """
        Reads and processes the disparity map at multiple scales.
        Returns disp_16x, disp_8x, disp_4x, disp as float32.
        """
        disp = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if disp is None:
            raise FileNotFoundError(f"Disparity map not found: {filename}")

        disp = disp.astype(np.float32)

        disp_16x = cv2.resize(disp, (64, 64)) / 16.0
        disp_8x = cv2.resize(disp, (128, 128)) / 8.0
        disp_4x = cv2.resize(disp, (256, 256)) / 4.0

        return disp_16x, disp_8x, disp_4x, disp

    # ------------------------------ #
    #  Augmentation Methods
    # ------------------------------ #

    def adjust_contrast(self, img, contrast_factor):
        """
        Adjusts the contrast of an image, channel-wise.
        """
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        return mean + contrast_factor * (img - mean)

    def adjust_brightness(self, img, brightness_factor):
        """
        Adjusts the brightness of an image, channel-wise.
        """
        return img + brightness_factor

    def adjust_gamma(self, img, gamma):
        """
        Adjusts the gamma of an image [0,1]->power->[0,1], or [0,255]->normalized->power->scale back.
        """
        img_min = img.min()
        img_max = img.max()
        img_normalized = (img - img_min) / (img_max - img_min + 1e-8)
        img_gamma = np.power(img_normalized, gamma)
        return img_gamma * (img_max - img_min) + img_min

    def random_intensity_augmentation(self, img):
        """
        Applies random intensity augmentations: contrast, brightness, and gamma adjustments.
        """
        contrast_factor = random.uniform(0.8, 1.2)
        img = self.adjust_contrast(img, contrast_factor)

        brightness_factor = random.uniform(-0.1, 0.1) * img.max()
        img = self.adjust_brightness(img, brightness_factor)

        gamma = random.uniform(0.8, 1.2)
        img = self.adjust_gamma(img, gamma)

        return img

    def random_vertical_flip(self, left, right,
                             disp_16x, disp_8x, disp_4x, disp):
        """
        Randomly flips the images (and optional dx, dy) vertically with 50% chance.
        """
        if random.random() < 0.5:
            left = np.flip(left, axis=0).copy()
            right = np.flip(right, axis=0).copy()
            disp_16x = np.flip(disp_16x, axis=0).copy()
            disp_8x = np.flip(disp_8x, axis=0).copy()
            disp_4x = np.flip(disp_4x, axis=0).copy()
            disp = np.flip(disp, axis=0).copy()

        return left, right, disp_16x, disp_8x, disp_4x, disp

    # ------------------------------ #
    #  Gradient Computation
    # ------------------------------ #

    def compute_channel_grad(self, img_bgr):
        """
        Compute Sobel gradient for each B/G/R channel independently.
        Returns dx, dy each shape [H,W,3].
        """
        bdx = convolve2d(img_bgr[:, :, 0], kx, mode='same')
        bdy = convolve2d(img_bgr[:, :, 0], ky, mode='same')
        gdx = convolve2d(img_bgr[:, :, 1], kx, mode='same')
        gdy = convolve2d(img_bgr[:, :, 1], ky, mode='same')
        rdx = convolve2d(img_bgr[:, :, 2], kx, mode='same')
        rdy = convolve2d(img_bgr[:, :, 2], ky, mode='same')

        dx = cv2.merge([bdx, gdx, rdx])  # B=0, G=1, R=2
        dy = cv2.merge([bdy, gdy, rdy])
        return dx, dy

    # ------------------------------ #
    #  __getitem__
    # ------------------------------ #

    def __getitem__(self, idx):
        """
        Retrieves the processed data sample at the specified index.
        """
        left_path = self.left_paths[idx]
        right_path = self.right_paths[idx]
        disp_path = self.disp_paths[idx]

        # 1. Read images (BGR) with IMREAD_UNCHANGED
        left_bgr = self.read_left(left_path)   # shape [H,W,3], uint8 (assumed)
        right_bgr = self.read_right(right_path)
        disp_16x, disp_8x, disp_4x, disp = self.read_disp(disp_path)  # float32

       
        # 3. Optional intensity augmentation
        if self.augment:
            # Convert images to float32 for augmentation
            left_bgr = left_bgr.astype(np.float32)
            right_bgr = right_bgr.astype(np.float32)

            # Apply random intensity augmentation asymmetrically
            left_bgr = self.random_intensity_augmentation(left_bgr)
            right_bgr = self.random_intensity_augmentation(right_bgr)

            # Apply random vertical flip
            left_bgr, right_bgr, disp_16x, disp_8x, disp_4x, disp = self.random_vertical_flip(
                left_bgr, right_bgr, disp_16x, disp_8x, disp_4x, disp
            )
        else:
            # If not augmenting, convert images to float32 without changes
            left_bgr = left_bgr.astype(np.float32)
            right_bgr = right_bgr.astype(np.float32)
            
        dx_bgr, dy_bgr = self.compute_channel_grad(left_bgr)    
        left_bgr = left_bgr / 255.0
        right_bgr = right_bgr / 255.0
        # 4. Compute channel-wise gradients in BGR domain
        dx_bgr = dx_bgr/255.0
        dy_bgr = dy_bgr/255.0
        dx_bgr = dx_bgr.astype(np.float32)
        dy_bgr = dy_bgr.astype(np.float32)

        # 7. Transpose from [H,W,3] -> [3,H,W]
        left_bgr = np.transpose(left_bgr, (2, 0, 1))  # [3, H, W]
        right_bgr = np.transpose(right_bgr, (2, 0, 1))
        dx_bgr = np.transpose(dx_bgr, (2, 0, 1))
        dy_bgr = np.transpose(dy_bgr, (2, 0, 1))

        # 8. Expand disp to [1,H,W]
        disp_16x = np.expand_dims(disp_16x, axis=0)  # [1, H/16, W/16]
        disp_8x = np.expand_dims(disp_8x, axis=0)
        disp_4x = np.expand_dims(disp_4x, axis=0)
        disp = np.expand_dims(disp, axis=0)          # [1, H, W]

        # 9. Convert to torch tensors
        left_tensor = torch.from_numpy(left_bgr).float()   # [3,H,W]
        right_tensor = torch.from_numpy(right_bgr).float()
        dx_tensor = torch.from_numpy(dx_bgr).float()       # [3,H,W]
        dy_tensor = torch.from_numpy(dy_bgr).float()

        disp_16x = torch.from_numpy(disp_16x).float()      # [1,H',W']
        disp_8x = torch.from_numpy(disp_8x).float()
        disp_4x = torch.from_numpy(disp_4x).float()
        disp = torch.from_numpy(disp).float()

        inputs = {
            'left': left_tensor,
            'right': right_tensor,
            'dx': dx_tensor,
            'dy': dy_tensor
        }
        targets = {
            'disp_16x': disp_16x,
            'disp_8x': disp_8x,
            'disp_4x': disp_4x,
            'disp': disp
        }

        return inputs, targets
# 计算 End-Point Error (EPE)
def compute_epe(pred, gt, min_disp, max_disp):
    mask = (gt >= min_disp) & (gt < max_disp)
    abs_error = torch.abs(pred - gt)
    masked_error = abs_error[mask]
    error = masked_error.sum().item()
    nums = mask.sum().item()
    epe = error / nums if nums > 0 else 0.0
    return error, nums, epe

# D1 computation
def compute_d1(pred, gt, min_disp, max_disp, threshold=3.0):
    mask = (gt >= min_disp) & (gt < max_disp)
    abs_error = torch.abs(pred - gt)
    err_mask = (abs_error > threshold) & mask
    err_disps = err_mask.sum().item()
    nums = mask.sum().item()
    d1 = err_disps / nums if nums > 0 else 0.0
    return err_disps, nums, d1

# 定义验证函数
def validate_model(val_loader, model_path, device="cuda", min_disp=-128, max_disp=64):
    model = CAStereo(min_disp, max_disp).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    val_epe, val_d1 = 0.0, 0.0
    total_epe_nums, total_d1_nums = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader, desc="Validation")):
            # inputs 是字典, targets 是字典
            lefts = inputs['left'].to(device)
            rights = inputs['right'].to(device)
            dxs = inputs['dx'].to(device)
            dys = inputs['dy'].to(device)

            ds = targets['disp'].to(device)

            # 前向传播
            outputs = model(lefts, rights, dxs, dys)
            refine = outputs[-1]  # 假设最后一个是 refine

            if refine is None or refine.shape[0] == 0:
                print("Error: refine output is empty or None")
                continue

            # 计算 EPE/D1
            epe_error, epe_nums, _ = compute_epe(refine, ds, min_disp, max_disp)
            d1_error, d1_nums, _ = compute_d1(refine, ds, min_disp, max_disp)
            
            val_epe += epe_error
            val_d1 += d1_error
            total_epe_nums += epe_nums
            total_d1_nums += d1_nums

    # 计算平均 EPE / D1
    avg_val_epe = val_epe / total_epe_nums if total_epe_nums > 0 else 0.0
    avg_val_d1  = val_d1 / total_d1_nums if total_d1_nums > 0 else 0.0

    print(f"Validation Results - Avg EPE: {avg_val_epe:.4f}, Avg D1: {avg_val_d1:.4f}")

if __name__ == "__main__":
    data_dir = "/mnt/data1/crlj/Dataset/US3D/oma"  # Update this path to your dataset directory

    # Retrieve all left, right, and disparity image paths using glob
    val_left_paths = sorted(glob.glob(os.path.join(data_dir, '*_LEFT_RGB.tif')))
    val_right_paths = sorted(glob.glob(os.path.join(data_dir, '*_RIGHT_RGB.tif')))
    val_disp_paths = sorted(glob.glob(os.path.join(data_dir, '*_LEFT_DSP.tif')))
    val_dataset = StereoDataset(val_left_paths, val_right_paths, val_disp_paths,augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    # 打印调试信息

    # 模型路径
    model_path = "/home/crlj/bh/Mynet/best_checkpoint.pth"  # 请根据需要修改模型路径

    # 调用验证函数
    validate_model(val_loader, model_path, device="cuda", min_disp=-96, max_disp=96)


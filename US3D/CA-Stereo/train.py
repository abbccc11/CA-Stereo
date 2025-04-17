import os
import cv2
import numpy as np
from scipy.signal import convolve2d
import random
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

# Removed Optuna import
# import optuna  # Import Optuna

from mynet import IGEVStereo  # Ensure this is correctly imported

# ======================
# Data Reader Components
# ======================

# Define Sobel kernels for gradient computation
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

def list_image_files(directory, extensions=(".tiff", ".tif")):
    """
    Lists all image files in the specified directory with given extensions.

    Args:
        directory (str): Directory path to search for images.
        extensions (tuple): Tuple of acceptable file extensions.

    Returns:
        list: Sorted list of image file paths.
    """
    return sorted([
        os.path.join(directory, f) for f in os.listdir(directory)
        if f.lower().endswith(extensions)
    ])

# ======================
# Loss and Metric Functions
# ======================

def masked_smooth_l1_loss(pred, target, mask):
    """
    Computes Smooth L1 Loss with a mask.

    Args:
        pred (torch.Tensor): Predicted disparity map [B, H, W].
        target (torch.Tensor): Ground truth disparity map [B, H, W].
        mask (torch.Tensor): Boolean mask [B, H, W] where True indicates valid pixels.

    Returns:
        torch.Tensor: Computed loss.
    """
    mask = mask.bool()
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    loss = F.smooth_l1_loss(pred[mask], target[mask], reduction='mean')
    return loss

def compute_epe(pred, gt, min_disp, max_disp):
    """
    Computes End-Point Error (EPE).

    Args:
        pred (torch.Tensor): Predicted disparity map [B, H, W].
        gt (torch.Tensor): Ground truth disparity map [B, H, W].
        min_disp (int): Minimum disparity value.
        max_disp (int): Maximum disparity value.

    Returns:
        tuple: (error, nums, epe)
    """
    mask = (gt >= min_disp) & (gt < max_disp)
    abs_error = torch.abs(pred - gt)
    masked_error = abs_error[mask]
    error = masked_error.sum().item()
    nums = mask.sum().item()
    epe = error / nums if nums > 0 else 0.0
    return error, nums, epe

def compute_d1(pred, gt, min_disp, max_disp, threshold=3.0):
    """
    Computes D1 metric.

    Args:
        pred (torch.Tensor): Predicted disparity map [B, H, W].
        gt (torch.Tensor): Ground truth disparity map [B, H, W].
        min_disp (int): Minimum disparity value.
        max_disp (int): Maximum disparity value.
        threshold (float): Threshold for error.

    Returns:
        tuple: (err_disps, nums, d1)
    """
    mask = (gt >= min_disp) & (gt < max_disp)
    abs_error = torch.abs(pred - gt)
    err_mask = (abs_error > threshold) & mask
    err_disps = err_mask.sum().item()
    nums = mask.sum().item()
    d1 = err_disps / nums if nums > 0 else 0.0
    return err_disps, nums, d1

# ======================
# Validation Function
# ======================

def validate_model(model, val_loader, device, min_disp, max_disp, lambda_refine=0.6, lambda1=1.0, lambda2=0.7, lambda3=0.5):
    """
    Validates the model on the validation dataset.

    Args:
        model (nn.Module): The neural network model.
        val_loader (DataLoader): DataLoader for validation data.
        device (str): Device to run validation on.
        min_disp (int): Minimum disparity value.
        max_disp (int): Maximum disparity value.
        lambda_refine (float): Weight for refine loss.
        lambda1 (float): Weight for disp_4x loss.
        lambda2 (float): Weight for disp_8x loss.
        lambda3 (float): Weight for disp_16x loss.

    Returns:
        tuple: (avg_val_loss, avg_epe, avg_d1)
    """
    model.eval()
    val_loss = 0.0
    total_epe = 0.0
    total_d1 = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            inputs, targets = batch
            lefts = inputs['left'].to(device, dtype=torch.float32)
            rights = inputs['right'].to(device, dtype=torch.float32)
            dxs = inputs['dx'].to(device, dtype=torch.float32)
            dys = inputs['dy'].to(device, dtype=torch.float32)
            d16s = targets['disp_16x'].to(device, dtype=torch.float32)
            d8s = targets['disp_8x'].to(device, dtype=torch.float32)
            d4s = targets['disp_4x'].to(device, dtype=torch.float32)
            ds = targets['disp'].to(device, dtype=torch.float32)
            
            # Forward pass
            d16_pred, d8_pred, d4_pred, refine = model(lefts, rights, dxs, dys)
            
            # Create masks
            mask_refine = ((ds >= min_disp) & (ds < max_disp))
            mask_d16 = ((d16s >= min_disp / 16) & (d16s < max_disp / 16))
            mask_d8 = ((d8s >= min_disp / 8) & (d8s < max_disp / 8))
            mask_d4 = ((d4s >= min_disp / 4) & (d4s < max_disp / 4))
            
            # Compute losses
            loss_refine = masked_smooth_l1_loss(refine, ds, mask_refine)
            loss_d16 = masked_smooth_l1_loss(d16_pred, d16s, mask_d16)
            loss_d8 = masked_smooth_l1_loss(d8_pred, d8s, mask_d8)
            loss_d4 = masked_smooth_l1_loss(d4_pred, d4s, mask_d4)
            
            # Total loss
            loss = (
                lambda_refine * loss_refine
                + lambda3 * loss_d16
                + lambda2 * loss_d8
                + lambda1 * loss_d4
            )
            
            val_loss += loss.item()
            
            # Compute metrics (EPE and D1)
            epe_error, epe_num, epe = compute_epe(refine, ds, min_disp, max_disp)
            d1_err, d1_num, d1 = compute_d1(refine, ds, min_disp, max_disp, threshold=3.0)
            total_epe += epe
            total_d1 += d1
            total_samples += 1
    
    avg_val_loss = val_loss / len(val_loader)
    avg_epe = total_epe / total_samples
    avg_d1 = total_d1 / total_samples
    print(f"Validation Loss: {avg_val_loss:.4f}, EPE: {avg_epe:.4f}, D1: {avg_d1:.4f}")
    return avg_val_loss, avg_epe, avg_d1

# ======================
# Training Function
# ======================

def train_model(
        train_loader,
        val_loader,
        device,
        hyperparameters,
        min_disp=-96,
        max_disp=96,
        patience=20,
        min_delta=0.0
    ):
    """
    Trains the model with given hyperparameters and implements early stopping.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        device (str): Device to train on.
        hyperparameters (dict): Dictionary containing hyperparameters.
        min_disp (int): Minimum disparity value.
        max_disp (int): Maximum disparity value.
        patience (int): Number of epochs to wait for improvement before stopping.
        min_delta (float): Minimum change in validation loss to qualify as improvement.

    Returns:
        tuple: (best_val_loss, best_epe, best_d1)
    """
    epochs = hyperparameters['epochs']
    lr = hyperparameters['lr']
    batch_size = hyperparameters['batch_size']
    lambda_refine = hyperparameters['lambda_refine']
    lambda1 = hyperparameters['lambda1']
    lambda2 = hyperparameters['lambda2']
    lambda3 = hyperparameters['lambda3']
    weight_decay = hyperparameters['weight_decay']
    pct_start = hyperparameters['pct_start']
    
    model = IGEVStereo(min_disp, max_disp).to(device)
    
    # Define optimizer with selective weight decay
    decay_layers = (nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d, nn.Linear)
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Check if parameter belongs to layers that require weight decay
        module_name = name.rsplit('.', 1)[0] if '.' in name else ''
        module = dict(model.named_modules()).get(module_name, model)
        if isinstance(module, decay_layers):
            decay_params.append(param)
        else:
            no_decay_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=lr, eps=1e-8)
    
    # Define scheduler
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(optimizer, max_lr=lr, pct_start=pct_start, steps_per_epoch=steps_per_epoch, epochs=epochs, cycle_momentum=False, anneal_strategy='linear')
    
    # Initialize mixed precision scaler
    scaler = GradScaler()
    
    # Initialize best metrics
    best_val_loss = float('inf')
    best_epe = float('inf')
    best_d1 = float('inf')
    best_epoch = -1
    epochs_without_improvement = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            inputs, targets = batch
            lefts = inputs['left'].to(device, dtype=torch.float32)
            rights = inputs['right'].to(device, dtype=torch.float32)
            dxs = inputs['dx'].to(device, dtype=torch.float32)
            dys = inputs['dy'].to(device, dtype=torch.float32)
            d16s = targets['disp_16x'].to(device, dtype=torch.float32)
            d8s = targets['disp_8x'].to(device, dtype=torch.float32)
            d4s = targets['disp_4x'].to(device, dtype=torch.float32)
            ds = targets['disp'].to(device, dtype=torch.float32)
            
            optimizer.zero_grad()
            
            with autocast():
                # Forward pass
                d16_pred, d8_pred, d4_pred, refine = model(lefts, rights, dxs, dys)
                
                # Create masks
                mask_refine = ((ds >= min_disp) & (ds < max_disp))
                mask_d16 = ((d16s >= min_disp / 16) & (d16s < max_disp / 16))
                mask_d8 = ((d8s >= min_disp / 8) & (d8s < max_disp / 8))
                mask_d4 = ((d4s >= min_disp / 4) & (d4s < max_disp / 4))
                
                # Compute losses
                loss_refine = masked_smooth_l1_loss(refine, ds, mask_refine)
                loss_d16 = masked_smooth_l1_loss(d16_pred, d16s, mask_d16)
                loss_d8 = masked_smooth_l1_loss(d8_pred, d8s, mask_d8)
                loss_d4 = masked_smooth_l1_loss(d4_pred, d4s, mask_d4)
                
                # Total loss
                loss = (
                    lambda_refine * loss_refine
                    + lambda3 * loss_d16
                    + lambda2 * loss_d8
                    + lambda1 * loss_d4
                )
            
            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            
            pbar.set_postfix({'Loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        avg_val_loss, avg_epe, avg_d1 = validate_model(model, val_loader, device, min_disp, max_disp, lambda_refine, lambda1, lambda2, lambda3)
        
        # Check for improvement based on EPE and D1
        if avg_epe < best_epe or (avg_epe == best_epe and avg_d1 < best_d1):
            best_val_loss = avg_val_loss
            best_epe = avg_epe
            best_d1 = avg_d1
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            # Save the best model
            checkpoint = {
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(checkpoint, "best_checkpoint.pth")
            print(f"Saved Best Checkpoint at Epoch {best_epoch} with EPE: {best_epe:.4f}, D1: {best_d1:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement in EPE and D1 for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break
        
        # Removed Optuna-specific reporting and pruning
        # trial.report(avg_epe, epoch)
        
        # Check if the trial should be pruned
        # if trial.should_prune():
        #     print("Trial was pruned.")
        #     raise optuna.exceptions.TrialPruned()
    
    print(f"Training Completed. Best Epoch: {best_epoch}, Best Validation Loss: {best_val_loss:.4f}, Best EPE: {best_epe:.4f}, Best D1: {best_d1:.4f}")
    return best_val_loss, best_epe, best_d1

# ======================
# Main Execution
# ======================

if __name__ == "__main__":
    # Define hyperparameters manually since Optuna is removed
    hyperparameters = {
        'epochs': 100,            # You can adjust this as needed
        'lr': 0.001,              # Learning rate
        'batch_size': 4,         # Batch size
        'lambda_refine': 1.0,    # Weight for refine loss
        'lambda1': 1.0,          # Weight for disp_4x loss
        'lambda2': 0.7,          # Weight for disp_8x loss
        'lambda3': 0.5,          # Weight for disp_16x loss
        'weight_decay': 0.00001,  # Weight decay
        'pct_start': 0.01,        # Percentage of cycle for the scheduler
    }

    data_dir = "/mnt/data1/crlj/Dataset/US3D/jax"  # Update this path to your dataset directory

    # Retrieve all left, right, and disparity image paths using glob
    left_images = sorted(glob.glob(os.path.join(data_dir, '*_LEFT_RGB.tif')))
    right_images = sorted(glob.glob(os.path.join(data_dir, '*_RIGHT_RGB.tif')))
    disp_images = sorted(glob.glob(os.path.join(data_dir, '*_LEFT_DSP.tif')))

    # Ensure that all lists have the same length
    assert len(left_images) == len(right_images) == len(disp_images), "Mismatch in number of left, right, and disparity images."

    total_size = len(left_images)
    train_size = 1600
    val_size = 539

    assert train_size + val_size <= total_size, f"Not enough data. Required: {train_size + val_size}, Available: {total_size}"

    # Set random seed for reproducibility
    random_seed = 42
    random.seed(random_seed)

    # Create a list of indices and shuffle them
    indices = list(range(total_size))
    random.shuffle(indices)

    # Split indices for training and validation
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]

    # Create training and validation file lists
    train_left_paths = [left_images[i] for i in train_indices]
    train_right_paths = [right_images[i] for i in train_indices]
    train_disp_paths = [disp_images[i] for i in train_indices]

    val_left_paths = [left_images[i] for i in val_indices]
    val_right_paths = [right_images[i] for i in val_indices]
    val_disp_paths = [disp_images[i] for i in val_indices]

    # Create datasets
    train_dataset = StereoDataset(train_left_paths, train_right_paths, train_disp_paths,augment=True)
    val_dataset = StereoDataset(val_left_paths, val_right_paths, val_disp_paths,augment=False)

    # Create data loaders based on suggested batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparameters['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparameters['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Determine the device to use
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Start training
    best_val_loss, best_epe, best_d1 = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        hyperparameters=hyperparameters,
        min_disp=-96,
        max_disp=96
    )

    print(f"Best Validation Loss: {best_val_loss:.4f}, Best EPE: {best_epe:.4f}, Best D1: {best_d1:.4f}")
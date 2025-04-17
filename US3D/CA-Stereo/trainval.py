import os
import cv2
import numpy as np
from scipy.signal import convolve2d
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

import optuna  # Import Optuna

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
    PyTorch Dataset for Stereo Image Data.
    """
    def __init__(self, left_paths, right_paths, disp_paths, min_disp=-128, max_disp=64, transform=None):
        """
        Initializes the dataset with paths to left images, right images, and disparity maps.

        Args:
            left_paths (list): List of file paths for left images.
            right_paths (list): List of file paths for right images.
            disp_paths (list): List of file paths for disparity maps.
            min_disp (int): Minimum disparity value.
            max_disp (int): Maximum disparity value.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        assert len(left_paths) == len(right_paths) == len(disp_paths), "Mismatch in number of left, right, and disparity images."
        self.left_paths = left_paths
        self.right_paths = right_paths
        self.disp_paths = disp_paths
        self.min_disp = min_disp
        self.max_disp = max_disp
        self.transform = transform

    def __len__(self):
        return len(self.left_paths)

    def read_left(self, filename):
        """
        Reads and processes the left image.

        Args:
            filename (str): Path to the left image.

        Returns:
            tuple: (img, dx, dy) where each is a numpy array with shape [1, H, W].
        """
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Left image not found: {filename}")

        # Normalize image
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)  # Avoid division by zero

        # Compute gradients using Sobel kernels
        dx = convolve2d(img, kx, mode='same', boundary='symm')
        dy = convolve2d(img, ky, mode='same', boundary='symm')

        # Expand dimensions to add channel dimension
        img = np.expand_dims(img.astype(np.float32), axis=0)  # Shape: [1, H, W]
        dx = np.expand_dims(dx.astype(np.float32), axis=0)    # Shape: [1, H, W]
        dy = np.expand_dims(dy.astype(np.float32), axis=0)    # Shape: [1, H, W]

        return img, dx, dy

    def read_right(self, filename):
        """
        Reads and processes the right image.

        Args:
            filename (str): Path to the right image.

        Returns:
            numpy.ndarray: Right image with shape [1, H, W].
        """
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Right image not found: {filename}")

        # Normalize image
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)

        # Expand dimensions to add channel dimension
        img = np.expand_dims(img.astype(np.float32), axis=0)  # Shape: [1, H, W]

        return img

    def read_disp(self, filename):
        """
        Reads and processes the disparity map at multiple scales.

        Args:
            filename (str): Path to the disparity map.

        Returns:
            tuple: (disp_16x, disp_8x, disp_4x, disp) each as numpy arrays with shape [1, H/s, W/s].
        """
        disp = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if disp is None:
            raise FileNotFoundError(f"Disparity map not found: {filename}")

        disp = disp.astype(np.float32)

        disp_16x = cv2.resize(disp, (64, 64)) / 16.0
        disp_8x = cv2.resize(disp, (128, 128)) / 8.0
        disp_4x = cv2.resize(disp, (256, 256)) / 4.0

        # Expand dimensions to add channel dimension
        disp = np.expand_dims(disp, axis=0)        # [1, H, W]
        disp_16x = np.expand_dims(disp_16x, axis=0)  # [1, H/16, W/16]
        disp_8x = np.expand_dims(disp_8x, axis=0)    # [1, H/8, W/8]
        disp_4x = np.expand_dims(disp_4x, axis=0)    # [1, H/4, W/4]

        return disp_16x, disp_8x, disp_4x, disp

    def __getitem__(self, idx):
        """
        Retrieves the processed data sample at the specified index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (inputs, targets) where inputs and targets are dictionaries containing tensors.
        """
        left_path = self.left_paths[idx]
        right_path = self.right_paths[idx]
        disp_path = self.disp_paths[idx]

        left, dx, dy = self.read_left(left_path)
        right = self.read_right(right_path)
        disp_16x, disp_8x, disp_4x, disp = self.read_disp(disp_path)

        # Convert to torch tensors
        left = torch.from_numpy(left).float()
        dx = torch.from_numpy(dx).float()
        dy = torch.from_numpy(dy).float()
        right = torch.from_numpy(right).float()
        disp_16x = torch.from_numpy(disp_16x).float()
        disp_8x = torch.from_numpy(disp_8x).float()
        disp_4x = torch.from_numpy(disp_4x).float()
        disp = torch.from_numpy(disp).float()

        inputs = {'left': left, 'right': right, 'dx': dx, 'dy': dy}
        targets = {'disp_16x': disp_16x, 'disp_8x': disp_8x, 'disp_4x': disp_4x, 'disp': disp}

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
        trial,             
        min_disp=-128,
        max_disp=64,
        patience=10,
        min_delta=0.0
    ):
    """
    Trains the model with given hyperparameters, implements early stopping, and integrates Optuna pruning.
    
    Args:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        device (str): Device to train on.
        hyperparameters (dict): Dictionary containing hyperparameters.
        trial (optuna.Trial): Optuna trial object for pruning.
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
        
        # Report intermediate objective value to Optuna and check for pruning
 
        trial.report(avg_epe, epoch)
        
        # Check if the trial should be pruned
        if trial.should_prune():
            print("Trial was pruned.")
            raise optuna.exceptions.TrialPruned()
    
    print(f"Training Completed. Best Epoch: {best_epoch}, Best Validation Loss: {best_val_loss:.4f}, Best EPE: {best_epe:.4f}, Best D1: {best_d1:.4f}")
    return best_val_loss, best_epe, best_d1
def objective(trial):
    # Define hyperparameter search space
    lr = trial.suggest_float('lr', 1e-5, 1e-4, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 6,8])
    lambda_refine = trial.suggest_categorical('lambda_refine', [0.6, 0.7, 0.8, 0.9, 1.0])
    weight_decay = trial.suggest_categorical('weight_decay', [0.0001, 0.00001])
    pct_start = trial.suggest_float('pct_start', 0.01, 0.3, log=True)
    epochs = 60  # To save time, use a smaller number of epochs for tuning

    # Define data directories
    train_dir = "/mnt/data1/crlj/Dataset/WHU-Stereo/experimental data/with ground truth/train"
    val_dir = "/mnt/data1/crlj/Dataset/WHU-Stereo/experimental data/with ground truth/val"

    # Subdirectories
    left_subdir = "left"
    right_subdir = "right"
    disp_subdir = "disp"

    # Get training file paths
    train_left_paths = list_image_files(os.path.join(train_dir, left_subdir))
    train_right_paths = list_image_files(os.path.join(train_dir, right_subdir))
    train_disp_paths = list_image_files(os.path.join(train_dir, disp_subdir))

    # Get validation file paths
    val_left_paths = list_image_files(os.path.join(val_dir, left_subdir))
    val_right_paths = list_image_files(os.path.join(val_dir, right_subdir))
    val_disp_paths = list_image_files(os.path.join(val_dir, disp_subdir))

    # Create datasets
    train_dataset = StereoDataset(train_left_paths, train_right_paths, train_disp_paths)
    val_dataset = StereoDataset(val_left_paths, val_right_paths, val_disp_paths)

    # Create data loaders based on suggested batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Define hyperparameters dictionary
    hyperparameters = {
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'lambda_refine': lambda_refine,
        'lambda1': 1.0,  # Fixed value
        'lambda2': 0.7,  # Fixed value
        'lambda3': 0.5,  # Fixed value
        'weight_decay': weight_decay,
        'pct_start': pct_start,
    }

    # Train the model and get the best validation metrics
    best_val_loss, best_epe, best_d1 = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        device="cuda" if torch.cuda.is_available() else "cpu",
        hyperparameters=hyperparameters,
        trial=trial,                  # Pass the trial object
        min_disp=-128,
        max_disp=64
    )

    # Return EPE as the optimization objective
    return best_epe
if __name__ == "__main__":
    # Define the pruner
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=1)

    # Create Optuna study with the pruner
    study = optuna.create_study(direction='minimize', pruner=pruner)

    # Start optimization
    study.optimize(objective, n_trials=30, timeout=None)  # You can set a timeout if desired

    # Print the best trial
    print("Best trial:")
    trial = study.best_trial

    print(f"  Validation Loss: {trial.value}")  
    print("  Parameters:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Optional: Visualize the optimization history
    try:
        import matplotlib.pyplot as plt
        fig = optuna.visualization.plot_optimization_history(study)
        fig.show()
    except ImportError:
        print("matplotlib is not installed, cannot display plots.")
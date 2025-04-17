from mynet import CAStereo
import torch
import os
from tqdm import tqdm
from dataloader import load_batch  # 假设您有自定义的数据加载函数

# 计算 End-Point Error (EPE)
def compute_epe(pred, gt, min_disp, max_disp):
    mask = (gt >= min_disp) & (gt < max_disp)
    abs_error = torch.abs(pred - gt)
    masked_error = abs_error[mask]
    error = masked_error.sum().item()
    nums = mask.sum().item()
    epe = error / nums if nums > 0 else 0.0
    return error, nums, epe

# D1 computation（默认阈值为 3px）
def compute_d1(pred, gt, min_disp, max_disp, threshold=3.0):
    mask = (gt >= min_disp) & (gt < max_disp)
    abs_error = torch.abs(pred - gt)
    err_mask = (abs_error > threshold) & mask
    err_disps = err_mask.sum().item()
    nums = mask.sum().item()
    d1 = err_disps / nums if nums > 0 else 0.0
    return err_disps, nums, d1

# 定义验证函数，并增加阈值 >1,2,3,5px 的错误比例计算
def validate_model(val_loader, model_path, device="cuda", min_disp=-128, max_disp=64, batch_size=4):
    model = CAStereo(min_disp, max_disp).to(device)
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    val_epe, val_d1 = 0.0, 0.0
    total_epe_nums, total_d1_nums = 0, 0

    # 新增：用于累加各阈值（1,2,3,5px）下的错误像素数和有效像素数
    threshold_error_counts = {1: 0, 2: 0, 3: 0, 5: 0}
    total_valid_pixels = 0

    total_samples = len(val_left_paths)  # 总样本数（全局变量，假设在 main 中定义）
    steps_per_epoch = total_samples // batch_size + (1 if total_samples % batch_size != 0 else 0)  # 每个 epoch 的批次数

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, total=steps_per_epoch, desc="Validation")):
            if batch_idx >= steps_per_epoch:
                break

            # 获取当前批次的数据
            lefts, rights, dxs, dys = [torch.tensor(x).to(device) for x in batch[0]]
            _, _, _, ds = [torch.tensor(x).to(device) for x in batch[1]]

            # 前向传播得到 refine 输出
            refine = model(lefts, rights, dxs, dys)[-1]
            
            # 检查 refine 是否为空或形状不对
            if refine is None or refine.shape[0] == 0:
                print("Error: refine output is empty or None")
                continue  # 如果 refine 无效，跳过当前批次

            # 计算 EPE 和 D1 指标
            epe_error, epe_nums, avg_epe = compute_epe(refine, ds, min_disp, max_disp)
            d1_error, d1_nums, avg_d1 = compute_d1(refine, ds, min_disp, max_disp)

            # 累加总的 EPE 和 D1 错误
            val_epe += epe_error
            val_d1 += d1_error
            total_epe_nums += epe_nums
            total_d1_nums += d1_nums

            # 计算每个阈值的错误比例
            mask = (ds >= min_disp) & (ds < max_disp)
            abs_error = torch.abs(refine - ds)
            valid_count = mask.sum().item()
            total_valid_pixels += valid_count

            for thresh in [1, 2, 3, 5]:
                err_count = ((abs_error > thresh) & mask).sum().item()
                threshold_error_counts[thresh] += err_count

    # 在所有批次加载完后计算最终的验证结果
    avg_val_epe = val_epe / total_epe_nums if total_epe_nums > 0 else 0.0
    avg_val_d1 = val_d1 / total_d1_nums if total_d1_nums > 0 else 0.0

    # 计算各阈值的错误比例
    ratio_thresh = {thresh: threshold_error_counts[thresh] / total_valid_pixels if total_valid_pixels > 0 else 0.0
                    for thresh in [1, 2, 3, 5]}

    # 输出最终的验证结果
    print(f"Validation Results - Avg EPE: {avg_val_epe:.4f}, Avg D1: {avg_val_d1:.4f}")
    print("Threshold Error Ratios:")
    print(f">1px: {ratio_thresh[1]:.4f}, >2px: {ratio_thresh[2]:.4f}, >3px: {ratio_thresh[3]:.4f}, >5px: {ratio_thresh[5]:.4f}")

if __name__ == "__main__":
    def list_image_files(directory, extensions=(".tiff", ".tif")):
        """
        列出指定目录中的所有图像文件，仅限指定扩展名。
        """
        return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(extensions)])

    # 数据路径
    val_dir = "/mnt/data1/crlj/Dataset/WHU-Stereo/experimental data/with ground truth/test"

    # 左图、右图和视差图子文件夹
    left_subdir = "left"
    right_subdir = "right"
    disp_subdir = "disp"

    # 获取验证集文件路径
    val_left_paths = list_image_files(os.path.join(val_dir, left_subdir))
    val_right_paths = list_image_files(os.path.join(val_dir, right_subdir))
    val_disp_paths = list_image_files(os.path.join(val_dir, disp_subdir))

    # 打印调试信息
    print(f"Number of validation left images: {len(val_left_paths)}")
    print(f"Number of validation disparity images: {len(val_disp_paths)}")

    # 批量大小
    batch_size = 1

    # 创建数据加载器
    val_loader = load_batch(val_left_paths, val_right_paths, val_disp_paths, batch_size=batch_size, reshuffle=False)

    # 模型路径
    model_path = "/home/crlj/bh/CAtest/best_checkpoint.pth"  # 请根据需要修改模型路径

    # 调用验证函数
    validate_model(val_loader, model_path, device="cuda", min_disp=-128, max_disp=64, batch_size=batch_size)
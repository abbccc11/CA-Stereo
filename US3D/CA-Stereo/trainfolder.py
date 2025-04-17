import os
import glob

def find_missing_files_in_train_folder(train_folder):
    """
    在 train_folder 中查找哪些前缀缺少 _LEFT_RGB, _RIGHT_RGB 或 _LEFT_AGL 文件。
    并打印缺失信息。
    """
    # 1. 收集三类文件的完整路径
    left_images  = sorted(glob.glob(os.path.join(train_folder, '*_LEFT_RGB.tif')))
    right_images = sorted(glob.glob(os.path.join(train_folder, '*_RIGHT_RGB.tif')))
    disp_images  = sorted(glob.glob(os.path.join(train_folder, '*_LEFT_AGL.tif')))

    print("Len of left_images:", len(left_images))
    print("Len of right_images:", len(right_images))
    print("Len of disp_images:", len(disp_images))

    # 2. 从每个文件路径中提取“前缀”，即去掉末尾的 "_LEFT_RGB.tif" 等后缀
    def get_prefix(path, suffix):
        return path.rsplit(suffix, 1)[0]

    prefix_left  = set(get_prefix(f, "_LEFT_RGB.tif") for f in left_images)
    prefix_right = set(get_prefix(f, "_RIGHT_RGB.tif") for f in right_images)
    prefix_disp  = set(get_prefix(f, "_LEFT_AGL.tif") for f in disp_images)

    # 3. 求所有出现过的前缀并逐一检查缺失情况
    all_prefixes = prefix_left | prefix_right | prefix_disp  # 并集

    missing_list = []  # 用于收集缺失信息
    for prefix in sorted(all_prefixes):
        missing_types = []
        if prefix not in prefix_left:
            missing_types.append("LEFT_RGB")
        if prefix not in prefix_right:
            missing_types.append("RIGHT_RGB")
        if prefix not in prefix_disp:
            missing_types.append("LEFT_AGL")

        if missing_types:
            missing_list.append((prefix, missing_types))

    # 4. 打印缺失报告
    if not missing_list:
        print("所有样本都齐全，没有缺失。")
    else:
        print("\n以下前缀缺少对应文件：")
        for prefix, missing_types in missing_list:
            print(f"  Prefix: {prefix}\n    Missing: {', '.join(missing_types)}")

if __name__ == "__main__":
    train_folder = "/mnt/data1/crlj/Dataset/US3D/testfolder"  # 修改为实际的 train_folder
    find_missing_files_in_train_folder(train_folder)
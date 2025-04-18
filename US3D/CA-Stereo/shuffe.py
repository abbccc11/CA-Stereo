import os
import shutil
import pandas as pd

# 读取CSV文件
csv_file = r"/home/crlj/bh/crlj/US3D/Mynet/val.csv"  # 替换为你的CSV文件路径
df = pd.read_csv(csv_file)

# 定义目标文件夹路径
target_folder = r"/mnt/data1/crlj/Dataset/US3D/val"  # 替换为目标文件夹路径

# 确保目标文件夹存在
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# 遍历CSV文件中的每一行
for index, row in df.iterrows():
    for column in ['left_img', 'right_img', 'gt_disp']:
        file_path = row[column]
        if os.path.exists(file_path):
            # 构造目标文件的路径
            target_path = os.path.join(target_folder, os.path.basename(file_path))
            
            # 复制文件到目标文件夹
            shutil.copy(file_path, target_path)
            
            # 删除原文件
            os.remove(file_path)
            print(f"文件 {file_path} 已复制并删除。")
        else:
            print(f"文件 {file_path} 不存在。")
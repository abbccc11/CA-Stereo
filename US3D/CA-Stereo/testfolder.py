import os

# =============== 请根据实际情况修改 ===============
folderA = r"/mnt/data1/crlj/Dataset/US3D/trainfolder"
folderB = r"/mnt/data1/crlj/Dataset/US3D/testfolder"
# =================================================

# 1. 获取 folderB 中所有文件的文件名 (不含目录)
#    这里用 os.listdir(folderB) 拿到所有条目，然后再判断是不是文件
files_in_B = set()
for entry in os.listdir(folderB):
    path_b = os.path.join(folderB, entry)
    if os.path.isfile(path_b):
        files_in_B.add(entry)

# 2. 遍历 folderA，如果文件名在 files_in_B，则删除
count_deleted = 0
for entry in os.listdir(folderA):
    path_a = os.path.join(folderA, entry)
    if os.path.isfile(path_a):
        # 如果该文件的文件名在 B 文件名集合中，就删除
        if entry in files_in_B:
            os.remove(path_a)
            count_deleted += 1
            print(f"Deleted {path_a}")

print(f"Done. Total deleted files: {count_deleted}")
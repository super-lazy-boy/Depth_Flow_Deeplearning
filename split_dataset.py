# KITTI没有公开的验证集标签，这里使用训练集进行验证

import os
from shutil import copy,rmtree
import random

def mk_file(file_path: str):
    if os.path.exists(file_path):
        # if the folder exists, delete it and create a new one
        rmtree(file_path)
    os.makedirs(file_path)

if __name__ == "__main__":
    random.seed(42)

    split_ratio = 0.1  # 10% 用作验证集
    base_path = "data/KITTI/training"
    base_image_path = os.path.join(base_path, "image_2")
    base_flow_path = os.path.join(base_path, "flow_occ")

    # 创建新的数据集的目录结构
    new_path = os.path.join("data/KITTI_splited")
    new_train_path = os.path.join(new_path, "training")
    new_val_path = os.path.join(new_path, "testing")
    mk_file(new_path)
    mk_file(new_train_path)
    mk_file(new_val_path)

    for subfolder in ["image_2", "flow_occ"]:
        mk_file(os.path.join(new_train_path, subfolder))
        mk_file(os.path.join(new_val_path, subfolder))

    images = os.listdir(base_image_path)
    num = len(images)

    for index ,file_name in enumerate(images):
        if file_name.endswith("_10.png"):
            # 决定这个样本是放在训练集还是验证集
            if random.random() < split_ratio:
                dest_image_folder = os.path.join(new_val_path, "image_2")
                dest_flow_folder = os.path.join(new_val_path, "flow_occ")
            else:
                dest_image_folder = os.path.join(new_train_path, "image_2")
                dest_flow_folder = os.path.join(new_train_path, "flow_occ")

            # 复制图像对
            img1_name = file_name
            img2_name = file_name.replace("_10.png", "_11.png")
            flow_name = file_name.replace("_10.png", "_10.png")  # 光流文件名与第一个图像对应

            copy(os.path.join(base_image_path, img1_name), os.path.join(dest_image_folder, img1_name))
            copy(os.path.join(base_image_path, img2_name), os.path.join(dest_image_folder, img2_name))
            copy(os.path.join(base_flow_path, flow_name), os.path.join(dest_flow_folder, flow_name))
        print("\r processing [{}/{}]".format(index+1, num), end="")  # processing bar

    print("processing done!")
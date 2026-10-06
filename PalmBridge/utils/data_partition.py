import shutil
import os

# 假设您的原始数据集路径是这个
source_dir = 'G:\dataset\palmprint\\NIR_ind'
# 您想要复制到的新路径，用于训练数据
train_destination_dir = 'G:\dataset\palmprint\\NIR_ind\\train_dataset'
# 您想要复制到的新路径，用于验证数据
val_destination_dir = 'G:\dataset\palmprint\\NIR_ind\\val_dataset'

# 确保目标文件夹存在
if not os.path.exists(train_destination_dir):
    os.makedirs(train_destination_dir)
if not os.path.exists(val_destination_dir):
    os.makedirs(val_destination_dir)

## Tongji

# # 获取所有图片文件并按名称排序
# all_images = sorted([f for f in os.listdir(source_dir) if f.endswith('.bmp')])
#
# # 每个类别的图片数量
# images_per_class = 10
#
# # 遍历每个类别
# for i in range(0, len(all_images), images_per_class):
#     # 计算类别的起始和结束索引
#     start_index = i
#     end_index = min(i + images_per_class, len(all_images))
#
#     # 提取类别名称
#     class_name = str(i // images_per_class)
#
#     # 复制前3张图片到训练目录
#     for j in range(start_index, min(start_index + 3, end_index)):
#         file_name = all_images[j]
#         file_path = os.path.join(source_dir, file_name)
#         new_file_name = f"{class_name}_{j - start_index + 1:03d}.png"
#         destination_file_path = os.path.join(train_destination_dir, class_name, new_file_name)
#         # 确保目标类别文件夹存在
#         if not os.path.exists(os.path.dirname(destination_file_path)):
#             os.makedirs(os.path.dirname(destination_file_path))
#         shutil.copy(file_path, destination_file_path)
#         print(f"Copied {file_path} to {destination_file_path}")
#
#     # 复制后3张图片到验证目录
#     for j in range(max(end_index - 3, start_index), end_index):
#         file_name = all_images[j]
#         file_path = os.path.join(source_dir, file_name)
#         new_file_name = f"{class_name}_{j - start_index + 1:03d}.png"
#         destination_file_path = os.path.join(val_destination_dir, class_name, new_file_name)
#         # 确保目标类别文件夹存在
#         if not os.path.exists(os.path.dirname(destination_file_path)):
#             os.makedirs(os.path.dirname(destination_file_path))
#         shutil.copy(file_path, destination_file_path)
#         print(f"Copied {file_path} to {destination_file_path}")

## IITD
# 获取所有图片文件并按名称排序
all_images = sorted([f for f in os.listdir(source_dir) if f.endswith('.jpg')])

# 遍历每个类别
for i in range(0, len(all_images), 12):  # 假设每个类别有5张图片
    # 计算类别的起始和结束索引
    start_index = i
    end_index = min(i + 12, len(all_images))

    # 提取类别名称
    class_name = int(all_images[start_index].split('_')[0])-1

    # 确保目标类别文件夹存在
    train_class_dir = os.path.join(train_destination_dir, str(class_name))
    val_class_dir = os.path.join(val_destination_dir, str(class_name))
    if not os.path.exists(train_class_dir):
        os.makedirs(train_class_dir)
    if not os.path.exists(val_class_dir):
        os.makedirs(val_class_dir)

    # 复制前一半图片到训练目录
    for j in range(start_index, start_index + 4):  # 只复制前2张图片
        file_name = all_images[j]
        file_path = os.path.join(source_dir, file_name)
        shutil.copy(file_path, train_class_dir)
        print(f"Copied {file_path} to {train_class_dir}")

    # 复制后一半图片到验证目录
    for j in range(start_index + 8 , end_index):  # 只复制后2张图片
        file_name = all_images[j]
        file_path = os.path.join(source_dir, file_name)
        shutil.copy(file_path, val_class_dir)
        print(f"Copied {file_path} to {val_class_dir}")
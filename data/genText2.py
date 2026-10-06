import os

# 路径设置
image_folder = 'F:\palmprint recognition\dataset\IITD\IITD_ind'
output_txt_first = 'F:\palmprint recognition\CCNet-main\data\\train_IITD_0.5.txt'
output_txt_last = 'F:\palmprint recognition\CCNet-main\data\\test_IITD_0.5.txt'

num_classes_each = 230

# Step 1: 获取所有类别编号
all_class_ids = set()
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        try:
            class_id = int(filename.split('_')[0])
            all_class_ids.add(class_id)
        except ValueError:
            continue

# Step 2: 排序并分成两部分
sorted_class_ids = sorted(all_class_ids)
first_230_ids = sorted_class_ids[:num_classes_each]
last_230_ids = sorted_class_ids[-num_classes_each:]

# Step 3: 创建映射表（类别编号 → 连续label，从1开始）
first_class_map = {cid: idx + 1 for idx, cid in enumerate(first_230_ids)}
last_class_map = {cid: idx + 231 for idx, cid in enumerate(last_230_ids)}

# Step 4: 遍历文件并按类别写入对应列表
first_data = []
last_data = []

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        try:
            class_id = int(filename.split('_')[0])
        except ValueError:
            continue

        full_path = os.path.join(image_folder, filename)

        if class_id in first_class_map:
            label = first_class_map[class_id]
            first_data.append(f"{full_path} {label}")
        elif class_id in last_class_map:
            label = last_class_map[class_id]
            last_data.append(f"{full_path} {label}")

# Step 5: 写入两个txt文件
with open(output_txt_first, 'w') as f:
    for line in first_data:
        f.write(line + '\n')

with open(output_txt_last, 'w') as f:
    for line in last_data:
        f.write(line + '\n')

print(f"写入完成：前230类 {len(first_data)} 条，后230类 {len(last_data)} 条。")

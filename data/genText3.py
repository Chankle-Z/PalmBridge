import os
from collections import defaultdict

# === 用户配置 ===
image_dir = r"F:\CASIAM\460"   # 图片文件夹路径
output_dir = r"F:\CASIAM\data"  # 输出txt文件目录
os.makedirs(output_dir, exist_ok=True)

# === Step 1: 读取文件 ===
all_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]

# 数据结构：{种类: {id: [图片路径列表]}}
data = defaultdict(lambda: defaultdict(list))

for fname in all_files:
    parts = fname[:-4].split('_')

    # 兼容 “ROI_001_l_460_01.jpg” 这种格式
    if len(parts) == 5 and parts[0].lower() == "roi":
        _, id_str, side, category, num = parts
    elif len(parts) == 4:
        id_str, side, category, num = parts
    else:
        print(f"跳过命名异常文件: {fname}")
        continue

    try:
        id_int = int(id_str) - 1
    except ValueError:
        print(f"跳过ID异常文件: {fname}")
        continue

    # 若为右手（r），id + 100
    if side.lower() == 'r':
        id_int += 100

    img_path = os.path.join(image_dir, fname)
    data[category][id_int].append(img_path)

# === Step 2: 写出 train/test 文件 ===
for category, id_dict in data.items():
    train_file = os.path.join(output_dir, f"train_{category}.txt")
    test_file = os.path.join(output_dir, f"test_{category}.txt")

    with open(train_file, 'w') as f_train, open(test_file, 'w') as f_test:
        for id_val, imgs in sorted(id_dict.items()):
            imgs.sort()
            n = len(imgs)
            half = n // 2

            for img_path in imgs[:half]:
                f_train.write(f"{img_path} {id_val}\n")

            for img_path in imgs[half:]:
                f_test.write(f"{img_path} {id_val}\n")

print("✅ 数据集划分完成！")
print(f"输出路径：{output_dir}")

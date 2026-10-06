from collections import defaultdict

# 输入输出文件路径
input_file = 'C:\\Users\qiaoy\Desktop\\train_MSNIR.txt'
output_file1 = 'C:\\Users\qiaoy\Desktop\\test_half1.txt'
output_file2 = 'C:\\Users\qiaoy\Desktop\\test_half2.txt'

# 用 defaultdict 按类别分组
data_dict = defaultdict(list)

# 读取原始文件并分组
with open(input_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split()
            if len(parts) == 2:
                path, label = parts
                data_dict[int(label)].append(line)

# 分别保存前一半和后一半
with open(output_file1, 'w') as f1, open(output_file2, 'w') as f2:
    for label in sorted(data_dict.keys()):
        samples = data_dict[label]
        mid = len(samples) // 2
        front_half = samples[:mid]
        back_half = samples[mid:]

        for line in front_half:
            f1.write(line + '\n')
        for line in back_half:
            f2.write(line + '\n')

print("处理完成！前一半保存为 train_half1.txt，后一半保存为 train_half2.txt")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把原始 list.txt 按类别号拆成
  train_0_249.txt   类别 0~249
  train_250_499.txt 类别 250~499
"""

import os

SRC_FILE = 'C:\\Users\qiaoy\Desktop\\test_MSNIR.txt'          # 原始文件
OUT_0 = 'C:\\Users\qiaoy\Desktop\\test_0_249.txt'
OUT_1 = 'C:\\Users\qiaoy\Desktop\\test_250_499.txt'

def split_by_250(src, dst0, dst1):
    with open(src, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    grp0, grp1 = [], []
    for line in lines:
        line = line.rstrip('\n')
        if not line:
            continue
        # 取最后一个空格后的数字作为类别号
        cls = int(line.strip().split()[-1])
        if 0 <= cls <= 249:
            grp0.append(line)
        elif 250 <= cls <= 499:
            grp1.append(line)
        # 类别号≥500 的行直接忽略，如需保留可自行调整

    # 写入
    with open(dst0, 'w', encoding='utf-8') as f:
        f.write('\n'.join(grp0) + '\n')
    with open(dst1, 'w', encoding='utf-8') as f:
        f.write('\n'.join(grp1) + '\n')

    print(f'类别 0~249：{len(grp0)} 条 → {OUT_0}')
    print(f'类别 250~499：{len(grp1)} 条 → {OUT_1}')

if __name__ == '__main__':
    split_by_250(SRC_FILE, OUT_0, OUT_1)
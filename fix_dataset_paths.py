#!/usr/bin/env python3
"""
修复dataset_split.json中的路径格式 - Linux路径转Windows路径
"""
import json
import os
from pathlib import Path

# 读取原始split文件
split_file = Path("outputs/segmentation/module0/dataset_split.json")
with open(split_file, 'r') as f:
    data = json.load(f)

# 转换路径函数
def convert_path(linux_path):
    # 从 /home/deepcare4090/jh/Orthodontics/datasets/segmentation_dataset/xxx.vtp
    # 转为 datasets/segmentation_dataset/xxx.vtp
    filename = os.path.basename(linux_path)
    return f"datasets/segmentation_dataset/{filename}"

# 转换所有路径
for split_type in ['train', 'val']:
    if split_type in data:
        data[split_type] = [convert_path(path) for path in data[split_type]]

# 保存修复后的文件
output_file = Path("outputs/segmentation/module0/dataset_split_fixed.json")
with open(output_file, 'w') as f:
    json.dump(data, f, indent=2)

print(f"✅ 路径修复完成！")
print(f"原始文件: {split_file}")
print(f"修复后文件: {output_file}")
print(f"训练集: {len(data['train'])} 个文件")
print(f"验证集: {len(data['val'])} 个文件")

# 验证文件是否存在
missing_files = []
for split_type in ['train', 'val']:
    for file_path in data[split_type]:
        full_path = Path(file_path)
        if not full_path.exists():
            missing_files.append(str(full_path))

if missing_files:
    print(f"⚠️  警告: {len(missing_files)} 个文件不存在")
    for f in missing_files[:5]:  # 只显示前5个
        print(f"  - {f}")
    if len(missing_files) > 5:
        print(f"  ... 还有 {len(missing_files)-5} 个文件")
else:
    print("✅ 所有数据文件都存在！")
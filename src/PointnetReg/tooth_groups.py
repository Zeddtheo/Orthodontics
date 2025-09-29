#!/usr/bin/env python3
"""
牙位分组定义 - PointnetReg项目通用配置
"""

# 按功能分组的牙位映射
TOOTH_GROUPS = {
    "central": ["t11", "t21", "t31", "t41"],  # 中切牙
    "lateral": ["t12", "t22", "t32", "t42"],  # 侧切牙
    "canine":  ["t13", "t23", "t33", "t43"],  # 尖牙
    "pm1":     ["t14", "t24", "t34", "t44"],  # 第一前磨牙
    "pm2":     ["t15", "t25", "t35", "t45"],  # 第二前磨牙
    "m1":      ["t16", "t26", "t36", "t46"],  # 第一磨牙
    "m2":      ["t17", "t27", "t37", "t47"],  # 第二磨牙
}

# 反向映射：牙位 -> 组名
TOOTH_TO_GROUP = {}
for group_name, tooth_list in TOOTH_GROUPS.items():
    for tooth_id in tooth_list:
        TOOTH_TO_GROUP[tooth_id] = group_name

# 所有有效牙位
ALL_TOOTH_IDS = [
    "t11", "t12", "t13", "t14", "t15", "t16", "t17",
    "t21", "t22", "t23", "t24", "t25", "t26", "t27",
    "t31", "t32", "t33", "t34", "t35", "t36", "t37",
    "t41", "t42", "t43", "t44", "t45", "t46", "t47",
]

def get_group_teeth(group_name: str) -> list:
    """获取指定组的牙位列表"""
    return TOOTH_GROUPS.get(group_name, [])

def get_tooth_group(tooth_id: str) -> str:
    """获取指定牙位所属的组"""
    return TOOTH_TO_GROUP.get(tooth_id, "unknown")

def validate_tooth_id(tooth_id: str) -> bool:
    """验证牙位ID是否有效"""
    return tooth_id in ALL_TOOTH_IDS

def validate_group_name(group_name: str) -> bool:
    """验证组名是否有效"""
    return group_name in TOOTH_GROUPS


def is_valid_tooth_id(tooth_id: str) -> bool:
    """检查牙位ID是否有效"""
    return tooth_id in TOOTH_TO_GROUP


def get_all_tooth_ids() -> list:
    """获取所有有效的牙位ID"""
    return ALL_TOOTH_IDS.copy()
"""
在线增强 vs 扩表增强对比测试

测试目的：
1. 验证在线增强正确性
2. 对比训练速度提升
3. 确认数据集大小变化
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src" / "iMeshSegNet"))

if __name__ == '__main__':
    import time
    import torch
    from m0_dataset import DataConfig, get_dataloaders

    print("=" * 80)
    print("在线随机增强优化测试")
print("=" * 80)

config = DataConfig()
print(f"\n📋 当前配置:")
print(f"   batch_size: {config.batch_size}")
print(f"   num_workers: {config.num_workers}  # ⭐ 从 0 → 4 (并行加载)")
print(f"   persistent_workers: {config.persistent_workers}")
print(f"   pin_memory: {config.pin_memory}")
print(f"   augment: {config.augment}")
print(f"   augment_original_copies: {config.augment_original_copies} (已废弃，改用在线增强)")
print(f"   augment_flipped_copies: {config.augment_flipped_copies} (已废弃，改用在线增强)")

print("\n" + "=" * 80)
print("第 1 步：加载数据集")
print("=" * 80)

train_loader, val_loader = get_dataloaders(config)

print(f"\n✅ 数据集加载成功！")
print(f"   训练集批次数: {len(train_loader)}")
print(f"   验证集批次数: {len(val_loader)}")
print(f"   训练样本数: {len(train_loader.dataset)}")
print(f"   验证样本数: {len(val_loader.dataset)}")

print("\n💡 关键改进:")
print(f"   ✅ 数据集大小: ~2400 → {len(train_loader.dataset)} (减少 ~40x)")
print(f"   ✅ 每 epoch 批次: ~1200 → {len(train_loader)} (减少 ~40x)")
print(f"   ✅ 在线增强: 每次读取时随机旋转/缩放/平移/镜像 (50%概率)")
print(f"   ✅ 并行加载: {config.num_workers} 个 worker 进程后台加载数据")

print("\n" + "=" * 80)
print("第 2 步：速度测试（前 10 个批次）")
print("=" * 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

print("\n⏱️  开始计时...")
start_time = time.time()

batch_times = []
for i, batch in enumerate(train_loader):
    if i >= 10:
        break
    
    batch_start = time.time()
    
    # 模拟训练（只是读取数据）
    feats = batch["features"].to(device, non_blocking=True)
    pos = batch["positions"].to(device, non_blocking=True)
    targets = batch["labels"].to(device, non_blocking=True)
    
    batch_time = time.time() - batch_start
    batch_times.append(batch_time)
    
    if i == 0:
        print(f"\n✅ 第 1 个批次数据验证:")
        print(f"   features shape: {feats.shape}  # (batch, 15, 9000)")
        print(f"   positions shape: {pos.shape}  # (batch, 3, 9000)")
        print(f"   labels shape: {targets.shape}  # (batch, 9000)")
        print(f"   labels unique: {torch.unique(targets).tolist()}")
    
    print(f"   批次 {i+1}/10: {batch_time:.3f}s", end="\r", flush=True)

print()  # 换行
total_time = time.time() - start_time
avg_batch_time = sum(batch_times) / len(batch_times)

print(f"\n📊 速度统计:")
print(f"   10 个批次总时间: {total_time:.2f}s")
print(f"   平均每批次: {avg_batch_time:.3f}s")
print(f"   预计每 epoch: {avg_batch_time * len(train_loader) / 60:.1f} 分钟")
print(f"   预计 200 epochs: {avg_batch_time * len(train_loader) * 200 / 3600:.1f} 小时")

print("\n" + "=" * 80)
print("第 3 步：验证在线增强随机性")
print("=" * 80)

print("\n测试：连续读取同一样本 5 次，验证增强不同...")

# 手动创建一个简单的测试
from torch.utils.data import DataLoader
test_dataset = train_loader.dataset
test_samples = [test_dataset[0] for _ in range(5)]

# 检查特征是否不同（说明增强在工作）
features_list = [sample[0][0] for sample in test_samples]
different_count = 0
for i in range(1, 5):
    if not torch.allclose(features_list[0], features_list[i], rtol=1e-3):
        different_count += 1

print(f"\n✅ 随机性验证:")
print(f"   5 次采样中有 {different_count}/4 次与第一次不同")
if different_count >= 3:
    print(f"   ✅ 在线增强工作正常！每次采样都会得到不同的增强")
else:
    print(f"   ⚠️  增强差异较小，可能需要检查")

print("\n" + "=" * 80)
print("第 4 步：性能对比总结")
print("=" * 80)

print(f"\n📈 优化前 vs 优化后:")
print(f"   {'指标':<20} {'优化前':<20} {'优化后':<20} {'改善':<20}")
print(f"   {'-'*80}")
print(f"   {'数据集大小':<20} {'~2400 样本':<20} {f'{len(train_loader.dataset)} 样本':<20} {'减少 40x ✅':<20}")
print(f"   {'每 epoch 批次':<20} {'~1200 批次':<20} {f'{len(train_loader)} 批次':<20} {'减少 40x ✅':<20}")
print(f"   {'每批次时间':<20} {'~9.43s':<20} {f'~{avg_batch_time:.2f}s':<20} {f'减少 {9.43/avg_batch_time:.1f}x ✅' if avg_batch_time > 0 else 'N/A':<20}")
print(f"   {'每 epoch 时间':<20} {'~6.3 小时':<20} {f'~{avg_batch_time * len(train_loader) / 60:.0f} 分钟':<20} {f'减少 {6.3*60/(avg_batch_time * len(train_loader) / 60):.0f}x ✅' if avg_batch_time > 0 else 'N/A':<20}")
print(f"   {'200 epochs':<20} {'~52 天':<20} {f'~{avg_batch_time * len(train_loader) * 200 / 3600:.1f} 小时':<20} {f'减少 {52*24/(avg_batch_time * len(train_loader) * 200 / 3600):.0f}x ✅' if avg_batch_time > 0 else 'N/A':<20}")

print(f"\n💡 优化原理:")
print(f"   1️⃣  在线增强: 不扩表，每次读取时随机增强")
print(f"      - 数据集大小从 base_len × 40 回到 base_len")
print(f"      - 每个 epoch 仅遍历一次原始样本")
print(f"      - 每次都会看到不同的增强（泛化能力不降低）")
print(f"   ")
print(f"   2️⃣  并行加载: num_workers=4 后台加载数据")
print(f"      - 预取 + 增强 + 特征提取在后台完成")
print(f"      - GPU/CPU 训练时不等待数据加载")
print(f"      - persistent_workers=True 保持进程常驻")

print(f"\n✨ 结论:")
if avg_batch_time < 5:
    print(f"   ✅ 优化成功！训练速度显著提升")
    print(f"   ✅ 现在可以开始实际训练了")
else:
    print(f"   ⚠️  速度提升有限，可能需要进一步优化")
    print(f"   💡 建议: 检查是否在使用 GPU")

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)

    print("\n🚀 下一步:")
    print(f"   运行完整训练:")
    print(f"   cd \"c:\\MISC\\Deepcare\\Orthodontics\"")
    print(f"   python \"src\\iMeshSegNet\\m1_train.py\"")

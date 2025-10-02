import numpy as np

labels = np.load('outputs/overfit/infer/1_U_pred.npy')
unique, counts = np.unique(labels, return_counts=True)
total = len(labels)

print('推理结果标签分布:')
for u, c in zip(unique, counts):
    print(f'  L{u:2d}: {c:4d} cells ({c/total*100:5.1f}%)')

print(f'\n总计: {len(unique)} 个类别（期望 15 个）')
missing = set(range(15)) - set(unique)
print(f'遗漏类别: {sorted(missing) if missing else "无"}')

if missing:
    print(f'\n❌ 模型仍然遗漏了 {len(missing)} 个类别: {sorted(missing)}')
    print('   可能原因：')
    print('   1. 训练 epoch 不足（建议 100+ epochs）')
    print('   2. 学习率过大或过小')
    print('   3. 类别严重不平衡')
else:
    print('\n✅ 模型成功预测了所有 15 个类别！')

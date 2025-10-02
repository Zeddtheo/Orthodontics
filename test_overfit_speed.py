"""
快速测试 m5_overfit.py 的性能
"""
import subprocess
import time
import sys

print("=" * 60)
print("🧪 测试 m5_overfit.py 优化效果")
print("=" * 60)

# 测试 10 epochs 的时间
print("\n⏱️  测试 10 epochs 训练时间...")
start_time = time.time()

try:
    result = subprocess.run(
        [sys.executable, "src/iMeshSegNet/m5_overfit.py", "--sample", "1_U", "--epochs", "10"],
        capture_output=False,
        text=True,
        timeout=300  # 5分钟超时
    )
    
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✅ 测试成功!")
        print(f"⏱️  10 epochs 耗时: {elapsed:.1f} 秒")
        print(f"📊 平均每 epoch: {elapsed/10:.1f} 秒")
        print(f"🎯 预估 50 epochs: {elapsed*5:.1f} 秒 (~{elapsed*5/60:.1f} 分钟)")
    else:
        print(f"\n❌ 测试失败，返回码: {result.returncode}")
        
except subprocess.TimeoutExpired:
    elapsed = time.time() - start_time
    print(f"\n⚠️  训练超时 (>5分钟)")
    print(f"⏱️  已运行: {elapsed:.1f} 秒")
    print("💡 建议检查是否还有性能瓶颈")

except KeyboardInterrupt:
    print("\n⚠️  用户中断测试")

print("\n" + "=" * 60)

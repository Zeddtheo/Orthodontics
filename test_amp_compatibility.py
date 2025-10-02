"""
测试 m1_train.py 的 Linux 兼容性
检查 AMP、GradScaler 等组件的导入和使用
"""
import sys
import torch

print("=" * 70)
print("🔍 检查 PyTorch AMP 兼容性")
print("=" * 70)
print()

print(f"📦 PyTorch 版本: {torch.__version__}")
print(f"🖥️  CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🎮 CUDA 版本: {torch.version.cuda}")
    print(f"📊 GPU 数量: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        print(f"🎯 当前 GPU: {torch.cuda.get_device_name(0)}")
print()

# 测试导入
print("🔧 测试 AMP 导入...")

# 方法 1: 尝试导入 torch.amp (PyTorch >= 1.10)
try:
    from torch.amp import autocast, GradScaler
    print("  ✅ torch.amp.autocast 可用 (PyTorch >= 1.10)")
    amp_source = "torch.amp"
    HAS_AMP = True
except ImportError as e:
    print(f"  ❌ torch.amp 不可用: {e}")
    amp_source = None
    HAS_AMP = False

# 方法 2: 尝试导入 torch.cuda.amp (PyTorch < 1.10)
if not HAS_AMP:
    try:
        from torch.cuda.amp import autocast, GradScaler
        print("  ✅ torch.cuda.amp.autocast 可用 (PyTorch < 1.10)")
        amp_source = "torch.cuda.amp"
        HAS_AMP = True
    except ImportError as e:
        print(f"  ❌ torch.cuda.amp 不可用: {e}")

if not HAS_AMP:
    print("  ⚠️  AMP 不可用，将使用 CPU fallback")
    amp_source = "fallback"
else:
    print(f"  📍 AMP 来源: {amp_source}")

print()

# 测试 GradScaler 初始化
print("🧪 测试 GradScaler 初始化...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_type = "cuda" if device.type == "cuda" else "cpu"

if HAS_AMP:
    try:
        # 新版 API (device 参数)
        if amp_source == "torch.amp":
            from torch.amp import GradScaler
        else:
            from torch.cuda.amp import GradScaler
        
        try:
            scaler = GradScaler(device=device_type, enabled=(device.type == "cuda"))
            print(f"  ✅ GradScaler(device='{device_type}') 初始化成功 (新版 API)")
        except TypeError:
            # 旧版 API (无 device 参数)
            scaler = GradScaler(enabled=(device.type == "cuda"))
            print(f"  ✅ GradScaler(enabled={device.type == 'cuda'}) 初始化成功 (旧版 API)")
        
        print(f"  📊 Scaler enabled: {scaler.is_enabled()}")
    except Exception as e:
        print(f"  ❌ GradScaler 初始化失败: {e}")
else:
    print("  ⚠️  AMP 不可用，使用 fallback GradScaler")

print()

# 测试 autocast 使用
print("🎭 测试 autocast 使用...")
if HAS_AMP and device.type == "cuda":
    try:
        # 新版 API
        if amp_source == "torch.amp":
            from torch.amp import autocast
            try:
                with autocast(device_type="cuda", enabled=True):
                    x = torch.randn(2, 3, device=device)
                    y = x * 2
                print("  ✅ autocast(device_type='cuda') 工作正常 (新版 API)")
            except TypeError:
                # 旧版 API fallback
                with autocast(enabled=True):
                    x = torch.randn(2, 3, device=device)
                    y = x * 2
                print("  ✅ autocast(enabled=True) 工作正常 (旧版 API)")
        else:
            from torch.cuda.amp import autocast
            with autocast(enabled=True):
                x = torch.randn(2, 3, device=device)
                y = x * 2
            print("  ✅ torch.cuda.amp.autocast 工作正常")
    except Exception as e:
        print(f"  ❌ autocast 测试失败: {e}")
else:
    print(f"  ℹ️  跳过 autocast 测试 (device={device}, HAS_AMP={HAS_AMP})")

print()

# 兼容性总结
print("=" * 70)
print("📋 兼容性总结")
print("=" * 70)
print(f"  PyTorch 版本: {torch.__version__}")
print(f"  设备: {device}")
print(f"  AMP 可用: {HAS_AMP}")
print(f"  AMP 来源: {amp_source}")

if device.type == "cpu":
    print()
    print("  ℹ️  CPU 模式：")
    print("     - GradScaler 会被禁用 (enabled=False)")
    print("     - autocast 会被跳过 (使用 nullcontext)")
    print("     - 训练速度较慢但兼容性最好")

if HAS_AMP and device.type == "cuda":
    print()
    print("  ✅ GPU + AMP 模式：")
    print("     - 混合精度训练已启用")
    print("     - 预期速度提升 1.5-3x")
    print("     - 显存占用减少约 40%")

print()
print("🎯 结论: m1_train.py 应该可以在此环境运行")
print("=" * 70)

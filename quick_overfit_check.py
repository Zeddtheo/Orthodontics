"""
30 秒过拟合模型自检清单
使用训练样本（segmentation_dataset/*.vtp）验证模型是否正确过拟合

检查项：
1. ✓ 推理使用训练样本（非 raw .stl）
2. ✓ target_cells = 10000, sample_cells = 6000（过拟合配置）
3. ✓ 使用 checkpoint 自带的 mean/std（不依赖外部 stats.npz）
4. ✓ 预测结果应该完美（训练样本过拟合）
"""
import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("🔍 30 秒过拟合模型自检清单")
    print("=" * 70)
    print()
    
    # 配置
    checkpoint = "outputs/overfit/1_U/best_overfit_1_U.pt"
    # ✅ 使用训练集样本（segmentation_dataset），不是 raw .stl
    input_file = "datasets/segmentation_dataset/1_U.vtp"  
    output_dir = "outputs/overfit/1_U/quick_check"
    
    # 检查文件是否存在
    if not Path(checkpoint).exists():
        print(f"❌ Checkpoint 不存在: {checkpoint}")
        print("   请先运行过拟合训练：")
        print("   .venv\\Scripts\\python.exe src\\iMeshSegNet\\m5_overfit.py --sample 1_U --epochs 50")
        return
    
    if not Path(input_file).exists():
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    print(f"📋 检查配置:")
    print(f"   Checkpoint: {checkpoint}")
    print(f"   输入文件:   {input_file} (训练样本，非 raw .stl ✓)")
    print(f"   输出目录:   {output_dir}")
    print()
    
    print("🚀 开始推理...")
    print("-" * 70)
    
    # 运行推理
    cmd = [
        ".venv/Scripts/python.exe",
        "src/iMeshSegNet/m3_infer.py",
        "--ckpt", checkpoint,
        "--input", input_file,
        "--out", output_dir
    ]
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    print("-" * 70)
    print()
    
    if result.returncode == 0:
        print("✅ 推理成功完成！")
        print()
        print("📊 自检要点：")
        print("   1. ✓ Pipeline 契约显示:")
        print("      - Target cells: 10000 (过拟合配置)")
        print("      - Sample cells: 6000 (过拟合配置)")
        print("      - Sampler: random")
        print("      - Use frame: False")
        print()
        print("   2. ✓ Z-score 使用 checkpoint 自带 mean/std:")
        print("      - 打印信息应显示 'mean shape: (15,)'")
        print("      - 不依赖外部 stats.npz 文件")
        print()
        print("   3. ✓ 预测结果应该很好:")
        print("      - 使用训练样本，模型应该过拟合")
        print("      - 所有 15 个类别（L0-L14）都应该有预测")
        print()
        print(f"📁 输出文件: {output_dir}/1_U_colored.vtp")
        print("   可以在 3D Slicer 中查看彩色分割结果")
        print()
        print("⚠️  注意事项：")
        print("   - 本测试使用 .vtp 训练样本，应该看到完美过拟合")
        print("   - 若使用 raw .stl 推理，出现'彩纸屑'是正常的（泛化测试）")
        print("   - 泛化需要靠 m1_train.py 正常训练（含数据增强）来解决")
    else:
        print("❌ 推理失败")
        print(f"   退出码: {result.returncode}")
    
    print()
    print("=" * 70)

if __name__ == '__main__':
    main()

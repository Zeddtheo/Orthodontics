#!/usr/bin/env python3
"""
PointnetReg增强版系统综合测试
测试数据集、训练、评估的全新功能
"""

import sys
import subprocess
from pathlib import Path
import json
import time

# 测试配置
TEST_ROOT = "c:\\MISC\\Deepcare\\Orthodontics\\datasets\\landmarks_dataset\\cooked\\p0\\samples_consistent"
TEST_OUTPUT = "runs_comprehensive_test"

def run_command(cmd, description):
    """运行命令并记录结果"""
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = time.time() - start_time
        print(f"✅ Success ({duration:.1f}s)")
        if result.stdout.strip():
            print("Output:", result.stdout.strip()[-500:])  # 显示最后500字符
        return True
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"❌ Failed ({duration:.1f}s)")
        print("Error:", e.stderr.strip()[-500:] if e.stderr else "No error details")
        return False

def test_dataset_functionality():
    """测试数据集功能"""
    print("\n" + "="*60)
    print("📁 测试数据集增强功能")
    print("="*60)
    
    success = run_command([
        "python", "test_p0_dataset_new.py"
    ], "测试新数据集功能")
    
    return success

def test_group_training():
    """测试组训练功能"""
    print("\n" + "="*60)
    print("🦷 测试组训练功能")
    print("="*60)
    
    # 测试central组训练（小规模）
    success = run_command([
        "python", "p1_train.py",
        "--mode", "per_group",
        "--group", "central",
        "--epochs", "3",
        "--batch_size", "4",
        "--augment",
        "--arch_align",
        "--mirror_prob", "0.3",
        "--out_dir", TEST_OUTPUT,
        "--root", TEST_ROOT
    ], "训练central组模型")
    
    return success

def test_per_tooth_training():
    """测试单牙位训练功能"""
    print("\n" + "="*60)  
    print("🦷 测试单牙位训练功能")
    print("="*60)
    
    # 测试t31训练（小规模）
    success = run_command([
        "python", "p1_train.py", 
        "--mode", "per_tooth",
        "--tooth", "t31",
        "--epochs", "3", 
        "--batch_size", "4",
        "--augment",
        "--out_dir", TEST_OUTPUT,
        "--root", TEST_ROOT
    ], "训练t31牙位模型")
    
    return success

def test_group_evaluation():
    """测试组评估功能"""
    print("\n" + "="*60)
    print("📊 测试组评估功能") 
    print("="*60)
    
    success = run_command([
        "python", "p2_eval.py",
        "--mode", "per_group", 
        "--group", "central",
        "--ckpt_root", TEST_OUTPUT,
        "--out_dir", f"{TEST_OUTPUT}_eval",
        "--root", TEST_ROOT
    ], "评估central组模型")
    
    return success

def test_per_tooth_evaluation():
    """测试单牙位评估功能"""
    print("\n" + "="*60)
    print("📊 测试单牙位评估功能")
    print("="*60)
    
    success = run_command([
        "python", "p2_eval.py",
        "--mode", "per_tooth",
        "--tooth", "t31", 
        "--ckpt_root", TEST_OUTPUT,
        "--out_dir", f"{TEST_OUTPUT}_eval",
        "--root", TEST_ROOT
    ], "评估t31牙位模型")
    
    return success

def analyze_results():
    """分析测试结果"""
    print("\n" + "="*60)
    print("📈 分析测试结果")
    print("="*60)
    
    # 检查训练输出
    train_outputs = list(Path(TEST_OUTPUT).glob("**/config.json"))
    print(f"Found {len(train_outputs)} training configurations:")
    
    for config_path in train_outputs:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"  - {config['mode']}/{config['target']}: {config['train_size']} samples, {config['epochs']} epochs")
    
    # 检查评估输出
    eval_outputs = list(Path(f"{TEST_OUTPUT}_eval").glob("**/summary.json"))
    print(f"\nFound {len(eval_outputs)} evaluation summaries:")
    
    for summary_path in eval_outputs:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        if summary['coord_mm']['mean']:
            print(f"  - {summary['mode']}/{summary['target']}: {summary['coord_mm']['mean']:.3f}mm mean error, P@2mm={summary['coord_mm']['pck@2mm']:.3f}")
        else:
            print(f"  - {summary['mode']}/{summary['target']}: No coordinate evaluation")

def main():
    """主测试流程"""
    print("🧪 PointnetReg增强版系统综合测试")
    print("=" * 80)
    
    # 切换工作目录
    original_cwd = Path.cwd()
    test_dir = Path("c:\\MISC\\Deepcare\\Orthodontics\\src\\PointnetReg")
    if test_dir.exists():
        import os
        os.chdir(test_dir)
        print(f"Working directory: {test_dir}")
    else:
        print(f"❌ Test directory not found: {test_dir}")
        return
    
    tests = [
        ("数据集增强功能", test_dataset_functionality),
        ("组训练功能", test_group_training),
        ("单牙位训练功能", test_per_tooth_training),
        ("组评估功能", test_group_evaluation),
        ("单牙位评估功能", test_per_tooth_evaluation),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} 测试出现异常: {e}")
            results[test_name] = False
    
    # 分析结果
    analyze_results()
    
    # 总结
    print("\n" + "="*80)
    print("📋 测试总结")
    print("="*80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有增强功能测试通过！系统完全就绪。")
    else: 
        print("⚠️  部分测试失败，请检查具体错误信息。")
    
    # 恢复工作目录
    import os
    os.chdir(original_cwd)

if __name__ == "__main__":
    main()
"""
检查缓存文件的完整性
"""
import pyvista as pv
from pathlib import Path
from tqdm import tqdm

def check_cache_integrity(cache_dir: str):
    """检查所有缓存文件是否完整"""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"❌ 缓存目录不存在: {cache_dir}")
        return
    
    vtp_files = list(cache_path.glob("*.vtp"))
    print(f"📁 缓存目录: {cache_dir}")
    print(f"📊 文件总数: {len(vtp_files)}")
    print()
    
    corrupted_files = []
    missing_label_files = []
    
    for vtp_file in tqdm(vtp_files, desc="检查缓存文件"):
        try:
            mesh = pv.read(str(vtp_file))
            
            # 检查是否有 Label 字段
            if 'Label' not in mesh.cell_data:
                missing_label_files.append(vtp_file)
                
        except Exception as e:
            corrupted_files.append((vtp_file, str(e)))
    
    print()
    print("=" * 60)
    
    if corrupted_files:
        print(f"❌ 发现 {len(corrupted_files)} 个损坏的文件:")
        for file, error in corrupted_files:
            print(f"  - {file.name}")
            print(f"    错误: {error[:100]}...")
    else:
        print("✅ 没有发现损坏的文件")
    
    print()
    
    if missing_label_files:
        print(f"⚠️  发现 {len(missing_label_files)} 个缺少 Label 字段的文件:")
        for file in missing_label_files:
            print(f"  - {file.name}")
    else:
        print("✅ 所有文件都有 Label 字段")
    
    print()
    
    # 建议修复
    if corrupted_files or missing_label_files:
        print("🔧 修复建议:")
        print("  删除这些损坏的文件，让系统重新生成:")
        print()
        for file, _ in corrupted_files:
            print(f"  del {file}")
        for file in missing_label_files:
            print(f"  del {file}")

if __name__ == '__main__':
    check_cache_integrity('outputs/segmentation/module0/cache_decimated')

import shutil
from pathlib import Path
source_root = Path('src')
vendor_root = Path('API') / 'vendor'
mesh_src = source_root / 'MeshSegNet'
mesh_dst = vendor_root / 'MeshSegNet'
if mesh_dst.exists():
    shutil.rmtree(mesh_dst)
shutil.copytree(mesh_src, mesh_dst)
point_src = source_root / 'PointnetReg'
point_dst = vendor_root / 'PointnetReg'
if point_dst.exists():
    shutil.rmtree(point_dst)
shutil.copytree(point_src, point_dst)
calc_src = source_root / 'calc_metrics.py'
calc_dst = vendor_root / 'calc_metrics.py'
calc_dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(calc_src, calc_dst)
dataset_src = Path('datasets') / 'landmarks_dataset'
dataset_dst = vendor_root / 'datasets' / 'landmarks_dataset'
if dataset_dst.exists():
    shutil.rmtree(dataset_dst)
shutil.copytree(dataset_src, dataset_dst)
print('Vendor copy complete (case-sensitive)')

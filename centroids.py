from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

pairs = {
    "pred_upper": Path('outputs/api_pipeline/case3/meshsegnet/3_U_api.vtp'),
    "gt_upper": Path('datasets/landmarks_dataset/raw/3/3_U.vtp'),
    "pred_lower": Path('outputs/api_pipeline/case3/meshsegnet/3_L_api.vtp'),
    "gt_lower": Path('datasets/landmarks_dataset/raw/3/3_L.vtp'),
}

def load_mesh(path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    return reader.GetOutput()

for name, path in pairs.items():
    pd = load_mesh(path)
    points = vtk_to_numpy(pd.GetPoints().GetData())
    labels = vtk_to_numpy(pd.GetCellData().GetArray('Label'))
    cells = pd.GetPolys()
    arr = cells.GetData()
    cell_size = 4  # triangles? first number is number of points per cell
    # Each cell stored as (n, v1, v2, v3)
    arr_np = vtk_to_numpy(arr)
    n_cells = pd.GetNumberOfPolys()
    ptr = 0
    centroids = { }
    for cid in range(n_cells):
        n = arr_np[ptr]
        verts = arr_np[ptr+1:ptr+1+n]
        pts = points[verts]
        centroid = pts.mean(axis=0)
        lab = int(labels[cid])
        centroids.setdefault(lab, []).append(centroid)
        ptr += 1 + n
    summary = {}
    for lab, lst in centroids.items():
        arr = np.vstack(lst)
        summary[lab] = arr.mean(axis=0)
    print(f"=== {name} ===")
    for lab in sorted(summary):
        cx, cy, cz = summary[lab]
        print(f"label {lab}: centroid=({cx:.2f}, {cy:.2f}, {cz:.2f})")
    print()

# %%
"""
Dataset for training teeth segmentation model and multi-task (segmention + landmark detection) model
"""
import numpy as np
import pandas as pd
import torch
import torch_geometric
import trimesh
import vedo
import vtk
from torch.utils.data import Dataset
from torch_geometric.utils import k_hop_subgraph


def VTKTransform(
    mesh,
    rotate_X=[-180, 180],
    rotate_Y=[-180, 180],
    rotate_Z=[-180, 180],
):
    """
    apply transformation matrix (4*4)
    """
    Trans = vtk.vtkTransform()

    ry_flag = np.random.randint(0, 2)  # if 0, no rotate
    rx_flag = np.random.randint(0, 2)  # if 0, no rotate
    rz_flag = np.random.randint(0, 2)  # if 0, no rotate
    if rx_flag == 1:
        # rotate along Xth axis
        Trans.RotateX(np.random.uniform(rotate_X[0], rotate_X[1]))
    if ry_flag == 1:
        # rotate along Yth axis
        Trans.RotateY(np.random.uniform(rotate_Y[0], rotate_Y[1]))
    if rz_flag == 1:
        # rotate along Zth axis
        Trans.RotateZ(np.random.uniform(rotate_Z[0], rotate_Z[1]))

    matrix = Trans.GetMatrix()
    mesh.apply_transform(matrix)

    return mesh


def label_cell_region(faces, cell_label):
    """
    Given the faces and cell labels of a mesh, this function computes the cell region labels
    for each cell in the mesh. The cell region labels are defined as follows:
    - 0: background
    - 1: cluster teeth cells
    - 2: boundary gingiva cells
    - 3: boundary teeth cells
    """
    edge_index = torch.LongTensor(trimesh.graph.face_adjacency(faces=faces).T)
    edge_row_label = cell_label[edge_index[0]]
    edge_col_label = cell_label[edge_index[1]]
    cluster_teeth_edge_mask = (edge_col_label == edge_row_label) & (edge_col_label != 0)
    cluster_teeth_edges = edge_index[:, cluster_teeth_edge_mask.flatten()]
    boundary_gingiva_edge_mask = (edge_col_label != edge_row_label) & (
        edge_col_label * edge_row_label == 0
    )
    boundary_gingiva_edges = edge_index[
        :,
        boundary_gingiva_edge_mask.flatten(),
    ]
    boundary_teeth_edge_mask = (edge_col_label != edge_row_label) & (
        edge_col_label * edge_row_label != 0
    )
    boundary_teeth_edges = edge_index[
        :,
        boundary_teeth_edge_mask.flatten(),
    ]
    cluster_teeth_cell_idx = torch.unique(cluster_teeth_edges)
    boundary_gingiva_cell_idx = torch.unique(boundary_gingiva_edges)
    boundary_teeth_cell_idx = torch.unique(boundary_teeth_edges)

    enhanced_boundary_gingiva_cell_idx, _, _, _ = k_hop_subgraph(
        node_idx=boundary_gingiva_cell_idx,
        num_hops=2,
        edge_index=edge_index,
    )
    enhanced_boundary_teeth_cell_idx, _, _, _ = k_hop_subgraph(
        node_idx=boundary_teeth_cell_idx,
        num_hops=2,
        edge_index=edge_index,
    )
    cell_region_label = torch.zeros_like(cell_label)
    cell_region_label[cluster_teeth_cell_idx] = 1
    cell_region_label[torch.unique(enhanced_boundary_gingiva_cell_idx)] = 2
    cell_region_label[torch.unique(enhanced_boundary_teeth_cell_idx)] = 3
    return cell_region_label


def missing_teeth_augmentation(faces, labels, missing_num=1):
    # TODO: to refactor other parts
    unique_labels = torch.unique(labels)
    random_indices = torch.randperm(unique_labels.size(0))
    exclude_labels = random_indices[:missing_num]
    for idx in exclude_labels:
        mask = labels != idx
        faces = faces[mask]
        labels = labels[mask]
    return faces, labels


def preprocess_cell_features(mesh, coordinate_scale):
    mesh.compute_normals()
    normals = torch.FloatTensor(mesh.celldata["Normals"])  # normals of cells
    points = torch.FloatTensor(mesh.points())  # 3d coordinates of points
    # move to center and rescale, need reversed operation in postprocess
    points = (points - points.mean(0)) / coordinate_scale
    faces = torch.LongTensor(mesh.faces())  # 3-point index of each face
    cells = points[faces]  # 3x3 dimensional coordinates of faces
    centers = cells.mean(1)  # 3d coordinates of cell centers
    x = torch.cat((centers, normals, cells.view(-1, 9)), dim=1)
    return x, points


def sample_cells(cell_region_label, sample_num):
    # use all boundary cells and sample from cluster cells
    boundary_cell_idx = torch.where(cell_region_label >= 2)[0]
    cluster_cell_idx = torch.where(cell_region_label < 2)[0]
    sampled_cluster_cell_idx = torch.multinomial(
        torch.ones(cluster_cell_idx.shape[0]),
        sample_num - len(boundary_cell_idx),
        replacement=False,
    )
    sampled_cluster_cell_idx = cluster_cell_idx[sampled_cluster_cell_idx]
    selected_idx_list = torch.cat([boundary_cell_idx, sampled_cluster_cell_idx])
    return selected_idx_list


def attach_mesh_data(graph, points, mesh):
    landmark_categories = [
        name.split("_")[0] for name in mesh.celldata.keys() if "heatmap" in name
    ]
    for k in ["abrasion", "twisted", "tilted", "ectopic"]:
        if k in mesh.metadata.keys():
            graph[k] = torch.FloatTensor(mesh.metadata[k].astype("float"))
    for landmark in landmark_categories:
        heatmap_name = f"{landmark}_heatmap"
        graph[heatmap_name] = torch.FloatTensor(
            mesh.celldata[heatmap_name].astype("float")
        )
        landmark_pos = torch.ones([17, 3]) * 1e8
        point_idx = mesh.metadata[f"{landmark}_point_idx"].astype("int")
        valid_mask = point_idx != -1
        landmark_pos[valid_mask] = points[point_idx[valid_mask]]
        graph[f"{landmark}_pos"] = landmark_pos
    return graph


def graph_preprocess(
    mesh,
    transform,
    graph_k,
    test,
    sample_num,
    return_mesh,
    coordinate_scale,
    multitask=False,
):
    sample_flag = sample_num is not None and sample_num < mesh.ncells
    # TODO: need more data augmentation including more rotation, position shifting, field of view
    if transform and torch.randint(0, 2, (1,)).item():
        mesh = VTKTransform(mesh)
    x, points = preprocess_cell_features(mesh, coordinate_scale)
    x_pos = x[:, :3]  # preprocessed cell center positions
    if not test:  # annotated data
        faces = torch.LongTensor(mesh.faces())  # 3-point index of each face
        cell_label = torch.LongTensor(
            mesh.celldata["Label"].astype("int32").reshape(-1, 1)
        )
        teeth_centroids = torch.ones([17, 3]) * 1e8
        for idx in torch.unique(cell_label):
            if idx == 0:  # ignore gingiva
                continue
            centroid = x_pos[torch.where(cell_label == idx)[0]].mean(0)
            teeth_centroids[idx] = centroid
    if sample_flag:
        cell_region_label = label_cell_region(faces, cell_label)
        selected_idx_list = sample_cells(cell_region_label, sample_num)
        x, x_pos, cell_label, cell_region_label = (
            x[selected_idx_list],
            x_pos[selected_idx_list],
            cell_label[selected_idx_list],
            cell_region_label[selected_idx_list],
        )
    edge_index = torch_geometric.nn.knn_graph(x_pos, k=graph_k, loop=True)
    edge_index = torch_geometric.utils.to_undirected(edge_index)
    graph = torch_geometric.data.Data(
        x=x,
        edge_index=edge_index,
    )
    if not test:  # annotated data
        graph.cell_label = cell_label
        graph.teeth_centroids = teeth_centroids
        if multitask:
            graph = attach_mesh_data(graph, points, mesh)
    if sample_flag:
        graph.cell_region_label = cell_region_label
        graph.selected_idx_list = selected_idx_list
    return (graph, mesh) if return_mesh else graph


class CellGraph_Dataset(Dataset):
    def __init__(
        self,
        data_list,
        sample_num,
        test=False,
        return_mesh=False,
        graph_k=12,
        transform=True,
        coordinate_scale=10,
        multitask=False,
    ):
        self.data_list = data_list
        self.sample_num = sample_num
        self.test = test
        self.return_mesh = return_mesh
        self.graph_k = graph_k
        self.transform = transform
        self.coordinate_scale = coordinate_scale
        self.multitask = multitask

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        mesh_path = self.data_list[idx]  # vtk file name
        mesh = vedo.load(mesh_path)
        output = graph_preprocess(
            mesh,
            self.transform,
            self.graph_k,
            self.test,
            self.sample_num,
            self.return_mesh,
            self.coordinate_scale,
            self.multitask,
        )
        return output


# %%
if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    from tqdm import tqdm
    from glob import glob

    train_list = np.load(
        "../data/intraoral_scanners/adult_man_vtp_5w_list_multitask.npy"
    )

    dataset = CellGraph_Dataset(
        data_list=train_list,
        sample_num=None,
        test=False,
        return_mesh=False,
        graph_k=8,
        transform=True,
        coordinate_scale=10,
        multitask=True,
    )
    loader = DataLoader(
        dataset, batch_size=3, shuffle=False, num_workers=8, drop_last=False
    )
    batch = next(iter(loader))
    print(batch)
# %%

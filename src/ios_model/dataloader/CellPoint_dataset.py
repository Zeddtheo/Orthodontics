# %%
"""
Dataset for training landmark detection model
"""
import numpy as np
import torch
import torch_geometric
import vedo
from torch.utils.data import Dataset
from dataloader.CellGraph_dataset import VTKTransform, preprocess_cell_features

# %%
EPS = 1e-16


def attach_mesh_data(graph, mesh):
    points = mesh.points()
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
        point_idx = mesh.metadata[f"{landmark}_point_idx"].astype("int")
        location = np.ones([len(point_idx), 3]) * 1e8
        location[point_idx != -1] = points[point_idx[point_idx != -1]]
        graph[f"{landmark}_location"] = torch.FloatTensor(location)
    return graph


def graph_preprocess(mesh, transform, graph_k, return_mesh, normalize_coordinate):
    if transform and torch.randint(0, 2, (1,)).item():
        mesh = VTKTransform(mesh)
    points = torch.FloatTensor(mesh.points())  # 3d coordinates of points
    faces = torch.LongTensor(mesh.faces())  # 3-point index of each face
    cells = points[faces]  # 3x3 dimensional coordinates of faces
    positions = cells.mean(1)  # 3d coordinates of cell centers
    x = preprocess_cell_features(points, cells, positions, normalize_coordinate)
    cell_label = torch.LongTensor(mesh.celldata["Label"].astype("int32").reshape(-1, 1))
    cell_label_norm = cell_label / 16  # normalize cell label of 16 teeth to [0, 1]
    # add cell labels as input features
    x = torch.cat((x, cell_label_norm), dim=1)
    edge_index = torch_geometric.nn.knn_graph(positions, k=graph_k, loop=True)
    edge_index = torch_geometric.utils.to_undirected(edge_index)
    graph = torch_geometric.data.Data(
        x=x,
        edge_index=edge_index,
        pos=positions,
        cell_label=cell_label,
    )
    graph = attach_mesh_data(graph, mesh)
    if return_mesh:
        return graph, mesh
    else:
        return graph


class CellPoint_Dataset(Dataset):
    def __init__(
        self,
        data_list,
        return_mesh=False,
        graph_k=12,
        transform=True,
        normalize_coordinate=False,
    ):
        self.data_list = data_list
        self.return_mesh = return_mesh
        self.graph_k = graph_k
        self.transform = transform
        self.normalize_coordinate = normalize_coordinate

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        mesh_path = self.data_list[idx]  # vtk file name
        mesh = vedo.load(mesh_path)
        output = graph_preprocess(
            mesh,
            self.transform,
            self.graph_k,
            self.return_mesh,
            self.normalize_coordinate,
        )
        return output


# %%
if __name__ == "__main__":
    import glob

    import numpy as np
    from torch_geometric.loader import DataLoader
    from tqdm import tqdm

    train_list = np.array(
        glob.glob(
            "../data/intraoral_scanners/max/landmark_heatmap_std0.9_threshold2.7/*.vtp"
        )
    )
    # %%

    dataset = CellPoint_Dataset(
        data_list=train_list,
        return_mesh=False,
        graph_k=8,
        transform=True,
        normalize_coordinate=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        num_workers=8,
        drop_last=False,
    )
    for idx, batch in tqdm(enumerate(iter(loader))):
        # create unique teeth-level batch idx
        batch.cell_label_batch = batch.batch * 17 + batch.cell_label.flatten()
        print(batch)
        break
    print(np.unique(batch.cell_label_batch))
    print(np.unique(batch["batch"], return_counts=True))
    print(batch["MCP_location"].reshape((-1, 3)))

# %%

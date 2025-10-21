# %%

import numpy as np
import pandas as pd
import vedo


def cell_to_point_label(mesh):
    """
    Assigns a label to each point of a mesh based on the most frequent label of its adjacent cells.

    Args:
        mesh (vedo.Mesh): The input mesh.

    Returns:
        vedo.Mesh: The output mesh with a "Label" point data array.

    """
    points = mesh.points()
    cells = np.array(mesh.faces())
    cell_labels = mesh.celldata["Label"]
    point_index = cells.flatten()
    point_labels = np.repeat(cell_labels, 3)
    df = pd.DataFrame({"point_index": point_index, "point_labels": point_labels})
    df_grouped = (
        df.groupby("point_index")["point_labels"]
        .apply(lambda x: x.value_counts().index[0])
        .reset_index(name="most_frequent_label")
    )
    point_labels = df_grouped["most_frequent_label"].values
    mesh.pointdata["Label"] = point_labels
    return mesh


def point_to_cell_label(mesh):
    """
    Assigns a label to each cell of a mesh based on the most frequent label of its points.

    Args:
        mesh (vedo.Mesh): The input mesh.

    Returns:
        vedo.Mesh: The output mesh with a "Label" cell data array.

    """
    points = mesh.points()
    cells = np.array(mesh.faces())
    point_labels = mesh.pointdata["Label"]
    cell_labels = point_labels[cells].astype("int64")
    cell_labels = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), axis=1, arr=cell_labels
    )
    mesh.celldata["Label_from_Points"] = cell_labels
    return mesh


# %%
if __name__ == "__main__":
    import glob
    from tqdm import tqdm

    mesh_path_list = glob.glob("../../ios_data/max/vtps/*.vtp")
    for mesh_path in tqdm(mesh_path_list):
        mesh = vedo.load(mesh_path)
        mesh = cell_to_point_label(mesh)
        mesh.write(mesh_path)
# %%

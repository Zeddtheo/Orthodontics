# %%
import glob
import os

import numpy as np
import pandas as pd
import vedo
from tqdm import tqdm


# %%
def remove_kid_teeth(data_path_list):
    adult_data_path_list = [
        path for path in data_path_list if max(vedo.load(path).celldata["Label"]) <= 16
    ]
    return adult_data_path_list


def check_teeth_label_match(csv_path, mesh_path_root):
    mesh_name = csv_path.split("/")[-1].split(".")[0]
    mesh_path = os.path.join(mesh_path_root, f"{mesh_name}.vtp")
    mesh = vedo.load(mesh_path)
    landmark_csv = pd.read_csv(csv_path)
    landmark_csv.insert(loc=0, column="filename", value=mesh_name)
    teeth_label = mesh.celldata["Label"]
    barycenters = mesh.cell_centers()
    error_row_idx = []
    for row_idx, row in landmark_csv.iterrows():
        category = row["category"]
        if category in ["TCP", "GPP"]:
            continue
        label = row["label"]
        # check duplicate
        if len(np.where(landmark_csv["label"] == label)[0]) > 1:
            error_row_idx.append(row_idx)
            continue
        loc = row[["x", "y", "z"]].values.astype("float")
        distance = np.sum((loc - barycenters) ** 2, axis=1)
        closest_cell_label = teeth_label[np.argsort(distance)[:10]]
        teeth_idx = row["teeth_idx"]
        try:
            teeth_idx = int(teeth_idx)
            if teeth_idx not in closest_cell_label:
                error_row_idx.append(row_idx)
        except:
            error_row_idx.append(row_idx)
    return landmark_csv[landmark_csv.index.isin(error_row_idx)]


# %%
mesh_path_root = "../data/intraoral_scanners/ios_gt_max/vtps/"
csv_path_root = "../data/intraoral_scanners/ios_gt_max/csvs/"
csv_path_list = glob.glob(os.path.join(csv_path_root, "*.csv"))
error_table = []
for csv_path in tqdm(csv_path_list):
    error_table.append(check_teeth_label_match(csv_path, mesh_path_root))
error_table = pd.concat(error_table)

# %%
error_table.to_csv(
    "../data/intraoral_scanners/ios_gt_max/landmark_error_max.csv",
    index=False,
)


# %%

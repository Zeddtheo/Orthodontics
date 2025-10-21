# %%
import glob
import pandas as pd
import numpy as np


# %%
def fcsv_preprocess(path):
    data = pd.read_csv(path, skiprows=2)
    data = data[["label", "x", "y", "z"]]
    data["category"] = data["label"].apply(lambda x: x[:3])
    data["teeth_idx"] = data["label"].apply(lambda x: x[3:])
    return data


def csv_to_fcsv(csv_path):
    pre_data = pd.read_csv(csv_path)
    pre_data["teeth_idx"] = pre_data["index"].apply(lambda x: int(x[3:]))
    # sort by teeth_idx and category to accelerate the mannual annotation revision
    pre_data.sort_values(by=["teeth_idx", "category"], inplace=True)
    df = pd.DataFrame(
        columns=[
            "# columns = id",
            "x",
            "y",
            "z",
            "ow",
            "ox",
            "oy",
            "oz",
            "vis",
            "sel",
            "lock",
            "label",
            "desc",
            "associatedNodeID",
        ]
    )
    df[["x", "y", "z", "label"]] = pre_data[["x", "y", "z", "index"]]
    df[["ow", "ox", "oy", "oz", "vis", "sel", "lock", "associatedNodeID"]] = [
        0.0,
        0.0,
        0.0,
        1.0,
        1,
        1,
        1,
        "vtkMRMLModelNode4",
    ]
    df["# columns = id"] = [
        f"vtkMRMLMarkupsFiducialNode_{idx}" for idx in range(len(df))
    ]
    fcsv_path = csv_path.replace(".csv", ".fcsv")
    df.to_csv(fcsv_path, index=False)
    with open(fcsv_path, "r") as f:
        lines = f.readlines()
        lines.insert(0, "# Markups fiducial file version = 4.10\n")
        lines.insert(1, "# CoordinateSystem = RAS\n")
    with open(fcsv_path, "w") as f:
        f.writelines(lines)
    return


max_teeth_landmark_dict = {
    7: ["DCP", "MCP", "PGP", "LGP", "CTF", "IEP"],
    8: ["DCP", "MCP", "PGP", "LGP", "CTF", "IEP"],
    6: ["DCP", "MCP", "PGP", "LGP", "CTF", "IEP"],
    9: ["DCP", "MCP", "PGP", "LGP", "CTF", "IEP"],
    5: ["DCP", "MCP", "PGP", "LGP", "CTF", "CCT"],
    10: ["DCP", "MCP", "PGP", "LGP", "CTF", "CCT"],
    4: ["DCP", "MCP", "PGP", "LGP", "CTF", "BTP", "BTT"],
    11: ["DCP", "MCP", "PGP", "LGP", "CTF", "BTP", "BTT"],
    3: ["DCP", "MCP", "PGP", "LGP", "CTF", "BTP", "BTT"],
    12: ["DCP", "MCP", "PGP", "LGP", "CTF", "BTP", "BTT"],
    2: ["DCP", "MCP", "PGP", "LGP", "CTF", "MBT", "DBT", "MTT", "DTT"],
    13: ["DCP", "MCP", "PGP", "LGP", "CTF", "MBT", "DBT", "MTT", "DTT"],
    1: ["DCP", "MCP", "PGP", "LGP", "CTF", "MBT", "DBT", "MTT", "DTT"],
    14: ["DCP", "MCP", "PGP", "LGP", "CTF", "MBT", "DBT", "MTT", "DTT"],
    15: ["DCP", "MCP", "PGP", "LGP", "CTF", "MBT", "DBT", "MTT", "DTT"],
    16: ["DCP", "MCP", "PGP", "LGP", "CTF", "MBT", "DBT", "MTT", "DTT"],
}

man_teeth_landmark_dict = {
    7: ["DCP", "MCP", "PGP", "LGP", "CTF", "IEP"],
    8: ["DCP", "MCP", "PGP", "LGP", "CTF", "IEP"],
    6: ["DCP", "MCP", "PGP", "LGP", "CTF", "IEP"],
    9: ["DCP", "MCP", "PGP", "LGP", "CTF", "IEP"],
    5: ["DCP", "MCP", "PGP", "LGP", "CTF", "CCT"],
    10: ["DCP", "MCP", "PGP", "LGP", "CTF", "CCT"],
    4: ["DCP", "MCP", "PGP", "LGP", "CTF", "BTP", "BTT"],
    11: ["DCP", "MCP", "PGP", "LGP", "CTF", "BTP", "BTT"],
    3: ["DCP", "MCP", "PGP", "LGP", "CTF", "BTP", "BTT"],
    12: ["DCP", "MCP", "PGP", "LGP", "CTF", "BTP", "BTT"],
    2: ["DCP", "MCP", "PGP", "LGP", "CTF", "MBT", "DBT", "MTT", "DTT", "DPP", "MBH"],
    13: ["DCP", "MCP", "PGP", "LGP", "CTF", "MBT", "DBT", "MTT", "DTT", "DPP", "MBH"],
    1: ["DCP", "MCP", "PGP", "LGP", "CTF", "MBT", "DBT", "MTT", "DTT", "DPP"],
    14: ["DCP", "MCP", "PGP", "LGP", "CTF", "MBT", "DBT", "MTT", "DTT", "DPP"],
    15: ["DCP", "MCP", "PGP", "LGP", "CTF", "MBT", "DBT", "MTT", "DTT", "DPP"],
    16: ["DCP", "MCP", "PGP", "LGP", "CTF", "MBT", "DBT", "MTT", "DTT", "DPP"],
}
landmark_category_dict = {
    "DCP": 1,
    "MCP": 2,
    "PGP": 3,
    "LGP": 4,
    "CTF": 5,
    "IEP": 6,
    "CCT": 7,
    "BTP": 8,
    "BTT": 9,
    "MBT": 10,
    "DBT": 11,
    "MTT": 12,
    "DTT": 13,
    "DPP": 14,
    "MBH": 15,
}


def get_position_idx(teeth_type="max"):
    idx_array = []  # (teeth_idx, category_idx)
    teeth_landmark_dict = (
        max_teeth_landmark_dict if teeth_type == "max" else man_teeth_landmark_dict
    )
    for teeth_idx in range(1, 17):
        for landmark in teeth_landmark_dict[teeth_idx]:
            category_idx = landmark_category_dict[landmark]
            idx_array.append([teeth_idx, category_idx])
    return np.array(idx_array)


def csv_to_array(csv_path, teeth_type="max"):
    table = pd.read_csv(csv_path)
    table.dropna(subset=["teeth_idx"], inplace=True)
    table["category_idx"] = table["category"].map(landmark_category_dict)
    coordinate_array = []
    teeth_landmark_dict = (
        max_teeth_landmark_dict if teeth_type == "max" else man_teeth_landmark_dict
    )
    for teeth_idx in range(1, 17):
        for landmark in teeth_landmark_dict[teeth_idx]:
            category_idx = landmark_category_dict[landmark]
            coord = table[
                (table["teeth_idx"] == teeth_idx)
                & (table["category_idx"] == category_idx)
            ][["x", "y", "z"]]
            if len(coord) == 0:
                coord_value = np.array([np.nan, np.nan, np.nan])
            else:
                coord_value = coord.values[0]
            coordinate_array.append(coord_value)
    return np.array(coordinate_array)


# %%
if __name__ == "__main__":
    path_list = glob.glob("../data/intraoral_scanners/ios_landmarks/fcsvs/*.fcsv")
    for path in path_list:
        data = fcsv_preprocess(path)
        save_path = path.replace(".fcsv", ".csv").replace("fcsvs", "csvs")
        data.to_csv(save_path, index=False)


# %%

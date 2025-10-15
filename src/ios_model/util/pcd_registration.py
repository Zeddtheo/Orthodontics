# %%
import trimesh
import vtk
import numpy as np


def match_centroid(
    path_refer,
    path_reg,
    center_matrix_path=None,
    inverse_transform=False,
    save_matrix=True,
    export_stl=True,
):
    mesh_refer = trimesh.load(path_refer)
    mesh_reg = trimesh.load(path_reg)
    if center_matrix_path is None:
        mesh_reg_vertices = mesh_reg.vertices
        mesh_refer_vertices = mesh_refer.vertices
        mesh_refer_vertices_centroid = np.mean(mesh_refer_vertices, axis=0)
        mesh_reg_vertices_centroid = np.mean(mesh_reg_vertices, axis=0)
        center_matrix = mesh_refer_vertices_centroid - mesh_reg_vertices_centroid
    else:
        center_matrix = np.load(center_matrix_path)
    if inverse_transform:
        center_matrix = -center_matrix
    if save_matrix:
        np.save(path_reg.replace(".stl", "_center_matrix.npy"), center_matrix)
    mesh_reg.vertices = mesh_reg.vertices + center_matrix
    path_reg_centered = path_reg.split(".stl")[0] + "_centered.stl"
    if export_stl:
        mesh_reg.export(file_obj=path_reg_centered)
    return path_reg_centered


def icptransform(
    path_refer,
    path_reg,
    transform_matrix_path=None,
    inverse_transform=False,
    save_matrix=True,
    export_stl=True,
):
    if transform_matrix_path is None:
        icptransform = vtk.vtkIterativeClosestPointTransform()
        reader_mesh_refer = vtk.vtkSTLReader()
        reader_mesh_refer.SetFileName(path_refer)
        reader_mesh_refer.Update()

        reader_mesh_reg = vtk.vtkSTLReader()

        reader_mesh_reg.SetFileName(path_reg)
        reader_mesh_reg.Update()

        icptransform.SetSource(reader_mesh_reg.GetOutput())
        icptransform.SetTarget(reader_mesh_refer.GetOutput())
        icptransform.GetLandmarkTransform().SetModeToRigidBody()
        # icptransform.StartByMatchingCentroidsOn()
        icptransform.SetMaximumNumberOfIterations(300)
        icptransform.SetMaximumNumberOfLandmarks(1000)
        icptransform.SetMaximumMeanDistance(0.001)
        icptransform.SetCheckMeanDistance(5)
        icptransform.Modified()
        icptransform.Update()

        # print(icptransform)

        transform_matrix = np.eye(4)
        m = icptransform.GetMatrix()
        for i in range(4):
            for j in range(4):
                transform_matrix[i, j] = m.GetElement(i, j)
    else:
        transform_matrix = np.load(transform_matrix_path)
    if inverse_transform:
        transform_matrix = np.array(
            trimesh.voxel.transforms.Transform(transform_matrix).inverse_matrix
        )
    if save_matrix:
        np.save(path_reg.replace(".stl", "_transform_matrix.npy"), transform_matrix)
    path_reg_transformed = path_reg.split(".stl")[0] + "_icp.stl"
    if export_stl:
        mesh_reg = trimesh.load(path_reg)
        mesh_reg.apply_transform(transform_matrix)
        mesh_reg.export(file_obj=path_reg_transformed)
    return path_reg_transformed


def global_register_stls(
    path_refer,
    path_reg,
):
    path_reg_centered = match_centroid(
        path_refer,
        path_reg,
        center_matrix_path=None,
        inverse_transform=False,
        save_matrix=True,
        export_stl=True,
    )

    path_reg_transformed = icptransform(
        path_refer,
        path_reg_centered,
        transform_matrix_path=None,
        inverse_transform=False,
        save_matrix=True,
        export_stl=True,
    )


def inverse_global_register_stls(
    path_refer, path_reg, transform_matrix_path, center_matrix_path
):
    path_reg_transformed = icptransform(
        path_refer,
        path_reg,
        transform_matrix_path=transform_matrix_path,
        inverse_transform=True,
        save_matrix=False,
        export_stl=True,
    )
    path_reg_centered = match_centroid(
        path_refer,
        path_reg_transformed,
        center_matrix_path=center_matrix_path,
        inverse_transform=True,
        save_matrix=False,
        export_stl=True,
    )


# %%
if __name__ == "__main__":
    path_cbct = "/Users/wangzichen/Desktop/intern/Deepcare/data/suntianyang/ct40_reg/stl_cbct/suntianyang_ctpre_teeth_u.stl"
    path_ios = "/Users/wangzichen/Desktop/intern/Deepcare/data/suntianyang/ct40_reg/stl_ios/40_u_0.stl"
    # global register two stls, first from ios to cbct, and then inverse map cbct to ios
    global_register_stls(path_cbct, path_ios)
    inverse_global_register_stls(
        path_ios,
        path_cbct,
        transform_matrix_path="/Users/wangzichen/Desktop/intern/Deepcare/data/suntianyang/ct40_reg/stl_ios/40_u_0_centered_transform_matrix.npy",
        center_matrix_path="/Users/wangzichen/Desktop/intern/Deepcare/data/suntianyang/ct40_reg/stl_ios/40_u_0_center_matrix.npy",
    )

    cbct2ios = {
        i: 8 - i if i < 8 else i for i in range(1, 15)
    }  # index error in cbct data
    # inverse map individual cbct teeth to ios with global transform matrix
    for k, v in cbct2ios.items():
        cbct = f"/Users/wangzichen/Desktop/intern/Deepcare/data/suntianyang/ct40_reg/stl_cbct/suntianyang_ctpre_teeth_{k}.stl"
        ios = f"/Users/wangzichen/Desktop/intern/Deepcare/data/suntianyang/ct40_reg/stl_ios/40_u_predicted_{v}.stl"
        inverse_global_register_stls(
            ios,
            cbct,
            transform_matrix_path="/Users/wangzichen/Desktop/intern/Deepcare/data/suntianyang/ct40_reg/stl_ios/40_u_0_centered_transform_matrix.npy",
            center_matrix_path="/Users/wangzichen/Desktop/intern/Deepcare/data/suntianyang/ct40_reg/stl_ios/40_u_0_center_matrix.npy",
        )
    # teeth-level icp transform from ios to cbct
    for k, v in cbct2ios.items():
        cbct = f"/Users/wangzichen/Desktop/intern/Deepcare/data/suntianyang/ct40_reg/stl_cbct/suntianyang_ctpre_teeth_{k}_icp_centered.stl"
        ios = f"/Users/wangzichen/Desktop/intern/Deepcare/data/suntianyang/ct40_reg/stl_ios/40_u_predicted_{v}.stl"
        path_reg_transformed = icptransform(
            cbct,
            ios,
            transform_matrix_path=None,
            inverse_transform=False,
            save_matrix=True,
            export_stl=True,
        )
    # teeth-level inverse icp transform from cbct to ios
    for k, v in cbct2ios.items():
        cbct = f"/Users/wangzichen/Desktop/intern/Deepcare/data/suntianyang/ct40_reg/stl_cbct/suntianyang_ctpre_teeth_{k}_icp_centered.stl"
        ios = f"/Users/wangzichen/Desktop/intern/Deepcare/data/suntianyang/ct40_reg/stl_ios/40_u_predicted_{v}.stl"
        path_reg_transformed = icptransform(
            ios,
            cbct,
            transform_matrix_path=f"/Users/wangzichen/Desktop/intern/Deepcare/data/suntianyang/ct40_reg/stl_ios/40_u_predicted_{v}_transform_matrix.npy",
            inverse_transform=True,
            save_matrix=False,
            export_stl=True,
        )
# %%
# TODO: replace ICP with the following vedo vtp format
# import vedo
#
# mesh_refer = vedo.load(
#    "../data/intraoral_scanners/man/landmark_heatmap_std1_threshold4/20200923_29905_man.vtp"
# )
# mesh_reg = vedo.load("../data/intraoral_scanners/lower_0707_5w.vtp")
# icp = vtk.vtkIterativeClosestPointTransform()
# icp.SetSource(mesh_reg.polydata())
# icp.SetTarget(mesh_refer.polydata())
# icp.GetLandmarkTransform().SetModeToRigidBody()
## icp.StartByMatchingCentroidsOn()
# icp.SetMaximumNumberOfIterations(300)
# icp.SetMaximumNumberOfLandmarks(1000)
# icp.SetMaximumMeanDistance(0.001)
# icp.SetCheckMeanDistance(5)
# icp.Modified()
# icp.Update()
# aligned_mesh = mesh_reg.clone().apply_transform(icp)

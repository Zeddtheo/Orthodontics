import vtk


def nii2mesh(filename_nii, filename_stl, label_list=[1, 2, 3]):
    """
    Read a nifti file including a binary map of a segmented organ with label id = label.
    Convert it to a smoothed mesh of type stl.
    filename_nii : Input nifti binary map
    filename_stl : Output mesh name in stl/ply format
    label_list : segmented label list
    """
    # read the file
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(filename_nii)
    reader.Update()

    # apply marching cube surface generation
    surf = vtk.vtkDiscreteMarchingCubes()
    surf.SetInputConnection(reader.GetOutputPort())
    if len(label_list) == 1:
        surf.SetValue(0, label_list[0])
    elif len(label_list) > 1:
        surf.GenerateValues(len(label_list), min(label_list), max(label_list))
    else:
        raise ("label list should not be empty!")
    surf.Update()

    # smoothing the mesh
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    if vtk.VTK_MAJOR_VERSION <= 5:
        smoother.SetInput(surf.GetOutput())
    else:
        smoother.SetInputConnection(surf.GetOutputPort())

    smoother.SetNumberOfIterations(30)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()  # The positions can be translated and scaled such that they fit within a range of [-1, 1] prior to the smoothing computation
    smoother.GenerateErrorScalarsOn()
    smoother.Update()

    # save the output
    writer = vtk.vtkSTLWriter()
    # writer = vtk.vtkPLYWriter()
    writer.SetInputConnection(smoother.GetOutputPort())
    writer.SetFileTypeToASCII()
    writer.SetFileName(filename_stl)
    writer.Write()


if __name__ == "__main__":
    nii_path = "/Users/wangzichen/Desktop/intern/Deepcare/data/suntianyang/ct40_reg/suntianyang_ct25_teeth.nii.gz"
    stl_path = nii_path.replace(".nii.gz", "_u.stl")
    label_list = list(range(1, 15))
    nii2mesh(nii_path, stl_path, label_list=label_list)

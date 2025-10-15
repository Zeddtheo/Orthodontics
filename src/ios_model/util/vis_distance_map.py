import vtk
from matplotlib import cm
import matplotlib.colors as mcolors
import numpy as np


input1 = vtk.vtkPolyData()
reader1 = vtk.vtkSTLReader()
reader1.SetFileName(
    "/Users/wangzichen/Desktop/intern/Deepcare/data/suntianyang/ct40_reg/stl_ios/40_u_0.stl"
)
reader1.Update()
input1 = reader1.GetOutput()  # 读取模型A (main model)

input2 = vtk.vtkPolyData()
reader2 = vtk.vtkSTLReader()
reader2.SetFileName(
    "/Users/wangzichen/Desktop/intern/Deepcare/data/suntianyang/ct40_reg/stl_cbct/suntianyang_ctpre_teeth_0_icp_centered_icp.stl"
)
reader2.Update()
input2 = reader2.GetOutput()  # 读取模型B

# 数据合并，可以合并显示两个模型
clean1 = vtk.vtkCleanPolyData()
clean1.SetInputData(input1)

clean2 = vtk.vtkCleanPolyData()
clean2.SetInputData(input2)

distanceFilter = vtk.vtkDistancePolyDataFilter()

distanceFilter.SetInputConnection(0, clean1.GetOutputPort())
distanceFilter.SetInputConnection(1, clean2.GetOutputPort())
distanceFilter.SignedDistanceOn()
distanceFilter.Update()  # 计算距离
distanceFilter.GetOutputPort()
mapper = vtk.vtkPolyDataMapper()  # 配置mapper
mapper.SetInputConnection(distanceFilter.GetOutputPort())

# mapper.SetScalarRange(  # 设置颜色映射范围
#     distanceFilter.GetOutput().GetPointData().GetScalars().GetRange()[0],
#     distanceFilter.GetOutput().GetPointData().GetScalars().GetRange()[1])
mapper.SetScalarRange(-2, 2)

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor1 = vtk.vtkActor()
actor1.SetMapper(mapper)
lut = vtk.vtkLookupTable()

# lut.SetHueRange(0.7, 0)  # 映射的颜色变换参数, 0-red, 0.9-purple
# lut.SetAlphaRange(1.0, 1.0)
# lut.SetValueRange(1.0, 1.0)
# lut.SetSaturationRange(1.0, 1.0)
# lut.SetNumberOfTableValues(256)


mycmap = cm.get_cmap("coolwarm")
color_list = [mcolors.rgb2hex(mycmap(i)) for i in range(mycmap.N)]
coolwarm_array = np.zeros((256, 3))
for j, k in enumerate(color_list):
    h = k.lstrip("#")
    v = tuple(int(h[i : i + 2], 16) / 255 for i in (0, 2, 4))
    coolwarm_array[j] = v

lutNum = 256
lut.SetNumberOfTableValues(lutNum)
ctf = vtk.vtkColorTransferFunction()
ctf.SetColorSpaceToDiverging()
ctf.AddRGBPoint(0.0, 0, 0, 1.0)  # blue
ctf.AddRGBPoint(1.0, 1.0, 0, 0)  # red
for ii, ss in enumerate([float(xx) / float(lutNum) for xx in range(lutNum)]):
    cc = ctf.GetColor(ss)
    # print(cc)
    cc = coolwarm_array[ii]
    lut.SetTableValue(ii, cc[0], cc[1], cc[2], 1.0)

mapper.SetLookupTable(lut)
mapper2 = vtk.vtkPolyDataMapper()
mapper2.SetInputData((distanceFilter.GetSecondDistanceOutput()))

# mapper2.SetScalarRange(  # 设置颜色映射范围
#     distanceFilter.GetSecondDistanceOutput().GetPointData().GetScalars().GetRange()[0],
#     distanceFilter.GetSecondDistanceOutput().GetPointData().GetScalars().GetRange()[1])
mapper2.SetScalarRange(-2, 2)


actor2 = vtk.vtkActor()
actor2.SetMapper(mapper2)

scalarBar = vtk.vtkScalarBarActor()
scalarBar.SetLookupTable(mapper.GetLookupTable())
scalarBar.SetTitle("(mm)")
scalarBar.SetNumberOfLabels(15)
scalarBar.SetMaximumNumberOfColors(256)

# # 设置标题颜色
scalarBar.DrawTickLabelsOn()
scalarBar.GetTitleTextProperty().SetColor(0, 0, 0)
scalarBar.GetLabelTextProperty().SetColor(0, 0, 0)
scalarBar.GetTitleTextProperty().SetFontSize(1)
scalarBar.GetLabelTextProperty().SetFontSize(1)

arender = vtk.vtkRenderer()
arender.SetViewport(0, 0.0, 1, 1.0)
renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(arender)
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
style = vtk.vtkInteractorStyleTrackballActor()
iren.SetInteractorStyle(style)
aCamera = vtk.vtkCamera()
aCamera.SetViewUp(0, 0, -1)
aCamera.SetPosition(0, -1, 0)
aCamera.ComputeViewPlaneNormal()
aCamera.Azimuth(30.0)
aCamera.Elevation(30.0)
aCamera.Dolly(1.5)

arender.AddActor(actor)
# arender.AddActor(actor1)
arender.SetActiveCamera(aCamera)
arender.ResetCamera()
arender.SetBackground(1, 1, 1)
arender.ResetCameraClippingRange()
arender.AddActor2D(scalarBar)

renWin.Render()
iren.Initialize()
iren.Start()

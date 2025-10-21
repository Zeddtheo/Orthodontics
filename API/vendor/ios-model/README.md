**Graph-based Deep Learning Framework for Automated Labeling of Raw Dental Surface from 3D Intraoral Scanners**
==============================================================================================================================

Created by Zichen Wang(zcwang@deepcare.com)


Prequisites
------------
python 3.8.16  
pytorch 1.8.0  
cudatoolkit 11.1
numpy 1.23.5  
pandas 1.5.2  
vedo 2023.4.3  
scikit-learn 1.2.0  
pygc 0.01   
torch-geometric 2.0.1   
torch-cluster 1.5.9  
torch-scatter 2.0.8  
torch-sparse 0.6.12  
torch-spline-conv 1.2.1  
DracoPy 1.2.0  
open3d 0.16.0



Modle API Instruction
-------------------------------------

1. change directory to model folder
```
cd ./teeth_seg_torchscript
```

2. define parameters in `teeth_seg_torchscript.py`

```py
### Define Parameters Below ###
jaw_type = "man"
path_drc = "./LowerJawScan.drc"
device = torch.device("cuda:3")
```
3. run `teeth_seg_torchscript.py`
```
python teeth_seg_torchscript.py
```
```
Returns:
        labels_downsample (np.ndarray): predicted labels of downsampled mesh
        labels_origin (np.ndarray): predicted labels of origin mesh
        points (np.ndarray): downsampled mesh points
        faces (np.ndarray): downsampled mesh faces
```
Please note that labels_downsample, points , faces will be used as the inputs of landmark detection module (working on downsampled mesh data with 50000 cells).
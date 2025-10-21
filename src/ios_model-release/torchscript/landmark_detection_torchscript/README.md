**Graph-based Deep Learning Framework for Landmark Detection of Raw Dental Surface from 3D Intraoral Scanners**
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
cd ./landmark_detection_torchscript
```

2. define parameters in `landmark_detection_torchscript.py`

```py
### Define Parameters Below ###
jaw_type = "man"
path_drc = "./LowerJawScan_downsample.drc" # should be downsampled to 50000 cells
path_label = "./seg_labels_downsample.npy"
```
3. run `landmark_detection_torchscript.py`
```
python landmark_detection_torchscript.py
```
```
Returns:
        outputs (Dict): dictionary of predicted landmark positions for each teeth
        {
                1: {
                        "MCP": [x0, y0, z0],
                        "DCP": [x1, y1, z1],
                        ... (all landmarks)
                }
                2: {
                        ...
                }
                ... (all detected teeth) 
        }
```
Please note that the landmark detection module works on downsampled mesh data with 50000 cells.
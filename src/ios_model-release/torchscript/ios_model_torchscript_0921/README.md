**Graph-based Deep Learning Framework for 3D Intraoral Scanner Analysis**
==============================================================================================================================

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
1. define parameters in `ios_model_torchscript.py`

```py
### Define Parameters Below ###
jaw_type = "man"
path_drc = "./LowerJawScan.drc"
device = torch.device("cuda:3")
```
3. run `ios_model_torchscript.py`
```
python ios_model_torchscript.py
```
```
Returns:
        preds_origin (np.ndarray): predicted labels of origin mesh
        landmark_outputs (Dict): dictionary of predicted landmark positions for each teeth
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
Please note that predicted labels (preds) of downsampled mesh (with 50000 cells) will not be returned. And the landmark detection module works on downsampled mesh.
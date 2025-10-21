## Instruction on Landmark Detection Module Usage

### Step 01: Generate Landmark Heatmap Data
- run `util/landmark_heatmap.py`
- Parameters:
    - `jaw_type`: max (上颌) man (下颌)
    - `load_teeth_label`: set False for only generating landmark heatmap data
    - `std`: std of guassian heatmap
    - `threshold`: threshold to truncate the heatmap value to zero

### Step 02: Train Landmark Detection Model
- run `train_point.py` for model training
- Parameters:
    - `data_path_list`: root folder path for data generated in step 01

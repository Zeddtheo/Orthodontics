## Teeth Segmentation TorchScript

This folder hosts the minimal ios-model TorchScript assets required to turn an
STL surface mesh into a labeled VTP. The shipped weights are:

- `man_teeth_seg_model_script.pt`: lower jaw (mandibular) model.
- `max_teeth_seg_model_script.pt`: upper jaw (maxillary) model.

### Dependencies

The original release was built with Python 3.8 + PyTorch 1.8, torch-geometric
2.0.1, vedo 2023.4.3, scikit-learn 1.2.0, DracoPy 1.2.0, and open3d 0.16.0. Any
modern environment with equivalent packages works; see `.venv_teeth_seg` for a
reference setup.

### CLI Usage

Invoke the helper script directly:

```powershell
python run_teeth_seg_from_stl.py `
  --input C:\path\to\jaw.stl `
  --jaw-type man `
  --device cpu `
  --output C:\path\to\jaw_seg.vtp
```

`--jaw-type` accepts `man` or `max`. When `--device` points to CUDA, the script
verifies GPU availability; otherwise it runs on CPU.

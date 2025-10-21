## ios-model TorchScript Inference

The `IOS-Model` directory has been trimmed to the components required for
TorchScript-based tooth segmentation that converts STL meshes to labeled VTP
files.

### Layout
- `teeth_seg/` — inference helpers, TorchScript weights, and the CLI entrypoint
  `run_teeth_seg_from_stl.py`.
- `.venv_teeth_seg/` — optional virtual environment prepared for the original
  release (kept for compatibility).

### Quick Start
Run the segmentation CLI from the project root (CPU shown; use `cuda:0` etc. for
GPU):

```powershell
python src/IOS-Model/teeth_seg/run_teeth_seg_from_stl.py `
  --input datasets/tests/316/316_L.stl `
  --jaw-type man `
  --device cpu `
  --output outputs/ios_seg/316_L_seg.vtp
```

Key arguments:
- `--jaw-type`: `man` for mandibular (下颌), `max` for maxillary (上颌).
- `--downsample`: target triangle count before inference (default `50000`).
- `--downsample-backend`: `vedo` (default) or `open3d` fallback when vedo fails.

The script writes a VTP file that contains both the original ios-model labels
(`Label_ios`) and FDI numbering (`Label`) for downstream pipelines.

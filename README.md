# Deepcare Orthodontics Pipeline

One-stop pipeline for intraoral 3D scans that delivers tooth segmentation and landmark annotation, already packaged for API deployment.

- **IOS-Model** (`src/IOS-Model/`) — TorchScript teeth segmentation that turns raw STL meshes into labeled VTP surfaces with ios-model labels and FDI numbering.
- **MeshSegNet** (`src/MeshSegNet/`) — mesh-based semantic segmentation training/inference pipeline tailored to orthodontic datasets.
- **PointNetReg** (`src/PointnetReg/`) — landmark regression on segmented teeth for downstream analytics and aligner workflows.
- **API & Wrappers** (`API/`) — production-ready entrypoints that orchestrate the individual models, bundle TorchScript assets, and expose evaluation utilities.
- **Datasets & Tools** (`datasets/`, `tools/`, `scripts/`) — test fixtures, quickstart scripts, and helper utilities for verification, metrics, and data preparation.

To try the full pipeline quickly, start with the helper scripts under `tools/` (e.g., `run_ios_model_quickstart.py`) and follow the comments inside each module. Each component keeps its own README for deeper configuration details.

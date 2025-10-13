from pathlib import Path
import random
import numpy as np
import pyvista as pv

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from iMeshSegNet.m0_dataset import (
    remap_labels_single_arch,
    _build_single_arch_label_maps,
    _infer_jaw_from_stem,
)


def main() -> None:
    root = Path("datasets/segmentation_dataset")
    files = sorted(root.glob("*.vtp"))
    upper = [f for f in files if f.stem.endswith("_U")]
    lower = [f for f in files if f.stem.endswith("_L")]
    random.seed(42)
    samples = random.sample(upper, min(2, len(upper))) + random.sample(lower, min(2, len(lower)))

    maps = _build_single_arch_label_maps(gingiva_src=0, gingiva_class=15, keep_void_zero=False)
    for path in samples:
        mesh = pv.read(str(path))
        labels = None
        for key in ("Label", "labels"):
            if key in mesh.cell_data:
                labels = mesh.cell_data[key]
                break
        if labels is None:
            print(f"{path.name}: missing label array")
            continue
        raw = np.asarray(labels, dtype=np.int64)
        post = remap_labels_single_arch(raw, path, maps)
        print(path.name)
        print("  raw_unique :", sorted(np.unique(raw).tolist()))
        print("  post_unique:", sorted(np.unique(post).tolist()))

    try:
        _infer_jaw_from_stem("123")
    except Exception as exc:
        print("No suffix check ->", repr(exc))


if __name__ == "__main__":
    main()

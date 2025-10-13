from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from iMeshSegNet.m0_dataset import _build_single_arch_label_maps  # noqa: E402


def main() -> None:
    try:
        _build_single_arch_label_maps(gingiva_src=0, gingiva_class=15, keep_void_zero=True)
    except Exception as exc:  # pragma: no cover - just reporting
        print("Conflict check raised:", repr(exc))
    else:
        print("Conflict check FAILED to raise.")


if __name__ == "__main__":
    main()

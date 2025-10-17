#!/usr/bin/env python3
import sys

import numpy as np
import vtk


PALETTE = np.array(
    [
        [200, 200, 200],
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
        [255, 128, 0],
        [128, 0, 255],
        [0, 128, 255],
        [128, 255, 0],
        [255, 0, 128],
        [0, 255, 128],
        [128, 0, 0],
        [0, 128, 0],
        [0, 0, 128],
    ],
    dtype=np.uint8,
)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: colorize_vtp.py <input.vtp> <output.vtp>")

    fn_in, fn_out = sys.argv[1], sys.argv[2]
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(fn_in)
    reader.Update()
    pd = reader.GetOutput()
    cd = pd.GetCellData()

    names = [cd.GetArrayName(i) for i in range(cd.GetNumberOfArrays())]
    target = next(
        (nm for nm in names if nm and nm.lower() in {"predictedid", "label", "labels", "pred", "predlabel"}),
        None,
    )
    if target is None:
        raise RuntimeError(f"Unable to locate prediction array in {fn_in}; found: {names}")

    arr = cd.GetArray(target)
    n = arr.GetNumberOfTuples()
    labels = np.fromiter((int(arr.GetTuple1(i)) for i in range(n)), dtype=np.int32, count=n)
    uniq = np.unique(labels)
    lut = {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(sorted(uniq))}
    rgb = np.vstack([lut[lab] for lab in labels]).astype(np.uint8)

    rgb_vtk = vtk.vtkUnsignedCharArray()
    rgb_vtk.SetNumberOfComponents(3)
    rgb_vtk.SetName("RGB")
    for color in rgb:
        rgb_vtk.InsertNextTypedTuple(color.tolist())

    cd.AddArray(rgb_vtk)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(fn_out)
    writer.SetInputData(pd)
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write {fn_out}")
    print(f"Wrote {fn_out} with RGB from {target}")


if __name__ == "__main__":
    main()

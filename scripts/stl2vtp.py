#!/usr/bin/env python3
import sys

import vtk


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: stl2vtp.py <input.stl> <output.vtp>")

    inp, outp = sys.argv[1], sys.argv[2]

    reader = vtk.vtkSTLReader()
    reader.SetFileName(inp)
    reader.Update()

    poly = reader.GetOutput()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.SplittingOff()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(outp)
    writer.SetInputData(normals.GetOutput())
    if writer.Write() != 1:
        raise RuntimeError(f"Failed to write {outp}")

    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()

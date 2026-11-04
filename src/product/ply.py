"""Đọc file PLY nhị phân của 3DGS, như gsplat `export_splats(format="ply")` ghi.

Bố cục của gsplat @6e8c837 (gsplat/exporter.py, splat2ply_bytes): một element `vertex`, mọi thuộc tính là float32
little-endian, theo thứ tự x y z, f_dc_0..2, f_rest_0..(3K-1), opacity, scale_0..2, rot_0..3 (quaternion w x y z).
Module đọc theo header (tên và kiểu từng thuộc tính), nên cũng đọc được PLY 3DGS ghi theo thứ tự khác hoặc có thêm
thuộc tính (vd. nx ny nz của bản gốc Inria).
"""

from __future__ import annotations

from pathlib import Path

PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}
MAX_HEADER_BYTES = 1 << 20


class PlyError(ValueError):
    """File không phải PLY nhị phân đọc được."""


def read_header(path: Path) -> dict:
    """{'format', 'elements': [{'name', 'count', 'properties': [(tên, kiểu)]}], 'header_bytes'}."""
    path = Path(path)
    elements: list[dict] = []
    fmt = None
    with open(path, "rb") as handle:
        if handle.readline().strip() != b"ply":
            raise PlyError(f"{path}: not a PLY file")
        while True:
            raw = handle.readline()
            if not raw or handle.tell() > MAX_HEADER_BYTES:
                raise PlyError(f"{path}: PLY header has no end_header")
            words = raw.decode("ascii", errors="replace").split()
            if not words or words[0] in ("comment", "obj_info"):
                continue
            if words[0] == "end_header":
                return {"format": fmt, "elements": elements, "header_bytes": handle.tell()}
            if words[0] == "format" and len(words) >= 2:
                fmt = words[1]
            elif words[0] == "element" and len(words) == 3 and words[2].isdigit():
                elements.append({"name": words[1], "count": int(words[2]), "properties": []})
            elif words[0] == "property" and elements:
                if words[1] == "list":
                    elements[-1]["properties"].append((words[-1], "list"))
                elif len(words) == 3 and words[1] in PLY_TYPES:
                    elements[-1]["properties"].append((words[2], words[1]))
                else:
                    raise PlyError(f"{path}: unsupported PLY property line: {raw!r}")
            else:
                raise PlyError(f"{path}: unexpected PLY header line: {raw!r}")


def vertex_count(path: Path) -> int:
    for element in read_header(path)["elements"]:
        if element["name"] == "vertex":
            return element["count"]
    raise PlyError(f"{path}: no vertex element")


def read_vertices(path: Path):
    """Mảng có cấu trúc (numpy) của element `vertex`; tên trường = tên thuộc tính trong header."""
    import numpy as np

    path = Path(path)
    header = read_header(path)
    if header["format"] != "binary_little_endian":
        raise PlyError(f"{path}: only binary_little_endian PLY files are supported, got {header['format']}")
    if not header["elements"] or header["elements"][0]["name"] != "vertex":
        raise PlyError(f"{path}: the first PLY element must be 'vertex'")
    vertex = header["elements"][0]
    if any(kind == "list" for _, kind in vertex["properties"]):
        raise PlyError(f"{path}: list properties are not supported in the vertex element")
    dtype = np.dtype([(name, "<" + PLY_TYPES[kind]) for name, kind in vertex["properties"]])
    expected = header["header_bytes"] + vertex["count"] * dtype.itemsize
    size = path.stat().st_size
    if size < expected:
        raise PlyError(f"{path}: file is truncated ({size} bytes, the header announces {expected})")
    with open(path, "rb") as handle:
        handle.seek(header["header_bytes"])
        return np.fromfile(handle, dtype=dtype, count=vertex["count"])

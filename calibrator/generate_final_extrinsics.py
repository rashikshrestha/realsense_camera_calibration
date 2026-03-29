#!/usr/bin/env python3
"""Generate a multi-camera extrinsics YAML from a camera_array TOML.

Reads rotation vectors and translations from `config/camera_array.toml`,
converts each rotation vector (Rodrigues) to a 3x3 rotation matrix, builds
a 3x4 extrinsic matrix [R|t] for each camera, and writes them to
`config/extrinsics.yaml` (or a user-specified path).

Usage:
    python extrinsic_generate.py 
    python extrinsic_generate.py --input config/camera_array.toml --output config/extrinsics.yaml
"""

from __future__ import annotations
import sys
import math
from pathlib import Path
from typing import List

# Prefer stdlib tomllib (Py3.11+), fall back to toml package if available
try:
    import tomllib as _toml
except Exception:
    try:
        import toml as _toml  # type: ignore
    except Exception:
        _toml = None  # type: ignore


def rodrigues_to_matrix(r: List[float]):
    """Convert a Rodrigues rotation vector (3,) to a 3x3 rotation matrix.

    Implements the Rodrigues formula without OpenCV.
    """
    import numpy as np

    r = np.asarray(r, dtype=float).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta == 0.0:
        return np.eye(3)
    k = r / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=float)
    R = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    return R


def format_float_list(lst: List[float]) -> str:
    return "[" + ", ".join(f"{float(v):.12g}" for v in lst) + "]"


def build_extrinsics_from_toml(toml_data):
    import numpy as np
    
    cameras = toml_data.get("cameras") if isinstance(toml_data, dict) else None
    if not cameras:
        # Some toml loaders return a dict with top-level keys like 'cameras.0'
        # Try to detect keys like 'cameras.0' and group them.
        cameras = {}
        for k, v in (toml_data or {}).items():
            if k.startswith("cameras"):
                # key can be 'cameras.0' or 'cameras'
                parts = k.split(".")
                if len(parts) >= 2 and parts[1].isdigit():
                    idx = parts[1]
                    cameras.setdefault(int(idx), v)

    # If cameras is a mapping keyed by indices or strings
    extrinsics = []
    if isinstance(cameras, dict):
        # sort by key to keep deterministic order
        for key in sorted(cameras.keys(), key=lambda x: int(x)):
            cam = cameras[key]
            rot = cam.get("rotation") or cam.get("rotation_vector")
            trans = cam.get("translation")
            cam_id = cam.get("cam_id", int(key))
            if rot is None or trans is None:
                continue
            # Convert Rodrigues to rotation matrix (camera pose: camera-to-world)
            R_cam_to_world = rodrigues_to_matrix(rot)
            t_cam_to_world = np.asarray([float(x) for x in trans])
            
            # Invert to get extrinsics (world-to-camera)
            # R_world_to_camera = R_cam_to_world^T
            # t_world_to_camera = -R_cam_to_world^T @ t_cam_to_world
            R_ext = R_cam_to_world.T
            t_ext = -R_ext @ t_cam_to_world
            
            extrinsic = {
                "cam_id": int(cam_id),
                "matrix": [
                    [float(R_ext[0, 0]), float(R_ext[0, 1]), float(R_ext[0, 2]), float(t_ext[0])],
                    [float(R_ext[1, 0]), float(R_ext[1, 1]), float(R_ext[1, 2]), float(t_ext[1])],
                    [float(R_ext[2, 0]), float(R_ext[2, 1]), float(R_ext[2, 2]), float(t_ext[2])],
                ],
            }
            extrinsics.append(extrinsic)
    else:
        # If cameras is a list
        for cam in cameras:
            rot = cam.get("rotation") or cam.get("rotation_vector")
            trans = cam.get("translation")
            cam_id = cam.get("cam_id")
            if rot is None or trans is None:
                continue
            # Convert Rodrigues to rotation matrix (camera pose: camera-to-world)
            R_cam_to_world = rodrigues_to_matrix(rot)
            t_cam_to_world = np.asarray([float(x) for x in trans])
            
            # Invert to get extrinsics (world-to-camera)
            # R_world_to_camera = R_cam_to_world^T
            # t_world_to_camera = -R_cam_to_world^T @ t_cam_to_world
            R_ext = R_cam_to_world.T
            t_ext = -R_ext @ t_cam_to_world
            
            extrinsics.append(
                {
                    "cam_id": int(cam_id) if cam_id is not None else None,
                    "matrix": [
                        [float(R_ext[0, 0]), float(R_ext[0, 1]), float(R_ext[0, 2]), float(t_ext[0])],
                        [float(R_ext[1, 0]), float(R_ext[1, 1]), float(R_ext[1, 2]), float(t_ext[1])],
                        [float(R_ext[2, 0]), float(R_ext[2, 1]), float(R_ext[2, 2]), float(t_ext[2])],
                    ],
                }
            )

    # sort extrinsics by cam_id if present
    extrinsics.sort(key=lambda e: e.get("cam_id") if e.get("cam_id") is not None else 0)
    return extrinsics


def write_extrinsics_yaml(extrinsics, out_path: Path):
    # Create a simple YAML structure without external deps
    lines = []
    lines.append("# Generated multi-camera extrinsics\n")
    lines.append("cams:")
    for ex in extrinsics:
        cam_id = ex.get("cam_id")
        lines.append(f"  - id: {int(cam_id) if cam_id is not None else 'null'}")
        lines.append("    extrinsic:")
        for row in ex["matrix"]:
            lines.append(f"      - {format_float_list(row)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def main(argv=None):
    import argparse
    argv = argv if argv is not None else sys.argv[1:]
    ap = argparse.ArgumentParser(description="Generate multi-camera extrinsics YAML from camera_array TOML")
    ap.add_argument("--input", "-i", default="config/camera_array.toml", help="Path to camera_array TOML")
    ap.add_argument("--output", "-o", default="config/extrinsics_multi.yaml", help="Path to write extrinsics YAML")
    args = ap.parse_args(argv)

    in_path = Path(args.input)
    out_path = Path(args.output)

    if _toml is None:
        print("ERROR: No TOML parser available. Install the 'toml' package or use Python 3.11+.", file=sys.stderr)
        return 2

    if not in_path.exists():
        print(f"ERROR: Input file not found: {in_path}", file=sys.stderr)
        return 2

    data = None
    with in_path.open("rb") as f:
        # tomllib loads from bytes; toml package expects text
        if hasattr(_toml, "loads") and isinstance(b"", bytes):
            # tomllib
            try:
                data = _toml.load(f)
            except Exception:
                f.seek(0)
                text = f.read().decode("utf-8")
                data = _toml.loads(text)
        else:
            # toml package
            text = f.read().decode("utf-8")
            data = _toml.loads(text)

    extrinsics = build_extrinsics_from_toml(data)
    if not extrinsics:
        print("No valid camera extrinsics found in the TOML file.", file=sys.stderr)
        return 1

    write_extrinsics_yaml(extrinsics, out_path)
    print(f"Wrote {len(extrinsics)} extrinsics to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Generate camera_array.toml from camera_mapping.yaml and camera_intrinsics.yaml

This script combines camera mapping and intrinsic calibration data into a single
TOML file that describes the camera array configuration.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_yaml(filepath: str) -> Dict[str, Any]:
    """Load YAML file and return its contents."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def load_camera_mapping(filepath: str) -> Dict[int, Dict[str, str]]:
    """
    Load camera mapping and convert to a dict keyed by cam_id.
    Returns: {cam_id: {'serial': ..., 'name': ...}, ...}
    """
    data = load_yaml(filepath)
    mapping = {}
    for cam_key, cam_info in data.items():
        # Extract cam_id from keys like 'cam_0', 'cam_1', etc.
        cam_id = int(cam_key.split('_')[1])
        mapping[cam_id] = cam_info
    return mapping


def load_camera_intrinsics(filepath: str) -> Dict[str, Dict[str, Any]]:
    """
    Load camera intrinsics file.
    Returns: {serial: intrinsic_data, ...}
    """
    return load_yaml(filepath)


def extract_matrix_and_distortions(color_stream: Dict[str, Any]) -> tuple:
    """
    Extract 3x3 camera matrix and distortion coefficients from color stream data.
    
    Args:
        color_stream: Dict with keys like 'width', 'height', 'ppx', 'ppy', 'fx', 'fy', 'coefficients'
    
    Returns:
        (matrix, distortions) where matrix is 3x3 and distortions is list of 5 coefficients
    """
    ppx = color_stream.get('ppx', 0.0)
    ppy = color_stream.get('ppy', 0.0)
    fx = color_stream.get('fx', 0.0)
    fy = color_stream.get('fy', 0.0)
    
    # Camera matrix [fx, 0, ppx]
    #               [0, fy, ppy]
    #               [0, 0, 1]
    matrix = [
        [fx, 0.0, ppx],
        [0.0, fy, ppy],
        [0.0, 0.0, 1.0]
    ]
    
    # Distortion coefficients (k1, k2, p1, p2, k3)
    distortions = color_stream.get('coefficients', [0.0, 0.0, 0.0, 0.0, 0.0])
    
    return matrix, distortions


def write_toml(data: Dict[str, Any], filepath: str) -> None:
    """
    Write data to TOML file (manual implementation without toml library).
    
    Args:
        data: Dictionary with structure {'cameras': {cam_id: camera_data, ...}}
        filepath: Path to write TOML file
    """
    lines = []
    
    if 'cameras' in data:
        for cam_id, cam_data in data['cameras'].items():
            lines.append(f"[cameras.{cam_id}]")
            
            # Write scalar values
            for key in ['cam_id', 'rotation_count', 'grid_count']:
                if key in cam_data:
                    lines.append(f"{key} = {cam_data[key]}")
            
            if 'error' in cam_data:
                lines.append(f"error = {cam_data['error']}")
            
            if 'fisheye' in cam_data:
                fisheye_str = "true" if cam_data['fisheye'] else "false"
                lines.append(f"fisheye = {fisheye_str}")
            
            # Write array values
            if 'size' in cam_data:
                size = cam_data['size']
                lines.append(f"size = [{size[0]}, {size[1]}]")
            
            if 'matrix' in cam_data:
                matrix = cam_data['matrix']
                # Format matrix in compact form: [[row1], [row2], [row3]]
                matrix_rows = []
                for row in matrix:
                    row_str = "[" + ", ".join(f"{val}" for val in row) + "]"
                    matrix_rows.append(row_str)
                matrix_str = "[" + ", ".join(matrix_rows) + "]"
                lines.append(f"matrix = {matrix_str}")
            
            if 'distortions' in cam_data:
                distortions = cam_data['distortions']
                distortions_str = "[" + ", ".join(f"{val}" for val in distortions) + "]"
                lines.append(f"distortions = {distortions_str}")
            
            lines.append("")  # blank line between sections
    
    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))


def generate_camera_array(
    camera_mapping_file: str,
    camera_intrinsics_file: str,
    output_file: str,
    defaults: Dict[str, Any] = None
) -> None:
    """
    Generate camera_array.toml from mapping and intrinsics files.
    
    Args:
        camera_mapping_file: Path to camera_mapping.yaml
        camera_intrinsics_file: Path to camera_intrinsics.yaml
        output_file: Path where to write camera_array.toml
        defaults: Default values for rotation_count, error, grid_count, fisheye
    """
    if defaults is None:
        defaults = {
            'rotation_count': 0,
            'error': 0.4,
            'grid_count': 30,
            'fisheye': False
        }
    
    # Load data
    camera_mapping = load_camera_mapping(camera_mapping_file)
    camera_intrinsics = load_camera_intrinsics(camera_intrinsics_file)
    
    # Build camera array
    cameras = {}
    
    for cam_id in sorted(camera_mapping.keys()):
        mapping_info = camera_mapping[cam_id]
        serial = mapping_info['serial']
        name = mapping_info['name']
        
        # Look up intrinsics by serial
        if serial not in camera_intrinsics:
            print(f"Warning: No intrinsics found for camera {cam_id} (serial: {serial})")
            continue
        
        intrinsics = camera_intrinsics[serial]
        color_stream = intrinsics.get('streams', {}).get('color', {})
        
        if not color_stream:
            print(f"Warning: No color stream data for camera {cam_id} (serial: {serial})")
            continue
        
        # Extract size and camera parameters
        width = color_stream.get('width', 640)
        height = color_stream.get('height', 480)
        matrix, distortions = extract_matrix_and_distortions(color_stream)
        
        # Build camera entry
        camera_entry = {
            'cam_id': cam_id,
            'rotation_count': defaults['rotation_count'],
            'error': defaults['error'],
            'grid_count': defaults['grid_count'],
            'fisheye': defaults['fisheye'],
            'size': [width, height],
            'matrix': matrix,
            'distortions': distortions
        }
        
        cameras[str(cam_id)] = camera_entry
    
    # Convert to TOML format
    output_data = {'cameras': cameras}
    
    # Write to file
    write_toml(output_data, output_file)
    
    print(f"Generated {output_file} with {len(cameras)} cameras")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate camera_array.toml from camera_mapping.yaml and camera_intrinsics.yaml'
    )
    parser.add_argument(
        '--mapping',
        default='camera_mapping.yaml',
        help='Path to camera_mapping.yaml (default: camera_mapping.yaml)'
    )
    parser.add_argument(
        '--intrinsics',
        default='camera_intrinsics.yaml',
        help='Path to camera_intrinsics.yaml (default: camera_intrinsics.yaml)'
    )
    parser.add_argument(
        '--output',
        default='camera_array.toml',
        help='Path where to write camera_array.toml (default: camera_array.toml)'
    )
    parser.add_argument(
        '--error',
        type=float,
        default=0.4,
        help='Default error value for all cameras (default: 0.4)'
    )
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not Path(args.mapping).exists():
        print(f"Error: {args.mapping} not found")
        return 1
    
    if not Path(args.intrinsics).exists():
        print(f"Error: {args.intrinsics} not found")
        return 1
    
    defaults = {
        'rotation_count': 0,
        'error': args.error,
        'grid_count': 30,
        'fisheye': False
    }
    
    try:
        generate_camera_array(args.mapping, args.intrinsics, args.output, defaults)
        print(f"Successfully wrote to {args.output}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())

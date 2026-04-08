"""
origin_solver.py

Utilities for redefining the world coordinate origin to be at an ArUco marker
detected from a specific camera.

Problem Setup
-------------
We have N cameras, each with a known extrinsic matrix T_world_to_cam (4x4)
that transforms a point from world coordinates into camera coordinates:

    p_cam = T_world_to_cam @ p_world

An ArUco marker has been detected from one of these cameras (the "reference" camera).
The detection gives us:
    - rvec : rotation vector  (Rodrigues) of the marker in camera frame
    - tvec : translation vector of the marker center in camera frame

Goal
----
Make the ArUco marker the new world origin.
Compute new extrinsics for every camera so that:
    p_cam = T_new_world_to_cam @ p_new_world

where the new world frame coincides with the ArUco marker frame.

Math
----
Let:
    T_wc  = T_world_to_cam   (4x4, for the reference camera, from cam_config)
    T_cw  = inv(T_wc)        (4x4, camera pose in old world frame)
    T_ca  = T_cam_to_aruco   (4x4, aruco pose in camera frame, from rvec/tvec)
    T_wa  = T_cw @ T_ca      (4x4, aruco pose in old world frame)

New extrinsic for any camera:
    T_new_wc = T_wc @ inv(T_wa)
             = T_wc @ T_aw

Because: p_cam = T_wc @ p_world
                = T_wc @ (T_aw @ p_new_world)     [since p_world = T_aw @ p_new_world]
                = (T_wc @ T_aw) @ p_new_world
"""

import numpy as np
import yaml
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    Convert a Rodrigues rotation vector and translation vector to a 4x4
    homogeneous transformation matrix.

        T = | R  t |
            | 0  1 |

    Args:
        rvec: (3,1) or (3,) rotation vector
        tvec: (3,1) or (3,) translation vector

    Returns:
        (4, 4) float64 transformation matrix
    """
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def T_to_rvec_tvec(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose a 4x4 homogeneous transformation matrix into a Rodrigues
    rotation vector and a translation vector.

    Args:
        T: (4, 4) transformation matrix

    Returns:
        rvec: (3, 1) rotation vector
        tvec: (3, 1) translation vector
    """
    R = T[:3, :3]
    t = T[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    return rvec, t.reshape(3, 1)


def extrinsic_3x4_to_T(extrinsic_3x4: List[List[float]]) -> np.ndarray:
    """
    Convert a 3x4 extrinsic matrix (as stored in cam_config.yaml) to a 4x4
    homogeneous transformation matrix.

    Args:
        extrinsic_3x4: 3x4 list of lists

    Returns:
        (4, 4) float64 transformation matrix
    """
    E = np.array(extrinsic_3x4, dtype=np.float64)   # shape (3, 4)
    T = np.eye(4, dtype=np.float64)
    T[:3, :] = E
    return T


def T_to_extrinsic_3x4(T: np.ndarray) -> List[List[float]]:
    """
    Convert a 4x4 homogeneous transformation matrix back to a 3x4 list of
    lists (for serialising back to YAML).

    Args:
        T: (4, 4) transformation matrix

    Returns:
        3x4 list of lists (Python floats)
    """
    return T[:3, :].tolist()


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def compute_aruco_pose_in_world(
    T_world_to_refcam: np.ndarray,
    rvec_aruco_in_refcam: np.ndarray,
    tvec_aruco_in_refcam: np.ndarray
) -> np.ndarray:
    """
    Compute the ArUco marker pose expressed in the current (old) world frame.

    Args:
        T_world_to_refcam : (4,4) extrinsic of the reference camera
                            (transforms world coords → camera coords)
        rvec_aruco_in_refcam : (3,1) or (3,) rvec from solvePnP
        tvec_aruco_in_refcam : (3,1) or (3,) tvec from solvePnP

    Returns:
        T_world_to_aruco: (4,4) pose of ArUco in old world frame
    """
    # Camera pose in world  (T_cam_to_world = inv(T_world_to_cam))
    T_refcam_to_world = np.linalg.inv(T_world_to_refcam)

    # ArUco pose in camera frame
    T_aruco_in_refcam = rvec_tvec_to_T(rvec_aruco_in_refcam, tvec_aruco_in_refcam)

    # ArUco pose in world frame
    T_aruco_in_world = T_refcam_to_world @ T_aruco_in_refcam

    return T_aruco_in_world


def rebase_extrinsics(
    cameras_config: List[Dict],
    T_aruco_in_world: np.ndarray
) -> List[Dict]:
    """
    Recompute every camera's extrinsic so that the ArUco marker frame becomes
    the new world origin.

    New extrinsic:
        T_new_world_to_cam = T_old_world_to_cam @ T_aruco_in_world

    Because the new world origin is the ArUco, any point expressed in the new
    world frame satisfies:
        p_old_world = T_aruco_in_world @ p_new_world

    Substituting:
        p_cam = T_old_wc @ p_old_world
              = T_old_wc @ T_aruco_in_world @ p_new_world
              = T_new_wc @ p_new_world

    Args:
        cameras_config : list of camera dicts from cam_config.yaml
        T_aruco_in_world : (4,4) ArUco pose in the old world frame

    Returns:
        List of updated camera dicts with new 'extrinsic' values
    """
    updated_cameras = []

    for cam in cameras_config:
        cam_copy = dict(cam)

        if 'extrinsic' not in cam:
            print(f"  Warning: Camera {cam.get('id')} has no extrinsic, skipping.")
            updated_cameras.append(cam_copy)
            continue

        T_old_wc = extrinsic_3x4_to_T(cam['extrinsic'])
        T_new_wc = T_old_wc @ T_aruco_in_world
        cam_copy['extrinsic'] = T_to_extrinsic_3x4(T_new_wc)

        updated_cameras.append(cam_copy)

    return updated_cameras


def solve_new_origin(
    cameras_config: List[Dict],
    selected_pose_data: Dict
) -> Tuple[List[Dict], np.ndarray]:
    """
    High-level entry point.

    Given the full camera config and the pose data for the ArUco marker
    detected from one camera, rebase every camera's extrinsic so that the
    ArUco marker is the new world origin.

    Args:
        cameras_config    : list of camera dicts from cam_config.yaml
        selected_pose_data: dict as returned by interactive_pose_query(), e.g.
            {
                'camera_id': 0,
                'serial': '138422077690',
                'rvecs': [np.ndarray],   # rvec of ArUco[0] in camera frame
                'tvecs': [np.ndarray],   # tvec of ArUco[0] in camera frame
                ...
            }

    Returns:
        updated_cameras_config : list of camera dicts with new extrinsics
        T_aruco_in_world       : (4,4) pose of ArUco in the old world frame
    """
    ref_camera_id = selected_pose_data['camera_id']
    rvec = selected_pose_data['rvecs'][0]   # ArUco[0]
    tvec = selected_pose_data['tvecs'][0]

    # Find reference camera's extrinsic
    ref_cam = next((c for c in cameras_config if c.get('id') == ref_camera_id), None)
    if ref_cam is None:
        raise ValueError(f"Camera ID {ref_camera_id} not found in cameras_config")
    if 'extrinsic' not in ref_cam:
        raise ValueError(f"Camera ID {ref_camera_id} has no extrinsic in cameras_config")

    T_world_to_refcam = extrinsic_3x4_to_T(ref_cam['extrinsic'])

    print(f"\n{'='*60}")
    print(f"Reference Camera ID : {ref_camera_id}")
    print(f"Serial              : {selected_pose_data['serial']}")
    print(f"\nArUco[0] pose in camera frame:")
    print(f"  rvec:\n{rvec.reshape(3)}")
    print(f"  tvec:\n{tvec.reshape(3)}")

    # --- Step 1: find ArUco in old world frame ---
    T_aruco_in_world = compute_aruco_pose_in_world(T_world_to_refcam, rvec, tvec)

    print(f"\nArUco[0] pose in old world frame (4x4):")
    print(T_aruco_in_world)

    # --- Step 2: rebase all cameras ---
    updated_cameras = rebase_extrinsics(cameras_config, T_aruco_in_world)

    # --- Verification ---
    print(f"\n{'='*60}")
    print("Verification: reference camera extrinsic in new world frame.")
    print("(The translation of the reference camera should show how far the")
    print(" camera is from the ArUco marker.)\n")
    ref_new = next(c for c in updated_cameras if c.get('id') == ref_camera_id)
    T_new_wc = extrinsic_3x4_to_T(ref_new['extrinsic'])
    print(f"New T_world_to_refcam:\n{T_new_wc}\n")

    print("Sanity check: project new-world origin through updated reference camera.")
    origin_new_world = np.array([0, 0, 0, 1], dtype=np.float64)
    origin_in_refcam = T_new_wc @ origin_new_world
    print(f"  New world origin in reference camera coords: {origin_in_refcam[:3]}")
    print(f"  (Should be close to tvec: {tvec.reshape(3)})")
    print("="*60)

    return updated_cameras, T_aruco_in_world


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_cam_config(config_path: Path) -> Dict:
    """Load cam_config.yaml and return the parsed dict."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_cam_config(config: Dict, output_path: Path) -> None:
    """
    Save an updated camera config dict to a YAML file.

    Args:
        config     : full config dict (with 'cams' key)
        output_path: path to write the YAML file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Saved updated config to: {output_path}")


def update_cam_config_with_new_origin(
    workspace_root: Path,
    selected_pose_data: Dict,
    output_filename: str = "cam_config_new_origin.yaml"
) -> Path:
    """
    Full pipeline: load cam_config, compute new extrinsics, save result.

    Args:
        workspace_root    : workspace directory (contains cam_config.yaml)
        selected_pose_data: pose data dict from interactive_pose_query()
        output_filename   : filename for the updated config (saved in workspace_root)

    Returns:
        Path to the saved output file
    """
    config_path = workspace_root / "cam_config.yaml"
    config = load_cam_config(config_path)

    if 'cams' not in config:
        raise ValueError("cam_config.yaml has no 'cams' key")

    print(f"\nLoaded cam_config from: {config_path}")
    print(f"Number of cameras: {len(config['cams'])}")

    updated_cameras, T_aruco_in_world = solve_new_origin(config['cams'], selected_pose_data)

    updated_config = dict(config)
    updated_config['cams'] = updated_cameras

    output_path = workspace_root / output_filename
    save_cam_config(updated_config, output_path)

    return output_path


# ---------------------------------------------------------------------------
# Quick standalone test (runs without cameras)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rebase camera extrinsics to an ArUco marker origin."
    )
    parser.add_argument(
        "--workspace", "-w",
        type=Path,
        required=True,
        help="Workspace directory containing cam_config.yaml"
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        required=True,
        help="ID of the camera that detected the ArUco marker"
    )
    parser.add_argument(
        "--rvec",
        type=float,
        nargs=3,
        required=True,
        metavar=('rx', 'ry', 'rz'),
        help="Rotation vector of ArUco[0] in the reference camera frame"
    )
    parser.add_argument(
        "--tvec",
        type=float,
        nargs=3,
        required=True,
        metavar=('tx', 'ty', 'tz'),
        help="Translation vector of ArUco[0] in the reference camera frame"
    )
    parser.add_argument(
        "--serial",
        type=str,
        default="",
        help="Serial number of the reference camera (optional, for display)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cam_config_new_origin.yaml",
        help="Output YAML filename (default: cam_config_new_origin.yaml)"
    )
    args = parser.parse_args()

    # Build a minimal pose_data dict identical to what set_origin.py produces
    pose_data = {
        'camera_id': args.camera_id,
        'serial': args.serial,
        'camera_name': 'CLI input',
        'rvecs': [np.array(args.rvec, dtype=np.float64).reshape(3, 1)],
        'tvecs': [np.array(args.tvec, dtype=np.float64).reshape(3, 1)],
        'marker_ids': [0],
        'num_markers': 1,
    }

    out_path = update_cam_config_with_new_origin(
        args.workspace, pose_data, output_filename=args.output
    )
    print(f"\nDone. New config saved to: {out_path}")

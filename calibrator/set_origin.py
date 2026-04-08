"""
set_origin.py

Captures a single image from each RealSense camera configured in cam_config.yaml
and displays them to the user for reference.
"""

import cv2
import yaml
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_recorder.camlib import CameraManager
from calibrator.aruco_detector import ArucoDetector


def load_camera_config(config_path: str) -> Dict:
    """Load camera configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_intrinsics(intrinsics_dir: Path, serial: str, stream: str = 'color') -> Optional[Dict]:
    """
    Load camera intrinsic parameters from YAML file.
    
    Args:
        intrinsics_dir: Path to directory containing intrinsic YAML files
        serial: Camera serial number
        stream: Stream type ('color' or 'depth')
    
    Returns:
        Dictionary containing camera matrix and distortion coefficients, or None if not found
    """
    intrinsics_file = intrinsics_dir / f"{serial}.yaml"
    
    if not intrinsics_file.exists():
        print(f"    Warning: Intrinsics file not found for serial {serial}")
        return None
    
    try:
        with open(intrinsics_file, 'r') as f:
            intrinsics = yaml.safe_load(f)
        
        if 'streams' not in intrinsics or stream not in intrinsics['streams']:
            print(f"    Warning: Stream '{stream}' not found in intrinsics")
            return None
        
        stream_data = intrinsics['streams'][stream]
        
        # Extract camera matrix parameters
        fx = stream_data.get('fx', 1.0)
        fy = stream_data.get('fy', 1.0)
        ppx = stream_data.get('ppx', 0.0)
        ppy = stream_data.get('ppy', 0.0)
        
        # Create camera matrix
        camera_matrix = np.array([
            [fx, 0, ppx],
            [0, fy, ppy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Extract distortion coefficients
        coefficients = stream_data.get('coefficients', [0, 0, 0, 0, 0])
        dist_coeffs = np.array(coefficients[:5], dtype=np.float32).reshape(5, 1)
        
        return {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'width': stream_data.get('width', 640),
            'height': stream_data.get('height', 480)
        }
    
    except Exception as e:
        print(f"    Warning: Error loading intrinsics for serial {serial}: {e}")
        return None


def get_camera_id_from_serial(cameras_config: List[Dict], serial: str) -> Optional[int]:
    """
    Get camera ID from configuration by serial number.
    
    Args:
        cameras_config: List of camera configurations
        serial: Camera serial number
    
    Returns:
        Camera ID or None if not found
    """
    for cam_config in cameras_config:
        if cam_config.get('serial') == serial:
            return cam_config.get('id')
    return None


def process_single_frame(frame_data: Dict, detector: ArucoDetector, 
                        intrinsics_dir: Path, cameras_config: List[Dict],
                        output_dir: Path) -> Optional[Dict]:
    """
    Process a single camera frame: detect ArUco markers, estimate pose, and save.
    
    Args:
        frame_data: Captured frame data from camera manager
        detector: ArucoDetector instance
        intrinsics_dir: Path to intrinsics directory
        cameras_config: List of camera configurations
        output_dir: Output directory for saving annotated images
    
    Returns:
        Dictionary with pose data for this camera, or None if processing failed
    """
    camera_name = frame_data['name']
    serial = frame_data['serial']
    image = frame_data['image']
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Detect ArUco markers
    corners, ids = detector.detect(image_bgr)
    marker_info = detector.get_marker_info(corners, ids)
    
    # Load intrinsics
    intrinsics = load_intrinsics(intrinsics_dir, serial, stream='color')
    if intrinsics is None:
        print(f"    ✗ Failed to process camera {serial}: no intrinsics found")
        return None
    
    camera_matrix = intrinsics['camera_matrix']
    dist_coeffs = intrinsics['dist_coeffs']
    
    # Estimate pose
    rvecs, tvecs = detector.estimate_pose(image_bgr, corners, camera_matrix, dist_coeffs)
    
    # Draw markers with pose
    image_with_markers = detector.draw_markers_with_pose(
        image_bgr, corners, ids, rvecs, tvecs, camera_matrix, dist_coeffs
    )
    
    # Get camera ID
    camera_id = get_camera_id_from_serial(cameras_config, serial)
    
    # Create filenames and titles
    if camera_id is not None:
        window_title = f"Camera ID: {camera_id} (Serial: {serial}) - ArUco: {marker_info['num_markers']} markers"
        filename = f"{camera_id}_{serial}.png"
    else:
        window_title = f"{camera_name} (Serial: {serial}) - ArUco: {marker_info['num_markers']} markers"
        filename = f"{serial}.png"
    
    # Save annotated image
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    success = cv2.imwrite(str(filepath), image_with_markers)
    
    if success:
        print(f"    ✓ Saved: {filename}")
    else:
        print(f"    ✗ Failed to save: {filename}")
    
    # Display image
    cv2.imshow(window_title, image_with_markers)
    print(f"    ✓ Displayed: {window_title}")
    
    # Store and return pose data
    if camera_id is not None:
        return {
            'camera_id': camera_id,
            'serial': serial,
            'camera_name': camera_name,
            'rvecs': rvecs,
            'tvecs': tvecs,
            'marker_ids': marker_info['marker_ids'],
            'num_markers': marker_info['num_markers']
        }
    
    return None


def display_and_process_frames(frames: Dict, detector: ArucoDetector,
                               intrinsics_dir: Path, cameras_config: List[Dict],
                               output_dir: Path) -> Dict:
    """
    Display and process all captured frames.
    
    Args:
        frames: Dictionary of captured frames
        detector: ArucoDetector instance
        intrinsics_dir: Path to intrinsics directory
        cameras_config: List of camera configurations
        output_dir: Output directory for saving annotated images
    
    Returns:
        Dictionary mapping camera IDs to their pose data
    """
    print("\nProcessing and displaying captured images...")
    pose_data = {}
    
    for cam_key, frame_data in frames.items():
        result = process_single_frame(frame_data, detector, intrinsics_dir, 
                                     cameras_config, output_dir)
        if result is not None:
            pose_data[result['camera_id']] = result
    
    return pose_data


def print_pose_data(pose_data: Dict, camera_id: int) -> None:
    """
    Print pose information for a specific camera.
    
    Args:
        pose_data: Dictionary mapping camera IDs to pose data
        camera_id: Camera ID to print information for
    """
    cam_info = pose_data[camera_id]
    
    print(f"\n{'='*60}")
    print(f"Camera ID: {camera_id}")
    print(f"Serial: {cam_info['serial']}")
    print(f"Camera Name: {cam_info['camera_name']}")
    print(f"Number of ArUco markers detected: {cam_info['num_markers']}")
    
    if cam_info['num_markers'] > 0:
        print(f"\nArUco Marker IDs detected: {cam_info['marker_ids']}")
        print(f"\n--- ArUco[0] Pose Information ---")
        print(f"Rotation Vector (rvec):\n{cam_info['rvecs'][0]}")
        print(f"\nTranslation Vector (tvec):\n{cam_info['tvecs'][0]}")
    else:
        print("\nNo ArUco markers detected in this camera's frame")
    print("="*60)


def interactive_pose_query(pose_data: Dict) -> None:
    """
    Allow user to interactively query pose data for different cameras.
    
    Args:
        pose_data: Dictionary mapping camera IDs to pose data
    """
    print("\n" + "="*60)
    print("Available Camera IDs:")
    for camera_id in sorted(pose_data.keys()):
        cam_info = pose_data[camera_id]
        print(f"  Camera ID: {camera_id} (Serial: {cam_info['serial']}) - Markers: {cam_info['num_markers']}")
    
    while True:
        try:
            user_input = input("\nEnter Camera ID to view ArUco pose (or 'q' to quit): ").strip()
            
            if user_input.lower() == 'q':
                print("Exiting...")
                break
            
            camera_id = int(user_input)
            
            if camera_id not in pose_data:
                print(f"Error: Camera ID {camera_id} not found. Available IDs: {sorted(pose_data.keys())}")
                continue
            
            print_pose_data(pose_data, camera_id)
            
        except ValueError:
            print("Error: Invalid input. Please enter a valid Camera ID or 'q' to quit.")


def capture_and_display_images(
    workspace_root: Path, 
    width: int = 640, 
    height: int = 480, 
    fps: int = 30,
    aruco_dict_name: str = "DICT_6X6_250") -> None:
    """
    Main workflow: capture images, detect ArUco markers, and allow user interaction.
    
    Args:
        workspace_root: Root directory of the workspace
        width: Image width in pixels (default: 640)
        height: Image height in pixels (default: 480)
        fps: Frames per second (default: 30)
        aruco_dict_name: ArUco dictionary to use (default: DICT_6X6_250)
    """
    # Setup
    config_path = workspace_root / "cam_config.yaml"
    print(f"Loading camera configuration from: {config_path}")
    config = load_camera_config(config_path)
    
    if 'cams' not in config:
        print("Error: No 'cams' key found in configuration")
        return
    
    cameras_config = config['cams']
    print(f"Found {len(cameras_config)} camera(s) in configuration")
    
    # Initialize camera manager
    print("Initializing camera manager...")
    camera_mgr = CameraManager(width=width, height=height, fps=fps)
    
    print("Discovering and starting cameras...")
    started_cameras = camera_mgr.discover_and_start()
    
    if len(started_cameras) == 0:
        print("Error: No cameras were started successfully")
        return
    
    print(f"Successfully started {len(started_cameras)} camera(s)")
    
    # Setup output directories
    output_dir = workspace_root / "set_origin_data"
    intrinsics_dir = workspace_root / "intrinsic_config"
    
    # Initialize ArUco detector
    print(f"\nInitializing ArUco detector with dictionary: {aruco_dict_name}")
    detector = ArucoDetector(aruco_dict_name=aruco_dict_name)
    
    try:
        # Warm up cameras
        print("\nWarming up cameras (discarding first 30 frames)...")
        for i in range(30):
            camera_mgr.get_frames(timeout_ms=500)
        
        # Capture frames
        print("Capturing images from all cameras (31st frame)...")
        frames = camera_mgr.get_frames(timeout_ms=2000)
        
        if len(frames) == 0:
            print("Error: No frames captured from any camera")
            return
        
        # Process frames and display
        pose_data = display_and_process_frames(frames, detector, intrinsics_dir, 
                                              cameras_config, output_dir)
        
        print("\nPress any key to close the image windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Interactive query
        if pose_data:
            interactive_pose_query(pose_data)
        
    finally:
        # Cleanup
        print("\nStopping all cameras...")
        camera_mgr.stop_all()
        print("Done!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Capture images from RealSense cameras configured in cam_config.yaml and detect ArUco markers"
    )
    parser.add_argument(
        "--workspace", "-w",
        type=Path,
        required=True,
        help="Root directory of the workspace"
    )
    parser.add_argument(
        "--aruco-dict",
        type=str,
        default="DICT_6X6_250",
        help="ArUco dictionary to use (default: DICT_6X6_250)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Image width in pixels (default: 640)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height in pixels (default: 480)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)"
    )
    
    args = parser.parse_args()
    
    capture_and_display_images(
        args.workspace,
        width=args.width,
        height=args.height,
        fps=args.fps,
        aruco_dict_name=args.aruco_dict
    )


if __name__ == "__main__":
    main()

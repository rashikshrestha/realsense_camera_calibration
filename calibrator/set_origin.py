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


def save_images(frames: Dict, output_dir: Path) -> None:
    """
    Save captured frames to disk.
    
    Args:
        frames: Dictionary of captured frames from camera manager
        output_dir: Path to directory where images will be saved
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving images to: {output_dir}")
    for cam_key, frame_data in frames.items():
        serial = frame_data['serial']
        image = frame_data['image']
        
        # Create filename with serial number or key
        filename = f"camera_{serial or cam_key}.png"
        filepath = output_dir / filename
        
        # Convert RGB to BGR for OpenCV (which expects BGR)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Save the image
        success = cv2.imwrite(str(filepath), image_bgr)
        if success:
            print(f"  ✓ Saved: {filename}")
        else:
            print(f"  ✗ Failed to save: {filename}")


def capture_and_display_images(
    workspace_root: Path, 
    width: int = 640, 
    height: int = 480, 
    fps: int = 30,
    aruco_dict_name: str = "DICT_6X6_250") -> None:
    """
    Capture a single image from each camera in the configuration, detect ArUco markers,
    save them, and display them to the user.
    
    Args:
        workspace_root: Root directory of the workspace
        width: Image width in pixels (default: 640)
        height: Image height in pixels (default: 480)
        fps: Frames per second (default: 30)
        aruco_dict_name: ArUco dictionary to use (default: DICT_6X6_250)
    """
    # Load camera configuration
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
    
    # Discover and start cameras
    print("Discovering and starting cameras...")
    started_cameras = camera_mgr.discover_and_start()
    
    if len(started_cameras) == 0:
        print("Error: No cameras were started successfully")
        return
    
    print(f"Successfully started {len(started_cameras)} camera(s)")
    
    # Create output directory for saving images
    output_dir = workspace_root / "set_origin_data"
    intrinsics_dir = workspace_root / "intrinsic_config"
    
    # Initialize ArUco detector
    print(f"\nInitializing ArUco detector with dictionary: {aruco_dict_name}")
    detector = ArucoDetector(aruco_dict_name=aruco_dict_name)
    
    try:
        # Discard first 30 frames to allow cameras to warm up
        print("\nWarming up cameras (discarding first 30 frames)...")
        for i in range(30):
            camera_mgr.get_frames(timeout_ms=500)
        
        # Capture frames from all cameras (31st frame)
        print("Capturing images from all cameras (31st frame)...")
        frames = camera_mgr.get_frames(timeout_ms=2000)
        
        if len(frames) == 0:
            print("Error: No frames captured from any camera")
            return
        
        # Store pose data for later retrieval
        pose_data = {}
        
        # Display images
        print("\nDisplaying captured images...")
        for cam_key, frame_data in frames.items():
            camera_name = frame_data['name']
            serial = frame_data['serial']
            image = frame_data['image']
            
            # Convert RGB to BGR for OpenCV display
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Detect ArUco markers
            corners, ids = detector.detect(image_bgr)
            
            # Get marker info
            marker_info = detector.get_marker_info(corners, ids)
            
            # Load camera intrinsics
            intrinsics = load_intrinsics(intrinsics_dir, serial, stream='color')
            camera_matrix = intrinsics['camera_matrix']
            dist_coeffs = intrinsics['dist_coeffs']
            
            # Estimate pose and draw markers with axes
            rvecs, tvecs = detector.estimate_pose(image_bgr, corners, camera_matrix, dist_coeffs)
            image_with_markers = detector.draw_markers_with_pose(
                image_bgr, corners, ids, rvecs, tvecs, camera_matrix, dist_coeffs
            )
            
            # Get camera ID from configuration
            camera_id = None
            for cam_config in cameras_config:
                if cam_config.get('serial') == serial:
                    camera_id = cam_config.get('id')
                    break
            
            # Create window title with camera ID and serial
            if camera_id is not None:
                window_title = f"Camera ID: {camera_id} (Serial: {serial}) - ArUco: {marker_info['num_markers']} markers"
                annotated_filename = f"{camera_id}_{serial}.png"
            else:
                window_title = f"{camera_name} (Serial: {serial}) - ArUco: {marker_info['num_markers']} markers"
                annotated_filename = f"{serial or cam_key}.png"
            
            # Display the image with markers and pose
            cv2.imshow(window_title, image_with_markers)
            print(f"    ✓ Displayed: {window_title}")
            
            # Save annotated image
            annotated_filepath = output_dir / annotated_filename
            success = cv2.imwrite(str(annotated_filepath), image_with_markers)
            if success:
                print(f"    ✓ Saved aruco image: {annotated_filename}")
            else:
                print(f"    ✗ Failed to save aruco image: {annotated_filename}")
            
            # Store pose data for this camera
            if camera_id is not None:
                pose_data[camera_id] = {
                    'serial': serial,
                    'camera_name': camera_name,
                    'rvecs': rvecs,
                    'tvecs': tvecs,
                    'marker_ids': marker_info['marker_ids'],
                    'num_markers': marker_info['num_markers']
                }

        print("\nPress any key to close the image windows and exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Ask user for camera ID
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
                
            except ValueError:
                print("Error: Invalid input. Please enter a valid Camera ID or 'q' to quit.")
        
    finally:
        # Stop all cameras
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

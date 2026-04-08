"""
set_origin.py

Captures a single image from each RealSense camera configured in cam_config.yaml
and displays them to the user for reference.
"""

import cv2
import yaml
import sys
import argparse
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
            
            # Draw markers on image
            image_with_markers = detector.draw_markers(image_bgr, corners, ids)
            
            # Create window title
            window_title = f"{camera_name} (Serial: {serial}) - ArUco: {marker_info['num_markers']} markers"
            
            # Display the image with markers
            cv2.imshow(window_title, image_with_markers)
            print(f"    ✓ Displayed: {window_title}")
            
            # Save annotated image
            annotated_filename = f"{serial or cam_key}.png"
            annotated_filepath = output_dir / annotated_filename
            success = cv2.imwrite(str(annotated_filepath), image_with_markers)
            if success:
                print(f"    ✓ Saved aruco image: {annotated_filename}")
            else:
                print(f"    ✗ Failed to save aruco image: {annotated_filename}")

        print("\nPress any key to close the image windows and exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
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

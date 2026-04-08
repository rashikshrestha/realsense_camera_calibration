"""
set_origin.py

Captures a single image from each RealSense camera configured in cam_config.yaml
and displays them to the user for reference.
"""

import cv2
import yaml
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path to import camlib
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_recorder.camlib import CameraManager


def load_camera_config(config_path: str) -> Dict:
    """Load camera configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def capture_and_display_images(config_path: str, width: int = 640, height: int = 480, fps: int = 30) -> None:
    """
    Capture a single image from each camera in the configuration and display them.
    
    Args:
        config_path: Path to the cam_config.yaml file
        width: Image width in pixels (default: 640)
        height: Image height in pixels (default: 480)
        fps: Frames per second (default: 30)
    """
    # Load camera configuration
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
        
        print(f"Captured {len(frames)} image(s)")
        
        # Display images
        print("\nDisplaying captured images...")
        for cam_key, frame_data in frames.items():
            camera_name = frame_data['name']
            serial = frame_data['serial']
            image = frame_data['image']
            timestamp = frame_data['timestamp']
            frame_number = frame_data['frame_number']
            
            # Create window title
            window_title = f"{camera_name} (Serial: {serial})"
            if timestamp is not None:
                window_title += f" [T: {timestamp:.3f}s]"
            if frame_number is not None:
                window_title += f" [FN: {frame_number}]"
            
            # Convert RGB to BGR for OpenCV display
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Display the image
            cv2.imshow(window_title, image_bgr)
            print(f"  ✓ Displayed: {window_title}")
        
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
    # Try to find cam_config.yaml in data directory
    script_dir = Path(__file__).parent
    workspace_root = script_dir.parent
    
    # Check common locations for cam_config.yaml
    config_paths = [
        workspace_root / "data" / "flatop" / "cam_config.yaml",
        workspace_root / "data" / "cam_config.yaml",
        Path("cam_config.yaml"),
    ]
    
    config_path = None
    for path in config_paths:
        if path.exists():
            config_path = str(path)
            break
    
    if config_path is None:
        print("Error: Could not find cam_config.yaml")
        print(f"Searched in:")
        for path in config_paths:
            print(f"  - {path}")
        sys.exit(1)
    
    print(f"Using configuration: {config_path}\n")
    capture_and_display_images(config_path)


if __name__ == "__main__":
    main()

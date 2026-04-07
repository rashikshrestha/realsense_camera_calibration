"""
RealSense Multi-Camera ArUco Marker Detection
Captures RGB images from all connected RealSense cameras, detects ArUco markers,
and displays them in a 1x3 grid.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add data_recorder directory to path to import camlib
sys.path.insert(0, str(Path(__file__).parent.parent / "data_recorder"))
from camlib import CameraManager


def get_default_camera_matrix(width, height):
    """
    Generate a default camera matrix based on image dimensions.
    This assumes a 45-degree field of view.
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        np.ndarray: Camera matrix
    """
    focal_length = (width + height) / 2
    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    return camera_matrix


def get_default_dist_coeffs():
    """
    Get default distortion coefficients (no distortion).
    
    Returns:
        np.ndarray: Distortion coefficients
    """
    return np.zeros((4, 1), dtype=np.float32)


def draw_xyz_axis(image, corners, ids, camera_matrix, dist_coeffs, marker_length=0.23):
    """
    Draw XYZ axis on detected ArUco marker.
    
    Args:
        image: Input image
        corners: Marker corners from detectMarkers
        ids: Marker IDs from detectMarkers
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        marker_length: Size of the marker in meters (default: 0.23m = 230mm)
        
    Returns:
        np.ndarray: Image with axis drawn
    """
    if ids is None or len(ids) == 0:
        return image
    
    # Estimate pose for each marker using the newer API
    for i, corner in enumerate(corners):
        # Reshape corner for solvePnP
        corner = corner[0]  # Get the 4 corners
        
        # Define 3D object points for the marker (in marker coordinate system)
        object_points = np.array([
            [-marker_length/2, marker_length/2, 0],
            [marker_length/2, marker_length/2, 0],
            [marker_length/2, -marker_length/2, 0],
            [-marker_length/2, -marker_length/2, 0]
        ], dtype=np.float32)
        
        # Use solvePnP to estimate pose
        success, rvec, tvec = cv2.solvePnP(
            object_points, corner, camera_matrix, dist_coeffs
        )
        
        if success:
            # Draw axis
            cv2.drawFrameAxes(
                image, camera_matrix, dist_coeffs,
                rvec, tvec,
                length=marker_length * 0.5,  # Axis length
                thickness=2
            )
    
    return image


def detect_aruco_markers(image, aruco_dict_name='DICT_6X6_250', draw_axis=True):
    """
    Detect ArUco markers in an image and draw them with XYZ axis.
    
    Args:
        image: Input BGR image (numpy array)
        aruco_dict_name: Name of the ArUco dictionary to use
        draw_axis: Whether to draw XYZ axis on markers
        
    Returns:
        tuple: (annotated_image, detections) where detections is the result from detectMarkers
    """
    # Get the ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, aruco_dict_name)
    )
    
    # Create detector parameters
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    # Detect markers
    corners, ids, rejected = detector.detectMarkers(image)
    
    # Create a copy for annotation
    annotated_image = image.copy()
    
    # Draw detected markers
    if ids is not None and len(ids) > 0:
        annotated_image = cv2.aruco.drawDetectedMarkers(annotated_image, corners, ids)
        print(f"Detected {len(ids)} marker(s) with IDs: {ids.flatten().tolist()}")
        
        # Draw XYZ axis if requested
        if draw_axis:
            height, width = annotated_image.shape[:2]
            camera_matrix = get_default_camera_matrix(width, height)
            dist_coeffs = get_default_dist_coeffs()
            
            annotated_image = draw_xyz_axis(
                annotated_image, corners, ids, camera_matrix, dist_coeffs
            )
    else:
        print("No markers detected")
    
    return annotated_image, (corners, ids, rejected)


def display_multi_camera_grid(frames, aruco_dict_name='DICT_6X6_250'):
    """
    Display ArUco detection results from multiple cameras in a 1xN grid.
    
    Args:
        frames: List of RGB frames from cameras
        aruco_dict_name: Name of the ArUco dictionary to use
    """
    annotated_frames = []
    
    print("\n=== ArUco Detection ===")
    for i, frame in enumerate(frames):
        if frame is None:
            print(f"Camera {i + 1}: Skipping (no frame)")
            continue
        
        print(f"\nCamera {i + 1}:")
        annotated_frame, detections = detect_aruco_markers(frame, aruco_dict_name)
        annotated_frames.append(annotated_frame)
    
    # Ensure we have frames to display
    if not annotated_frames:
        print("No frames available to display!")
        return
    
    # Pad with black frames if fewer than 3 cameras
    while len(annotated_frames) < 3:
        h, w = annotated_frames[0].shape[:2] if annotated_frames else (720, 1280)
        annotated_frames.append(np.zeros((h, w, 3), dtype=np.uint8))
    
    # Concatenate frames horizontally (1x3 grid)
    grid_image = np.hstack(annotated_frames[:3])
    
    # Add camera labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    font_thickness = 2
    text_color = (0, 255, 0)
    
    frame_width = annotated_frames[0].shape[1]
    for i in range(3):
        x_offset = i * frame_width + 20
        cv2.putText(
            grid_image,
            f"Camera {i + 1}",
            (x_offset, 40),
            font,
            font_scale,
            text_color,
            font_thickness
        )
    
    # Display the grid
    cv2.imshow("Multi-Camera ArUco Detection (1x3)", grid_image)
    print("\nPress any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_frames_from_manager(camera_manager):
    """
    Capture RGB frames from CameraManager and convert to list format.
    
    Args:
        camera_manager: CameraManager instance
        
    Returns:
        list: List of RGB frames (numpy arrays)
    """
    frame_dict = camera_manager.get_frames(timeout_ms=5000)
    
    # Convert dict to ordered list of frames
    frames = []
    for key in sorted(frame_dict.keys()):
        frame_data = frame_dict[key]
        rgb_image = frame_data['image']
        # Convert from RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        frames.append(bgr_image)
        print(f"Frame captured from {frame_data['name']} (Serial: {frame_data['serial']})")
    
    return frames


def main():
    """Main function to run multi-camera ArUco detection."""
    print("=== RealSense Multi-Camera ArUco Detection ===\n")
    
    # Initialize camera manager
    camera_manager = CameraManager(width=1280, height=720, fps=30)
    cameras = camera_manager.discover_and_start()
    
    if not cameras:
        print("No cameras available. Exiting.")
        return
    
    print(f"Started {len(cameras)} camera(s)\n")
    
    try:
        # Give sensors time to warm up
        print("Warming up cameras...")
        for i in range(5):
            camera_manager.get_frames(timeout_ms=5000)
            print(f"  Warmup {i + 1}/5...")
        
        # Capture and process frames
        print("\nCapturing frames...")
        frames = get_frames_from_manager(camera_manager)
        
        if frames:
            # Display results
            display_multi_camera_grid(frames, aruco_dict_name='DICT_6X6_250')
        else:
            print("No frames captured!")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        camera_manager.stop_all()
        print("\nAll cameras stopped.")


if __name__ == "__main__":
    main()

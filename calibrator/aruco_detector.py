"""
aruco_detector.py

ArUco marker detection utilities. Provides detection, tracking, and visualization
of ArUco markers in images.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


class ArucoDetector:
    """
    Detect and analyze ArUco markers in images.
    
    Usage:
        detector = ArucoDetector()
        corners, ids, rvecs, tvecs = detector.detect(image)
        image_with_markers = detector.draw_markers(image, corners, ids)
    """
    
    def __init__(self, aruco_dict_name: str = "DICT_6X6_250"):
        """
        Initialize ArUco detector.
        
        Args:
            aruco_dict_name: Name of the ArUco dictionary to use.
                           Options: "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250",
                                   "DICT_4X4_1000", "DICT_5X5_50", "DICT_5X5_100",
                                   "DICT_5X5_250", "DICT_5X5_1000", "DICT_6X6_50",
                                   "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
                                   "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250",
                                   "DICT_7X7_1000", "DICT_ARUCO_ORIGINAL"
        """
        # Get the ArUco dictionary
        aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, aruco_dict_name))
        self.aruco_dict = aruco_dict
        
        # Create detector parameters
        self.params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, self.params)
    
    def detect(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect ArUco markers in an image.
        
        Args:
            image: Input image (BGR or grayscale)
        
        Returns:
            Tuple containing:
            - corners: List of (4, 2) arrays for each marker corner positions
            - ids: Array of detected marker IDs
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        return corners, ids
    
    def estimate_pose(self, image: np.ndarray, corners: List[np.ndarray], 
                     camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                     marker_length: float = 0.05) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Estimate the pose (rotation and translation vectors) of detected markers.
        
        Args:
            image: Input image
            corners: List of marker corners from detect()
            camera_matrix: Camera intrinsic matrix (3x3)
            dist_coeffs: Camera distortion coefficients
            marker_length: Length of the marker side in meters (default: 0.05m = 5cm)
        
        Returns:
            Tuple containing:
            - rvecs: List of rotation vectors (one per marker)
            - tvecs: List of translation vectors (one per marker)
        """
        if corners is None or len(corners) == 0:
            return [], []
        
        rvecs = []
        tvecs = []
        
        for corner in corners:
            # Define 3D points of the marker (in marker coordinate system)
            # Marker is assumed to be centered at origin with size marker_length
            marker_points = np.array([
                [-marker_length/2, -marker_length/2, 0],
                [marker_length/2, -marker_length/2, 0],
                [marker_length/2, marker_length/2, 0],
                [-marker_length/2, marker_length/2, 0]
            ], dtype=np.float32)
            
            # Solve PnP
            success, rvec, tvec = cv2.solvePnP(
                marker_points,
                corner.reshape(4, 2),
                camera_matrix,
                dist_coeffs
            )
            
            if success:
                rvecs.append(rvec)
                tvecs.append(tvec)
        
        return rvecs, tvecs
    
    def draw_markers(self, image: np.ndarray, corners: Optional[List[np.ndarray]], 
                    ids: Optional[np.ndarray]) -> np.ndarray:
        """
        Draw detected ArUco markers on the image.
        
        Args:
            image: Input image
            corners: List of marker corners from detect()
            ids: Array of marker IDs from detect()
        
        Returns:
            Image with markers drawn
        """
        result = image.copy()
        
        if corners is None or len(corners) == 0:
            return result
        
        # Draw rectangles around markers
        for i, corner in enumerate(corners):
            corner_int = corner[0].astype(int)
            
            # Draw rectangle
            cv2.polylines(result, [corner_int], True, (0, 255, 0), 2)
            
            # Draw marker ID
            if ids is not None and i < len(ids):
                marker_id = ids[i][0]
                center = corner_int.mean(axis=0).astype(int)
                cv2.putText(result, f"ID: {marker_id}", tuple(center),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result
    
    def draw_markers_with_pose(self, image: np.ndarray, corners: Optional[List[np.ndarray]], 
                              ids: Optional[np.ndarray], rvecs: List[np.ndarray],
                              tvecs: List[np.ndarray], camera_matrix: np.ndarray,
                              dist_coeffs: np.ndarray, axis_length: float = 0.05) -> np.ndarray:
        """
        Draw detected ArUco markers and their pose (coordinate axes) on the image.
        
        Args:
            image: Input image
            corners: List of marker corners from detect()
            ids: Array of marker IDs from detect()
            rvecs: List of rotation vectors
            tvecs: List of translation vectors
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Camera distortion coefficients
            axis_length: Length of the axis to draw (default: 0.05m)
        
        Returns:
            Image with markers and axes drawn
        """
        result = self.draw_markers(image, corners, ids)
        
        if len(rvecs) == 0 or len(tvecs) == 0:
            return result
        
        # Define 3D axes points
        axis_points = np.float32([
            [0, 0, 0],                    # Origin (red)
            [axis_length, 0, 0],          # X axis (red)
            [0, axis_length, 0],          # Y axis (green)
            [0, 0, axis_length]           # Z axis (blue)
        ])
        
        # Draw axes for each marker
        for rvec, tvec in zip(rvecs, tvecs):
            # Project 3D points to image
            img_points, _ = cv2.projectPoints(
                axis_points, rvec, tvec, camera_matrix, dist_coeffs
            )
            img_points = img_points.astype(int)
            
            origin = tuple(img_points[0][0])
            
            # Draw axes
            # X axis (red)
            result = cv2.line(result, origin, tuple(img_points[1][0]), (0, 0, 255), 3)
            # Y axis (green)
            result = cv2.line(result, origin, tuple(img_points[2][0]), (0, 255, 0), 3)
            # Z axis (blue)
            result = cv2.line(result, origin, tuple(img_points[3][0]), (255, 0, 0), 3)
        
        return result
    
    def get_marker_info(self, corners: Optional[List[np.ndarray]], 
                       ids: Optional[np.ndarray]) -> Dict:
        """
        Get information about detected markers.
        
        Args:
            corners: List of marker corners from detect()
            ids: Array of marker IDs from detect()
        
        Returns:
            Dictionary with marker information
        """
        info = {
            'num_markers': 0,
            'marker_ids': [],
            'marker_corners': []
        }
        
        if ids is not None:
            info['num_markers'] = len(ids)
            info['marker_ids'] = ids.flatten().tolist()
        
        if corners is not None:
            info['marker_corners'] = [corner[0].tolist() for corner in corners]
        
        return info

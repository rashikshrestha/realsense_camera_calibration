import argparse
from pathlib import Path
from caliscope.api import (
    Charuco, CharucoTracker, CameraArray,
    CaptureVolume, extract_image_points, extract_image_points_multicam,
    calibrate_intrinsics,
)
from caliscope.reporting import (
    print_intrinsic_report, print_extrinsic_report, print_camera_pair_coverage,
)
from get_intrinsics import get_camera_intrinsics, save_intrinsics_to_yaml


def main(workspace_dir: str):
    workspace_dir = Path(workspace_dir)
    
    # Define Calibration Target
    charuco = Charuco.from_squares(columns=4, rows=5, square_size_cm=5.0)
    tracker = CharucoTracker(charuco)
   
    # Get Intrinsics 
    intrinsics_dir = workspace_dir / "intrinsic"
    
    try:
        intrinsics_data = get_camera_intrinsics()
        save_intrinsics_to_yaml(intrinsics_data, intrinsics_dir)
    except Exception as e:
        print(f"Failed to extract intrinsics: {e}")
        return
   
    # Compute Extrinsics 
    extrinsic_videos = {0: workspace_dir / "extrinsic/cam_0.mp4", 1: workspace_dir / "extrinsic/cam_1.mp4"}
    
    try:
        ext_points = extract_image_points_multicam(
            extrinsic_videos, 
            tracker, 
            timestamps=str(workspace_dir / "timestamps.csv")
        )
    except Exception as e:
        print(f"Failed to extract extrinsic calibration points: {e}")
        return
    
    print("\nExtrinsic Calibration Report:")
    print_extrinsic_report(ext_points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=str)
    args = parser.parse_args()
    main(workspace_dir=args.workspace)

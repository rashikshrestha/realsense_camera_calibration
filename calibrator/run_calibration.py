from pathlib import Path
from caliscope.api import (
    Charuco, CharucoTracker, CameraArray,
    CaptureVolume, extract_image_points, extract_image_points_multicam,
    calibrate_intrinsics,
)
from caliscope.reporting import (
    print_intrinsic_report, print_extrinsic_report, print_camera_pair_coverage,
)

# Define Calibration Target
charuco = Charuco.from_squares(columns=4, rows=5, square_size_cm=5.0)
tracker = CharucoTracker(charuco)

# Extract time-aligned points for extrinsic calibration
extrinsic_videos = {0: Path("extrinsic/cam_0.mp4"), 1: Path("extrinsic/cam_1.mp4")}
ext_points = extract_image_points_multicam(
    extrinsic_videos, 
    tracker, 
    timestamps="timestamps.csv"
)

print("Extrinsic Calibration Report:")
print_extrinsic_report(ext_points)

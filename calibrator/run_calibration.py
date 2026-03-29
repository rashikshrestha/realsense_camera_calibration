import argparse
import numpy as np
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
   
    # ----- Get Intrinsics -----
    intrinsics_dir = workspace_dir / "intrinsic_config"
    
    try:
        intrinsics_data = get_camera_intrinsics()
        save_intrinsics_to_yaml(intrinsics_data, intrinsics_dir)
    except Exception as e:
        print(f"Failed to extract intrinsics: {e}")
        return
    print("Saved intrinsics to YAML files in:", intrinsics_dir)
    
    # ----- Calibration Target -----
    charuco = Charuco.from_squares(
        columns=4, rows=5, 
        square_size_cm=5.0, 
        inverted=True
    )
    tracker = CharucoTracker(charuco) 
    
    # ----- Compute Extrinsics -----
    # Get the video files with timestamps
    extrinsic_dir = workspace_dir / "extrinsic"
    
    extrinsic_videos = {}
    for video_file in sorted(extrinsic_dir.glob("cam_*.mp4")):
        cam_id = int(video_file.stem.split("_")[1])
        extrinsic_videos[cam_id] = video_file
    
    timestamps_file = extrinsic_dir / "timestamps.csv"
    cameras = CameraArray.from_video_metadata(extrinsic_videos)
    print(cameras) 
    
    # Load camera mapping and intrinsics
    import yaml
    
    camera_mapping_file = workspace_dir / "camera_mapping.yaml"
    with open(camera_mapping_file, 'r') as f:
        camera_mapping = yaml.safe_load(f)
    
    # Build cameras dict with intrinsics data
    for cam_key, cam_info in camera_mapping.items():
        cam_id = int(cam_key.split("_")[1])
        serial = cam_info['serial']
        intrinsics_file = intrinsics_dir / f"{serial}.yaml"
        
        with open(intrinsics_file, 'r') as f:
            intrinsics_data = yaml.safe_load(f)
        
        color_stream = intrinsics_data['streams']['color']
        cameras[cam_id].matrix = np.array([
                [color_stream['fx'], 0, color_stream['ppx']],
                [0, color_stream['fy'], color_stream['ppy']],
                [0, 0, 1]
            ])
        cameras[cam_id].distortions = np.array(color_stream['coefficients'])

    print(cameras)
    
    # Extract Multicam Image Points
    try:
        ext_points = extract_image_points_multicam(
            extrinsic_videos, 
            tracker, 
            timestamps=str(timestamps_file)
        )
    except Exception as e:
        print(f"Failed to extract extrinsic calibration points: {e}")
        return
    
    # Bootstrap and Optimize
    volume = CaptureVolume.bootstrap(ext_points, cameras)
    volume = volume.optimize(strict=False)
    
    # Filter outliers and re-optimize
    volume = volume.filter_by_percentile_error(2.5)
    volume = volume.optimize(strict=False)
    
    # Save final caliscope results
    volume.save(workspace_dir / "extrinsic/capture_volume")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--workspace", type=str, required=True, help="Workspace directory")
    args = parser.parse_args()
    main(workspace_dir=args.workspace)

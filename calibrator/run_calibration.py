import argparse
import numpy as np
import yaml
import toml
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
from generate_final_extrinsics import build_extrinsics_from_toml, write_extrinsics_yaml

BLUE = '\033[94m'
RESET = '\033[0m'

def print_section(title):
    """Print a colored section header"""
    print(f"\n{BLUE}---------- {title} ----------{RESET}")


def main(workspace_dir: str):
    workspace_dir = Path(workspace_dir)
   
    print_section("Get Intrinsics")
    intrinsics_dir = workspace_dir / "intrinsic_config"
    
    try:
        intrinsics_data = get_camera_intrinsics()
        save_intrinsics_to_yaml(intrinsics_data, intrinsics_dir)
    except Exception as e:
        print(f"Failed to extract intrinsics: {e}")
        return
    print("Saved intrinsics to YAML files in:", intrinsics_dir)
    
    print_section("Calibration Target")
    charuco = Charuco.from_squares(
        columns=4, rows=6, 
        square_size_cm=6.505, 
        inverted=False
    )
    tracker = CharucoTracker(charuco) 
    
    print_section("Load Video Data")
    extrinsic_dir = workspace_dir / "extrinsic"
    
    extrinsic_videos = {}
    for video_file in sorted(extrinsic_dir.glob("cam_*.mp4")):
        cam_id = int(video_file.stem.split("_")[1])
        extrinsic_videos[cam_id] = video_file
    
    timestamps_file = extrinsic_dir / "timestamps.csv"
    cameras = CameraArray.from_video_metadata(extrinsic_videos)
    
    print_section("Load Camera Intrinsics")
    
    camera_mapping_file = workspace_dir / "camera_mapping.yaml"
    with open(camera_mapping_file, 'r') as f:
        camera_mapping = yaml.safe_load(f)
    
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
    print("Loaded camera mapping from:", camera_mapping_file)
    print("Loaded intrinsics for:")
    for cam_id, cam_info in camera_mapping.items():
        print(f"  Camera ID: {cam_id}, Serial: {cam_info['serial']}, Name: {cam_info['name']}") 

    print_section("Extract Image Points")
    try:
        ext_points = extract_image_points_multicam(
            extrinsic_videos, 
            tracker, 
            timestamps=str(timestamps_file)
        )
    except Exception as e:
        print(f"Failed to extract extrinsic calibration points: {e}")
        return
    
    print_section("Bootstrap and Optimize")
    volume = CaptureVolume.bootstrap(ext_points, cameras)
    volume = volume.optimize(strict=False)
    
    print_section("Filter Outliers")
    volume = volume.filter_by_percentile_error(10.0)
    volume = volume.optimize(strict=False)
    
    print_section("Save Results")
    volume.save(workspace_dir / "extrinsic/capture_volume")
    print("Calibration results saved to:", workspace_dir / "extrinsic/capture_volume")
    
    print_section("Calculate Camera Distances")
    camera_array_file = workspace_dir / "extrinsic/capture_volume/camera_array.toml"
    with open(camera_array_file, 'r') as f:
        camera_array = toml.load(f)
    
    cameras_list = []
    for cam_key, cam_data in camera_array['cameras'].items():
        cam_id = int(cam_key)
        translation = np.array(cam_data['translation'])
        cameras_list.append((cam_id, translation))
    
    cameras_list.sort(key=lambda x: x[0])
    
    print(f"\nCamera Distances (Euclidean):")
    distance_info = []
    for i in range(len(cameras_list)):
        for j in range(i + 1, len(cameras_list)):
            cam_id_1, trans_1 = cameras_list[i]
            cam_id_2, trans_2 = cameras_list[j]
            distance = np.linalg.norm(trans_1 - trans_2)
            print(f"  Camera {cam_id_1} <-> Camera {cam_id_2}: {distance:.6f} m")
            distance_info.append({
                'camera_1': cam_id_1,
                'camera_2': cam_id_2,
                'distance_m': distance
            })
    
    # Create distance matrix
    n_cameras = len(cameras_list)
    distance_matrix = np.zeros((n_cameras, n_cameras))
    
    for i in range(n_cameras):
        for j in range(n_cameras):
            if i == j:
                distance_matrix[i, j] = 0.0
            elif i < j:
                cam_id_1, trans_1 = cameras_list[i]
                cam_id_2, trans_2 = cameras_list[j]
                distance = np.linalg.norm(trans_1 - trans_2)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
    
    # Save distance matrix to file
    distances_file = workspace_dir / "camera_distance_matrix.txt"
    np.savetxt(distances_file, distance_matrix, fmt='%.6f')
    print(f"Saved camera distance matrix to: {distances_file}")
    
    print_section("Generate Final Extrinsics")
    
    extrinsics = build_extrinsics_from_toml(camera_array)
    
    # Add serial numbers to extrinsics
    for extrinsic in extrinsics:
        cam_id = extrinsic.get("cam_id")
        cam_key = f"cam_{cam_id}"
        if cam_key in camera_mapping:
            extrinsic["serial"] = camera_mapping[cam_key]["serial"]
            extrinsic["name"] = camera_mapping[cam_key]["name"]
    
    output_extrinsics_file = workspace_dir / "cam_config.yaml"
    write_extrinsics_yaml(extrinsics, output_extrinsics_file)
    print(f"Generated extrinsics file: {output_extrinsics_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--workspace", type=str, required=True, help="Workspace directory")
    args = parser.parse_args()
    main(workspace_dir=args.workspace)

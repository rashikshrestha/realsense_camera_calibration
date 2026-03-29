"""
get_intrinsics.py

Extract camera intrinsics from connected RealSense cameras and save them to a YAML file.
"""

import pyrealsense2 as rs
import yaml
from pathlib import Path
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_camera_intrinsics() -> Dict[str, Any]:
    """
    Discover all connected RealSense cameras and extract their intrinsic parameters.
    
    Returns:
        A dictionary with camera serial numbers as keys and intrinsic data as values.
        Each intrinsic data includes:
        - name: Camera model name
        - serial: Serial number
        - streams: Dict of stream names with their intrinsic parameters
    """
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        logger.warning("No RealSense devices found!")
        return {}
    
    logger.info(f"Found {len(devices)} RealSense device(s)")
    
    intrinsics_data = {}
    
    for dev in devices:
        try:
            serial = dev.get_info(rs.camera_info.serial_number)
        except Exception as e:
            logger.warning(f"Could not get serial number: {e}")
            serial = "unknown"
        
        try:
            name = dev.get_info(rs.camera_info.name)
        except Exception as e:
            logger.warning(f"Could not get camera name: {e}")
            name = "Unknown"
        
        logger.info(f"Processing camera: {name} (Serial: {serial})")
        
        # Start a pipeline to get intrinsics
        pipeline = rs.pipeline()
        config = rs.config()
        
        if serial != "unknown":
            config.enable_device(serial)
        
        # Enable color and depth streams to get their intrinsics
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        try:
            profile = pipeline.start(config)
        except Exception as e:
            logger.error(f"Failed to start pipeline for {serial}: {e}")
            continue
        
        try:
            camera_data = {
                'name': name,
                'serial': serial,
                'streams': {}
            }
            
            # Get intrinsics for color and depth streams
            for stream_type, stream_enum in [('color', rs.stream.color), ('depth', rs.stream.depth)]:
                try:
                    stream_profile = profile.get_stream(stream_enum)
                    if stream_profile:
                        intrinsics = stream_profile.as_video_stream_profile().get_intrinsics()
                        
                        camera_data['streams'][stream_type] = {
                            'width': intrinsics.width,
                            'height': intrinsics.height,
                            'ppx': float(intrinsics.ppx),  # Principal point x
                            'ppy': float(intrinsics.ppy),  # Principal point y
                            'fx': float(intrinsics.fx),    # Focal length x
                            'fy': float(intrinsics.fy),    # Focal length y
                            'coefficients': [float(c) for c in intrinsics.coeffs]
                        }
                        logger.info(f"  {stream_type}: {intrinsics.width}x{intrinsics.height}, "
                                  f"fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
                except Exception as e:
                    logger.warning(f"  Could not get {stream_type} intrinsics: {e}")
            
            intrinsics_data[serial] = camera_data
            
        finally:
            pipeline.stop()
    
    return intrinsics_data


def save_intrinsics_to_yaml(intrinsics_data: Dict[str, Any], output_file: Path) -> None:
    """
    Save camera intrinsics to a YAML file.
    
    Args:
        intrinsics_data: Dictionary of intrinsics data from get_camera_intrinsics()
        output_file: Path to the output YAML file
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        yaml.dump(intrinsics_data, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Intrinsics saved to {output_file}")


def main():
    """Main function to extract and save camera intrinsics."""
    # Get intrinsics from all connected cameras
    intrinsics_data = get_camera_intrinsics()
    
    if not intrinsics_data:
        logger.error("No intrinsics data collected. Please check your RealSense connections.")
        return
    
    # Save to YAML file
    output_file = Path(__file__).parent.parent / 'camera_intrinsics.yaml'
    save_intrinsics_to_yaml(intrinsics_data, output_file)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("INTRINSICS EXTRACTION SUMMARY")
    logger.info("="*60)
    for serial, data in intrinsics_data.items():
        logger.info(f"\nCamera: {data['name']} (Serial: {serial})")
        for stream_name, stream_data in data['streams'].items():
            logger.info(f"  {stream_name.upper()}:")
            logger.info(f"    Resolution: {stream_data['width']}x{stream_data['height']}")
            logger.info(f"    Principal Point (cx, cy): ({stream_data['ppx']:.2f}, {stream_data['ppy']:.2f})")
            logger.info(f"    Focal Length (fx, fy): ({stream_data['fx']:.2f}, {stream_data['fy']:.2f})")
            logger.info(f"    Distortion Coefficients: {stream_data['coefficients']}")
        logger.info("="*60)


if __name__ == "__main__":
    main()

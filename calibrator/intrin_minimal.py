# In Python using pyrealsense2
import pyrealsense2 as rs

# 1. Create a context object
ctx = rs.context()

# 2. Enumerate devices
devices = ctx.query_devices()
if len(devices) == 0:
    print("No RealSense device found")
else:
    # Get the first device
    device = devices[0]
    print(f"Using device: {device.get_info(rs.camera_info.name)}")

    # 3. Enumerate sensors
    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            # Check if the profile is a video stream profile and active
            if profile.is_video_stream_profile():
                video_profile = profile.as_video_stream_profile()
                # Get intrinsics for the specific stream (e.g., depth, color, infrared)
                try:
                    intrinsics = video_profile.get_intrinsics()
                    print(f"Intrinsics for {profile.stream_name()} stream:")
                    print(f"  Width: {intrinsics.width}, Height: {intrinsics.height}")
                    print(f"  FX: {intrinsics.fx}, FY: {intrinsics.fy}")
                    print(f"  CX: {intrinsics.ppx}, CY: {intrinsics.ppy}")
                    print(f"  Distortion Model: {intrinsics.model}")
                    print(f"  Distortion Coeffs: {intrinsics.coeffs}")
                except Exception as e:
                    # Some stream types (like Y16 infrared used for calibration) may not expose intrinsics
                    # through the standard API
                    pass

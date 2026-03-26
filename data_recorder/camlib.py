"""
camlib.py

Camera management utilities for Intel RealSense devices.
Provides a small CameraManager which discovers connected devices,
starts one pipeline per device, and can fetch the latest color frames
from each camera.

This module keeps responsibilities small and testable so the UI code
can remain focused on presentation.
"""

import pyrealsense2 as rs
import numpy as np
from typing import Dict, List, Optional


class CameraManager:
    """
    Discover and manage multiple RealSense cameras.

    Usage:
        mgr = CameraManager(width=640, height=480, fps=30)
        mgr.discover_and_start()
        frames = mgr.get_frames()
        mgr.stop_all()
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.cameras: List[Dict] = []  # each entry: {'serial','name','pipeline'}

    def discover_and_start(self) -> List[Dict]:
        """
        Discover connected RealSense devices and start a pipeline for each.
        Returns a list of camera dictionaries describing started cameras.
        """
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            return []

        for dev in devices:
            try:
                serial = dev.get_info(rs.camera_info.serial_number)
            except Exception:
                serial = None
            try:
                name = dev.get_info(rs.camera_info.name)
            except Exception:
                name = 'Unknown'

            pipeline = rs.pipeline()
            cfg = rs.config()
            if serial:
                cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)

            try:
                pipeline.start(cfg)
                self.cameras.append({'serial': serial, 'name': name, 'pipeline': pipeline})
            except Exception:
                # If one camera fails to start, skip it and continue with others
                continue

        return self.cameras

    def stop_all(self) -> None:
        """Stop all running pipelines managed by this manager."""
        for cam in self.cameras:
            try:
                cam['pipeline'].stop()
            except Exception:
                pass

    def get_frames(self, timeout_ms: int = 200) -> Dict[str, Dict]:
        """
        Poll each camera for a color frame and return a mapping keyed by
        camera serial (or name if serial is None). Each value contains
        'name', 'serial', 'image' (a numpy array in RGB order),
        'timestamp' (frame timestamp in seconds) and 'frame_number'.
        """
        results: Dict[str, Dict] = {}
        for cam in self.cameras:
            key = cam['serial'] if cam['serial'] is not None else cam['name']
            try:
                frames = cam['pipeline'].wait_for_frames(timeout_ms=timeout_ms)
                color_frame = frames.get_color_frame()
                if color_frame:
                    arr = np.asanyarray(color_frame.get_data())
                    # RealSense frame timestamp (seconds) and frame number
                    try:
                        # pyrealsense2 reports timestamps in milliseconds; convert to seconds
                        ts = float(color_frame.get_timestamp()) / 1000.0
                    except Exception:
                        ts = None
                    try:
                        fn = int(color_frame.get_frame_number())
                    except Exception:
                        fn = None
                    results[key] = {
                        'name': cam['name'],
                        'serial': cam['serial'],
                        'image': arr,
                        'timestamp': ts,
                        'frame_number': fn,
                    }
            except Exception:
                # timeout or device hiccup -> skip this camera for now
                continue
        return results

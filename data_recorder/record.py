# Simple RealSense D455 viewer
# filepath: /home/rashik/ws/camcal/record_simple.py

"""
Simple viewer for Intel RealSense D455 color stream.
- Resolution: 1280x720
- FPS: 30
- Format: RGB8

Run: python record_simple.py
Quit: press 'q' in the window or Ctrl+C in the terminal
"""

import signal
import sys
import time
import os

import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk

try:
    import pyrealsense2 as rs
except Exception:
    print("Error: pyrealsense2 is required. Install with 'pip install pyrealsense2'.")
    raise

from camlib import CameraManager
from ui import CameraGridUI

STOP = False

def _sigint_handler(sig, frame):
    global STOP
    STOP = True

signal.signal(signal.SIGINT, _sigint_handler)


def main():
    width, height, fps = 640, 480, 30

    # Create camera manager and start cameras
    mgr = CameraManager(width=width, height=height, fps=fps)
    cams = mgr.discover_and_start()
    if len(cams) == 0:
        print("No RealSense cameras found or failed to start.")
        return

    # Build UI
    root = tk.Tk()
    root.title('RealSense Cameras')
    
    # add a control frame above the camera grid
    control_frame = tk.Frame(root)
    control_frame.grid(row=0, column=0, columnspan=4, pady=4)
    record_button = tk.Button(control_frame, text='Start Recording')
    status_label = tk.Label(control_frame, text='Not recording')
    record_button.pack(side=tk.LEFT, padx=4)
    status_label.pack(side=tk.LEFT, padx=8)

    ui = CameraGridUI(root, cams, fps=fps, start_row=1)

    # Recording state
    recording = {'active': False, 'writers': {}, 'id_map': {}, 'csv_file': None}

    # assign simple integer camera ids based on list order
    for idx, cam in enumerate(cams):
        key = cam['serial'] if cam['serial'] is not None else cam['name']
        recording['id_map'][key] = idx

    def start_recording():
        if recording['active']:
            return
        # prepare output directory
        os.makedirs('recordings', exist_ok=True)
        # prepare VideoWriter for each camera that has recording enabled
        for cam in cams:
            key = cam['serial'] if cam['serial'] is not None else cam['name']
            # Check if this camera's recording is enabled
            if not ui.recording_enabled[key].get():
                print(f"Skipping {key} (recording disabled)")
                continue
            cam_id = recording['id_map'][key]
            fname = os.path.join('recordings', f'cam_{cam_id}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # writer will accept BGR frames; use capture resolution
            writer = cv2.VideoWriter(fname, fourcc, fps, (width, height))
            if writer.isOpened():
                recording['writers'][key] = {'writer': writer, 'path': fname}
                print(f"Recording {key} -> {fname}")
        # open timestamps.csv and write header
        csv_path = os.path.join('recordings', 'timestamps.csv')
        recording['csv_file'] = open(csv_path, 'w', buffering=1)
        recording['csv_file'].write('cam_id,frame_time\n')

        recording['active'] = True
        record_button.config(text='Stop Recording')
        status_label.config(text='Recording')

    def stop_recording():
        if not recording['active']:
            return
        for key, w in recording['writers'].items():
            try:
                w['writer'].release()
                print(f"Saved recording: {w['path']}")
            except Exception:
                pass
        recording['writers'].clear()
        # close csv file
        try:
            if recording['csv_file'] is not None:
                recording['csv_file'].close()
        except Exception:
            pass
        recording['csv_file'] = None
        recording['active'] = False
        record_button.config(text='Start Recording')
        status_label.config(text='Not recording')

    def toggle_recording():
        if recording['active']:
            stop_recording()
        else:
            start_recording()

    record_button.config(command=toggle_recording)

    def on_close():
        global STOP
        STOP = True
        mgr.stop_all()
        try:
            root.destroy()
        except Exception:
            pass

    root.protocol('WM_DELETE_WINDOW', on_close)
    root.bind('q', lambda e: on_close())

    def update_loop():
        if STOP:
            return
        frames = mgr.get_frames()
        ui.update(frames)
        # if recording, write each frame to its VideoWriter
        if recording['active']:
            for key, info in frames.items():
                w = recording['writers'].get(key)
                if not w:
                    continue
                # write image after converting RGB->BGR and resizing to capture size
                img = info['image']
                try:
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    if (bgr.shape[1], bgr.shape[0]) != (width, height):
                        bgr = cv2.resize(bgr, (width, height))
                    w['writer'].write(bgr)
                    # log timestamp to CSV if available
                    ts = info.get('timestamp')
                    if ts is not None and recording.get('csv_file') is not None:
                        cam_id = recording['id_map'].get(key, '')
                        try:
                            recording['csv_file'].write(f"{cam_id},{ts}\n")
                        except Exception:
                            pass
                except Exception:
                    continue
        root.after(int(1000 / fps), update_loop)

    root.after(0, update_loop)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass

    mgr.stop_all()


if __name__ == '__main__':
    main()

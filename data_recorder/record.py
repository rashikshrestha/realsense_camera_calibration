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
import argparse

import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import yaml

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


def main(workspace_dir):
    # width, height, fps = 640, 480, 30
    width, height, fps = 1280, 720, 30

    # Create camera manager and start cameras
    mgr = CameraManager(width=width, height=height, fps=fps)
    cams = mgr.discover_and_start()
    if len(cams) == 0:
        print("No RealSense cameras found or failed to start.")
        return

    # Build UI
    root = tk.Tk()
    root.title('RealSense Recorder')
    
    # Output directory state
    output_dir = {'path': os.path.join(workspace_dir, 'extrinsic')}
    
    # add a control frame above the camera grid
    control_frame = tk.Frame(root)
    control_frame.grid(row=0, column=0, columnspan=4, pady=4, sticky='w')
    
    # Directory selection section (top-left)
    dir_frame = tk.Frame(control_frame)
    dir_frame.pack(side=tk.LEFT, padx=4)
    dir_label = tk.Label(dir_frame, text='Output Dir:')
    dir_label.pack(side=tk.LEFT, padx=2)
    dir_display = tk.Label(dir_frame, text=output_dir['path'], fg='blue', relief=tk.SUNKEN, width=20)
    dir_display.pack(side=tk.LEFT, padx=2)
    
    def choose_directory():
        selected_dir = filedialog.askdirectory(title='Choose Output Directory')
        if selected_dir:
            output_dir['path'] = selected_dir
            dir_display.config(text=output_dir['path'])
            print(f"Output directory set to: {output_dir['path']}")
    
    dir_button = tk.Button(dir_frame, text='Browse...', command=choose_directory)
    dir_button.pack(side=tk.LEFT, padx=2)
    
    # Timestamp CSV save option
    save_timestamps_var = tk.BooleanVar(value=True)
    timestamps_frame = tk.Frame(control_frame)
    timestamps_frame.pack(side=tk.LEFT, padx=4)
    timestamps_checkbox = tk.Checkbutton(timestamps_frame, text='Save Timestamp', variable=save_timestamps_var)
    timestamps_checkbox.pack(side=tk.LEFT)
    
    # Recording controls
    record_button = tk.Button(control_frame, text='Start Recording')
    status_label = tk.Label(control_frame, text='Not recording')
    record_button.pack(side=tk.LEFT, padx=4)
    status_label.pack(side=tk.LEFT, padx=8)

    ui = CameraGridUI(root, cams, fps=fps, start_row=1)

    # Recording state
    recording = {'active': False, 'writers': {}, 'id_map': {}, 'csv_file': None}

    def start_recording():
        if recording['active']:
            return
        
        # Get custom camera IDs from UI before recording starts
        recording['id_map'] = ui.get_camera_ids()
        
        # prepare output directory
        os.makedirs(output_dir['path'], exist_ok=True)
        
        # Check if any output files already exist
        existing_files = []
        for cam in cams:
            key = cam['serial'] if cam['serial'] is not None else cam['name']
            # Check if this camera's recording is enabled
            if not ui.recording_enabled[key].get():
                continue
            cam_id = recording['id_map'].get(key, '0')
            fname = os.path.join(output_dir['path'], f'cam_{cam_id}.mp4')
            if os.path.exists(fname):
                existing_files.append(fname)
        
        # Check if timestamps CSV will be created and already exists
        if save_timestamps_var.get():
            csv_path = os.path.join(output_dir['path'], 'timestamps.csv')
            if os.path.exists(csv_path):
                existing_files.append(csv_path)
        
        # If files exist, ask user for confirmation
        if existing_files:
            file_list = '\n'.join(existing_files)
            response = messagebox.askyesno(
                'Files Already Exist',
                f'The following files already exist:\n\n{file_list}\n\nDo you want to overwrite them?'
            )
            if not response:
                print("Recording cancelled by user")
                return
            print("Overwriting existing files")
        
        # prepare VideoWriter for each camera that has recording enabled
        for cam in cams:
            key = cam['serial'] if cam['serial'] is not None else cam['name']
            # Check if this camera's recording is enabled
            if not ui.recording_enabled[key].get():
                print(f"Skipping {key} (recording disabled)")
                continue
            cam_id = recording['id_map'].get(key, '0')
            fname = os.path.join(output_dir['path'], f'cam_{cam_id}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # writer will accept BGR frames; use capture resolution
            writer = cv2.VideoWriter(fname, fourcc, fps, (width, height))
            if writer.isOpened():
                recording['writers'][key] = {'writer': writer, 'path': fname}
                print(f"Recording {key} -> {fname}")
        # open timestamps.csv and write header if enabled
        recording['csv_file'] = None
        if save_timestamps_var.get():
            csv_path = os.path.join(output_dir['path'], 'timestamps.csv')
            recording['csv_file'] = open(csv_path, 'w', buffering=1)
            recording['csv_file'].write('cam_id,frame_time\n')
            print(f"Saving timestamps to: {csv_path}")
        else:
            print("Timestamps CSV save is disabled")

        # Save camera mapping (camera index to serial number) as YAML
        camera_mapping = {}
        for cam in cams:
            key = cam['serial'] if cam['serial'] is not None else cam['name']
            # Only include cameras that are recording
            if key in recording['writers']:
                cam_id = recording['id_map'].get(key, '0')
                camera_mapping[f'cam_{cam_id}'] = {
                    'serial': cam['serial'],
                    'name': cam['name']
                }
        
        if camera_mapping:
            yaml_path = os.path.join(workspace_dir, 'camera_mapping.yaml')
            with open(yaml_path, 'w') as f:
                yaml.dump(camera_mapping, f, default_flow_style=False)
            print(f"Saved camera mapping to: {yaml_path}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workspace', type=str, required=True, help='Workspace directory')
    args = parser.parse_args()
    main(workspace_dir=args.workspace)

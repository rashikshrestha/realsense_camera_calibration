"""
ui.py

Tkinter-based UI for showing multiple RealSense camera color streams.
This module depends on camlib.CameraManager for camera access and focuses
on layout and presentation logic.
"""

import tkinter as tk
from PIL import Image, ImageTk
from typing import Dict


class CameraGridUI:
    """Simple grid UI that shows a labeled image widget per camera.

    start_row: an optional grid row offset so callers can place controls
    (buttons, labels) above the camera grid.
    """

    def __init__(self, root: tk.Tk, cameras: list, fps: int = 30, start_row: int = 0):
        self.root = root
        self.cameras = cameras
        self.fps = fps
        self.start_row = int(start_row)
        self.widgets = {}
        # store max display sizes per camera key (width, height)
        self.max_sizes = {}
        # track recording state per camera (key -> BooleanVar)
        self.recording_enabled = {}
        # store custom camera IDs (key -> StringVar)
        self.camera_ids = {}
        # track camera keys for ID uniqueness
        self.camera_keys = []
        self._build_ui()

    def _build_ui(self):
        import math
        cols = int(math.ceil(math.sqrt(len(self.cameras))))
        # determine available screen size and compute a maximum size per cell
        # call update_idletasks to ensure root has correct geometry information
        try:
            self.root.update_idletasks()
            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
        except Exception:
            # fallback to reasonable defaults
            screen_w, screen_h = 1920, 1080

        rows = int(math.ceil(len(self.cameras) / cols)) if cols > 0 else 1
        # leave margins for window chrome, titles and info labels
        margin_w = 40 * cols
        margin_h = 120 * rows
        avail_w = max(100, screen_w - margin_w)
        avail_h = max(100, screen_h - margin_h)
        cell_w = max(100, avail_w // max(1, cols))
        cell_h = max(80, avail_h // max(1, rows))

        for idx, cam in enumerate(self.cameras):
            r = idx // cols
            c = idx % cols
            frame = tk.Frame(self.root, bd=2, relief=tk.RIDGE)
            # apply start_row offset so callers can reserve space above the grid
            frame.grid(row=r + self.start_row, column=c, padx=5, pady=5)
            
            # Create header with title and camera ID input
            header_frame = tk.Frame(frame)
            header_frame.pack(fill=tk.X, padx=2, pady=2)
            
            title = tk.Label(header_frame, text=f"{cam['name']}\n{cam['serial']}", justify=tk.LEFT)
            title.pack(side=tk.LEFT, padx=2)
            
            # Add camera ID input field
            key = cam['serial'] if cam['serial'] is not None else cam['name']
            cam_id_var = tk.StringVar(value=str(idx))
            self.camera_ids[key] = cam_id_var
            self.camera_keys.append(key)
            
            id_label = tk.Label(header_frame, text="ID:", font=(None, 8))
            id_label.pack(side=tk.LEFT, padx=2)
            id_entry = tk.Entry(header_frame, textvariable=cam_id_var, width=3, font=(None, 8))
            id_entry.pack(side=tk.LEFT, padx=2)
            
            # Bind callback to ensure unique camera IDs
            cam_id_var.trace('w', lambda name, index, mode, k=key: self._on_camera_id_change(k))
            
            # Add checkbox for recording enable/disable
            record_var = tk.BooleanVar(value=True)
            self.recording_enabled[key] = record_var
            checkbox = tk.Checkbutton(header_frame, text="Record", variable=record_var)
            checkbox.pack(side=tk.LEFT, padx=2)
            
            lbl = tk.Label(frame)
            lbl.pack()
            info_lbl = tk.Label(frame, text="timestamp: -\nframe: -", font=(None, 8))
            info_lbl.pack()
            self.widgets[key] = {'image': lbl, 'info': info_lbl}
            # reserve a max display size for this camera's image (subtract title/info space)
            # keep some padding inside the cell
            max_w = max(80, cell_w - 20)
            max_h = max(60, cell_h - 40)
            self.max_sizes[key] = (max_w, max_h)

    def update(self, frames: Dict[str, Dict]):
        """Update the UI with the latest frames mapping returned by CameraManager.get_frames()."""
        for key, info in frames.items():
            img = info['image']  # RGB numpy array
            pil = Image.fromarray(img)
            # flip the image horizontally to create mirror image for easy viewing
            pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
            # resize for display if larger than allocated cell while preserving aspect ratio
            max_size = self.max_sizes.get(key)
            max_size = (640, 480) #! inforce the max size to avoid thumbnail issues with some PIL versions, can be removed later
            if max_size is not None:
                try:
                    pil.thumbnail(max_size, Image.LANCZOS)
                except Exception:
                    # fallback to ANTIALIAS for older PIL versions
                    try:
                        pil.thumbnail(max_size, Image.ANTIALIAS)
                    except Exception:
                        pass
            photo = ImageTk.PhotoImage(image=pil)
            widget = self.widgets.get(key)
            if widget:
                widget['image'].config(image=photo)
                widget['image'].image = photo
                # Update timestamp and frame number if present
                ts = info.get('timestamp')
                fn = info.get('frame_number')
                ts_text = f"timestamp: {ts:.3f} ms" if ts is not None else "timestamp: -"
                fn_text = f"frame: {fn}" if fn is not None else "frame: -"
                widget['info'].config(text=f"{ts_text}\n{fn_text}")

    def get_camera_ids(self) -> Dict[str, str]:
        """Get the custom camera IDs mapping from key to ID."""
        return {key: var.get() for key, var in self.camera_ids.items()}

    def _on_camera_id_change(self, changed_key: str):
        """Handle camera ID change with uniqueness constraint.
        
        When a camera ID is changed, if that ID already exists on another camera,
        swap the IDs between the two cameras.
        """
        new_id = self.camera_ids[changed_key].get()
        
        # Find if another camera already has this ID
        duplicate_key = None
        for key in self.camera_keys:
            if key != changed_key and self.camera_ids[key].get() == new_id:
                duplicate_key = key
                break
        
        if duplicate_key:
            # Swap IDs: give the old ID of changed_key to duplicate_key
            old_id = None
            # Find the old ID by checking what wasn't in use
            all_ids = {self.camera_ids[k].get() for k in self.camera_keys}
            used_ids = {self.camera_ids[k].get() for k in self.camera_keys if k != duplicate_key}
            
            # Get the ID that duplicate_key had before (we need to track this)
            # Actually, we need to find what ID is missing from the set
            # Get all possible IDs (0 to n-1)
            expected_ids = set(str(i) for i in range(len(self.camera_keys)))
            available_ids = expected_ids - used_ids
            
            if available_ids:
                old_id = available_ids.pop()
            else:
                # Fallback: find the ID that duplicate_key currently has
                old_id = self.camera_ids[duplicate_key].get()
            
            # Perform the swap
            self.camera_ids[duplicate_key].set(old_id)
            print(f"ID conflict resolved: {changed_key} -> {new_id}, {duplicate_key} -> {old_id}")

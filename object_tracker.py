import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import queue
from collections import deque # Added for a standard structure, though not strictly required here

# Load YOLOv8 segmentation model
model = YOLO('yolov8n-seg.pt')

# Globals
selected_track_id = None
last_seen_time = None
RESET_TIMEOUT = 5
masks_data = []

highlight_color = [0, 255, 0]  # Initial green color


video_sources = [0]  # Start with camera 0 by default
STANDARD_HEIGHT = 480 # Target height for all videos
command_queue = queue.Queue()

# --- NEW: Threading Globals ---
video_readers = [] # List to hold VideoReader objects
# NOTE: The global 'caps' list is no longer used for the main loop, but will be 
# temporarily updated by reload_captures if we need to fall back to the old style.

# Editing mode globals
edit_mode = False
drawing = False
erasing = False
brush_size = 10
edited_mask = None
paused_frame = None
current_mask_index = None

# Persistent storage for user edited masks: {track_id: {'mask': mask, 'center': (x, y)}}
edited_masks_dict = {}
# Store original masks for reset: {track_id: {'mask': mask, 'center': (x, y)}}
original_masks_dict = {}

# --- NEW: VideoReader Class for Parallel Frame Capture ---
class VideoReader:
    """A thread-safe class to continuously read frames from a video source."""
    def __init__(self, source, target_height):
        self.source = source
        self.target_height = target_height
        self.cap = None
        self.thread = None
        self.running = False
        self.frame = None # Holds the latest frame
        self.read_lock = threading.Lock()
        self.is_camera = isinstance(source, int)
        self.is_video_file = not self.is_camera
        self.is_opened = False

        self._open_capture()

    def _open_capture(self):
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            
        try:
            if self.is_camera:
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
            else:
                self.cap = cv2.VideoCapture(self.source)
            
            if self.cap.isOpened():
                self.is_opened = True
            else:
                print(f"Error: Could not open video source: {self.source}")

        except Exception as e:
            print(f"Failed to open {self.source}: {e}")

    def start(self):
        """Starts the reader thread."""
        if self.is_opened:
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        """Stops the reader thread and releases the capture."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=0.1) # Wait briefly for thread to finish
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False

    def _run(self):
        """The main loop for the reader thread."""
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                if self.is_video_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                
                if not ret:
                    time.sleep(0.01) # Avoid busy loop
                    continue
            
            # Resize frame once in the reader thread (reduces main thread load)
            try:
                h, w, _ = frame.shape
                scale = self.target_height / h
                new_w = int(w * scale)
                frame_resized = cv2.resize(frame, (new_w, self.target_height), interpolation=cv2.INTER_AREA)
            except Exception:
                time.sleep(0.01) 
                continue

            with self.read_lock:
                self.frame = frame_resized
            
            if self.is_video_file:
                # Add a small delay for files to prevent maxing out CPU/disk
                time.sleep(0.001) 
            else:
                # Read as fast as camera allows
                pass 


    def get_latest_frame(self):
        """Returns the latest frame in a thread-safe manner."""
        with self.read_lock:
            # Use 'if self.frame is not None' to avoid error on startup
            frame_copy = self.frame.copy() if self.frame is not None else None 
        return frame_copy

# --- MODIFIED: reload_captures now manages VideoReader objects ---
def reload_captures():
    """
    Closes all current VideoReader threads/objects and reloads them 
    based on the video_sources list. MUST be called from the main thread.
    """
    global video_readers, video_sources
    
    # 1. Stop and clear all existing readers
    for reader in video_readers:
        reader.stop()
    video_readers.clear()

    valid_sources = []
    
    # 2. Create new readers for all sources and start their threads
    for source in video_sources:
        reader = VideoReader(source, STANDARD_HEIGHT)
        if reader.is_opened:
            reader.start()
            video_readers.append(reader)
            valid_sources.append(source)
        else:
            print(f"Skipping invalid source: {source}")
    
    # Update video_sources to only include valid, opened sources
    video_sources.clear()
    video_sources.extend(valid_sources)

    # 3. Fallback if no sources were valid (This part is crucial)
    if not video_readers:
        print("No valid video sources. Trying default camera 0...")
        video_sources.clear()
        video_sources.append(0)
        reader = VideoReader(0, STANDARD_HEIGHT)
        if reader.is_opened:
            reader.start()
            video_readers.append(reader)
            video_sources.clear()
            video_sources.append(0)
        else:
            print("Error: Failed to open default camera 0.")
            video_sources.clear() # Ensure the list is empty if fallback fails
            
# --- Utility Functions (Same as original) ---

def translate_mask(mask, dx, dy):
    """Translate mask by dx, dy pixels."""
    rows, cols = mask.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    translated = cv2.warpAffine(mask, M, (cols, rows), borderValue=0)
    return translated

def mouse_callback(event, x, y, flags, param):
    global selected_track_id, last_seen_time, masks_data, edit_mode
    global drawing, erasing, edited_mask, current_mask_index

    if edit_mode and edited_mask is not None:
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            erasing = False
            cv2.circle(edited_mask, (x, y), brush_size, 255, -1)
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(edited_mask, (x, y), brush_size, 255, -1)
            elif erasing:
                cv2.circle(edited_mask, (x, y), brush_size, 0, -1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
        elif event == cv2.EVENT_RBUTTONDOWN:
            erasing = True
            drawing = False
            cv2.circle(edited_mask, (x, y), brush_size, 0, -1)
        elif event == cv2.EVENT_RBUTTONUP:
            erasing = False

    elif not edit_mode:
        if event == cv2.EVENT_LBUTTONDOWN:
            for idx, (mask, track_id) in enumerate(masks_data):
                # Ensure y, x are within mask bounds
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
                    selected_track_id = track_id
                    last_seen_time = time.time()
                    break

def update_highlight_color():
    global highlight_color, r_slider, g_slider, b_slider, gray_slider
    r = r_slider.get()
    g = g_slider.get()
    b = b_slider.get()
    gray = gray_slider.get()
    highlight_color = [int(b * (gray / 255)), int(g * (gray / 255)), int(r * (gray / 255))]

def gui_thread():
    global edit_mode, selected_track_id, brush_size, r_slider, g_slider, b_slider, gray_slider

    root = tk.Tk()
    root.title("Object Tracker Controls")
    root.geometry("400x700")

    # Functions to queue commands
    def add_video_file(): command_queue.put(('add_video', filedialog.askopenfilename(title="Select Video File", filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*")))))
    def switch_to_cam(idx): command_queue.put(('set_cam', idx))
    def reset_to_laptop_cam(): command_queue.put(('set_cam', 0))
    def clear_all_videos(): command_queue.put(('clear_all', None))
    def reset_track_command(): command_queue.put(('reset_track', None))
    def clear_mask_edits_command(): command_queue.put(('clear_edits', None))
    def toggle_edit_mode_command(): command_queue.put(('toggle_edit', None))
    def adjust_brush(delta):
        global brush_size
        brush_size = max(1, min(50, brush_size + delta))

    # Buttons for actions
    ttk.Button(root, text="Quit", command=lambda: command_queue.put(('quit', None))).pack(pady=5)
    ttk.Button(root, text="Reset Track", command=reset_track_command).pack(pady=5)
    ttk.Button(root, text="Clear Mask Edits", command=clear_mask_edits_command).pack(pady=5)
    ttk.Button(root, text="Pause/Resume for Mask Editing", command=toggle_edit_mode_command).pack(pady=5)
    
    # File/camera controls
    file_frame = ttk.LabelFrame(root, text="Video Sources")
    file_frame.pack(pady=10)
    ttk.Button(file_frame, text="Add Video File", command=add_video_file).pack(pady=5)
    ttk.Button(file_frame, text="Reset to Laptop Cam (0)", command=reset_to_laptop_cam).pack(pady=5)
    ttk.Button(file_frame, text="Clear All Sources", command=clear_all_videos).pack(pady=5)
    
    # Camera switching buttons
    cam_frame = ttk.LabelFrame(root, text="Switch to Camera Index")
    cam_frame.pack(pady=10)
    for i in range(1, 10):
        ttk.Button(cam_frame, text=str(i), command=lambda idx=i: switch_to_cam(idx)).grid(row=(i-1)//5, column=(i-1)%5, padx=5, pady=5)

    # Brush size buttons
    brush_frame = ttk.LabelFrame(root, text="Brush Size")
    brush_frame.pack(pady=10)
    ttk.Button(brush_frame, text="+", command=lambda: adjust_brush(1)).grid(row=0, column=0, padx=5)
    ttk.Button(brush_frame, text="-", command=lambda: adjust_brush(-1)).grid(row=0, column=1, padx=5)

    # Sliders for color
    color_frame = ttk.LabelFrame(root, text="Color Controls")
    color_frame.pack(pady=10)
    # ... (Slider creation code remains the same) ...
    ttk.Label(color_frame, text="R").grid(row=0, column=0)
    r_slider = tk.Scale(color_frame, from_=0, to=255, orient=tk.HORIZONTAL, command=lambda v: update_highlight_color())
    r_slider.set(0)
    r_slider.grid(row=0, column=1)

    ttk.Label(color_frame, text="G").grid(row=1, column=0)
    g_slider = tk.Scale(color_frame, from_=0, to=255, orient=tk.HORIZONTAL, command=lambda v: update_highlight_color())
    g_slider.set(255)
    g_slider.grid(row=1, column=1)

    ttk.Label(color_frame, text="B").grid(row=2, column=0)
    b_slider = tk.Scale(color_frame, from_=0, to=255, orient=tk.HORIZONTAL, command=lambda v: update_highlight_color())
    b_slider.set(0)
    b_slider.grid(row=2, column=1)

    ttk.Label(color_frame, text="Gray").grid(row=3, column=0)
    gray_slider = tk.Scale(color_frame, from_=0, to=255, orient=tk.HORIZONTAL, command=lambda v: update_highlight_color())
    gray_slider.set(255)
    gray_slider.grid(row=3, column=1)

    root.mainloop()


def reset_track():
    global selected_track_id, last_seen_time, edit_mode, paused_frame
    selected_track_id = None
    last_seen_time = None
    if edit_mode:
        edit_mode = False
        paused_frame = None
        cv2.setMouseCallback('Object Tracker', mouse_callback)

def clear_mask_edits():
    global selected_track_id
    if selected_track_id in edited_masks_dict:
        if selected_track_id in original_masks_dict:
            edited_masks_dict[selected_track_id] = original_masks_dict[selected_track_id].copy()
        else:
            del edited_masks_dict[selected_track_id]
        print(f"Cleared mask edits for track ID {selected_track_id}, restored to original")

def toggle_edit_mode():
    global edit_mode, selected_track_id
    if selected_track_id is not None:
        edit_mode = not edit_mode
        if not edit_mode:
            exit_edit_mode()

def exit_edit_mode():
    global edit_mode, paused_frame, edited_mask, selected_track_id, results
    edit_mode = False
    if edited_mask is not None and selected_track_id is not None:
        new_mask = (edited_mask / 255).astype(np.uint8)
        original_center = None
        if 'results' in globals() and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else np.arange(len(boxes))
            for i, tid in enumerate(track_ids):
                if int(tid) == selected_track_id and i < len(boxes):
                    x1, y1, x2, y2 = boxes[i]
                    original_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    break
        if original_center is not None:
            edited_masks_dict[selected_track_id] = {'mask': new_mask, 'center': original_center}
    paused_frame = None
    cv2.setMouseCallback('Object Tracker', mouse_callback)

def adjust_brush(delta):
    global brush_size
    brush_size = max(1, min(50, brush_size + delta))

# --- Main Execution ---

# Start GUI in a separate thread
threading.Thread(target=gui_thread, daemon=True).start()

cv2.namedWindow('Object Tracker')
cv2.setMouseCallback('Object Tracker', mouse_callback)

# --- Initial load of captures, now using VideoReader threads ---
reload_captures() 
print(f"Starting with video sources: {video_sources}")

# --- Main Application Loop (Consumer) ---
while True:
    
    # --- 1. Process commands from the GUI thread ---
    try:
        command, data = command_queue.get(block=False)
        
        if command == 'add_video':
            if len(video_sources) == 1 and video_sources[0] == 0:
                video_sources.clear()
            video_sources.append(data)
            reload_captures()
        elif command == 'set_cam':
            video_sources.clear()
            video_sources.append(data)
            reload_captures()
        elif command == 'clear_all':
            video_sources.clear()
            video_sources.append(0)
            reload_captures()
        elif command == 'quit':
            break
        elif command == 'reset_track':
            reset_track()
        elif command == 'clear_edits':
            clear_mask_edits()
        elif command == 'toggle_edit':
            toggle_edit_mode()

    except queue.Empty:
        pass
    
    
    overlay = None
    
    # --- 2. Tracking/Edit Mode Logic ---
    if not edit_mode:
        frames = []
        
        # 2a. Retrieve latest frame from ALL reader threads
        for reader in video_readers:
            frame = reader.get_latest_frame()
            if frame is not None:
                frames.append(frame)

        if not frames:
            time.sleep(0.01) 
            pass # Use pass so waitKey runs
        else:
            try:
                # 2b. Stack the frames (Main Thread, CPU task)
                combined_frame = np.hstack(frames)
            except ValueError as e:
                print(f"Error stacking frames: {e}")
                pass
            else:
                # 2c. YOLO Inference (Main Thread, CPU/GPU task)
                results = model.track(combined_frame, persist=True, tracker="bytetrack.yaml")
                
                overlay = combined_frame.copy() 
                masks_data = []

                if results[0].masks is not None:
                    masks = results[0].masks.data.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else np.arange(len(masks))
                    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else None

                    for i, mask in enumerate(masks):
                        track_id = int(track_ids[i]) if len(track_ids) > i else i
                        
                        # >>> FIX FOR IndexError: RESIZE MASK TO MATCH combined_frame <<<
                        # combined_frame.shape[1] is width, combined_frame.shape[0] is height
                        mask_resized = cv2.resize(mask, 
                                                  (combined_frame.shape[1], combined_frame.shape[0]), 
                                                  interpolation=cv2.INTER_NEAREST)
                        mask_binary = (mask_resized > 0.5).astype(np.uint8)
                        
                        current_center = None
                        if boxes is not None and i < len(boxes):
                            x1, y1, x2, y2 = boxes[i]
                            current_center = ((x1 + x2) / 2, (y1 + y2) / 2)

                        # Apply user-edited mask translation
                        if track_id in edited_masks_dict and current_center is not None:
                            original_center = edited_masks_dict[track_id]['center']
                            dx = current_center[0] - original_center[0]
                            dy = current_center[1] - original_center[1]
                            # Use the stored (already resized) mask and translate it
                            mask_binary = translate_mask(edited_masks_dict[track_id]['mask'], dx, dy)
                            
                        # Highlighting logic
                        if selected_track_id is None or track_id == selected_track_id:
                            color = highlight_color if track_id == selected_track_id else np.random.randint(0,255,3).tolist()
                            if track_id == selected_track_id: last_seen_time = time.time()
                            
                            # --- 1. Draw the transparent color fill ---
                            colored_mask = np.zeros_like(combined_frame)
                            colored_mask[mask_binary == 1] = color # This now works because mask_binary size matches
                            overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)
                            
                            # --- 2. Draw the outline ---
                            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(overlay, contours, -1, color, 2)

                        masks_data.append((mask_binary, track_id))

                    # Timeout check
                    if selected_track_id is not None and last_seen_time and (time.time() - last_seen_time) > RESET_TIMEOUT:
                        selected_track_id = None
                        last_seen_time = None

                    instruction_text = "Click to track. Use GUI for other controls."
                    cam_text = f"Sources: {len(video_sources)}"
                    cv2.putText(overlay, instruction_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(overlay, cam_text, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)


    else:
        # --- Edit Mode Logic (Remains identical) ---
        if paused_frame is None:
            if 'combined_frame' in locals():
                paused_frame = combined_frame.copy()
            else:
                edit_mode = False
                continue
            
            current_mask_index = None
            for idx, (mask, track_id) in enumerate(masks_data):
                if track_id == selected_track_id:
                    current_mask_index = idx
                    break
            
            if current_mask_index is None:
                edit_mode = False
                paused_frame = None
                continue

            original_mask = masks_data[current_mask_index][0]
            ys, xs = np.where(original_mask > 0)
            if len(xs) > 0 and len(ys) > 0:
                original_center = (np.mean(xs), np.mean(ys))
                original_masks_dict[selected_track_id] = {'mask': original_mask.copy(), 'center': original_center}

            if selected_track_id in edited_masks_dict:
                edited_mask = (edited_masks_dict[selected_track_id]['mask'] * 255).astype(np.uint8)
            else:
                edited_mask = (masks_data[current_mask_index][0] * 255).astype(np.uint8)
            edited_mask = cv2.threshold(edited_mask, 127, 255, cv2.THRESH_BINARY)[1]
            cv2.setMouseCallback('Object Tracker', mouse_callback)
        
        # Draw mask overlay
        colored_mask = np.zeros_like(paused_frame)
        colored_mask[edited_mask == 255] = np.array(highlight_color, dtype=np.uint8)
        overlay = cv2.addWeighted(paused_frame, 1, colored_mask, 0.5, 0)

        cv2.putText(overlay, "Edit Mode: LMB paint, RMB erase, +/- brush size, 'e' exit edit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.putText(overlay, f"Brush size: {brush_size}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    
    # --- 3. Universal Display and Key Handler ---
    if overlay is not None:
        cv2.imshow("Object Tracker", overlay)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        reset_track()
    elif key == ord('c'):
        clear_mask_edits()
    elif key == ord('e'):
        toggle_edit_mode()
        if not edit_mode:
            exit_edit_mode()
    elif key == ord('+') or key == ord('='):
        adjust_brush(1)
    elif key == ord('-') or key == ord('_'):
        adjust_brush(-1)

# --- 4. Cleanup (Updated to stop threads) ---
for reader in video_readers:
    reader.stop()
cv2.destroyAllWindows()
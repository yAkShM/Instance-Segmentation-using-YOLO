
import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import queue

# Load YOLOv8 segmentation model
model = YOLO('yolov8n-seg.pt')

# Globals
selected_track_id = None
last_seen_time = None
RESET_TIMEOUT = 5
masks_data = []

highlight_color = [0, 255, 0]  # Initial green color


video_sources = [0]  # Start with camera 0 by default
caps = []  # List to hold all cv2.VideoCapture objects
STANDARD_HEIGHT = 480 # Target height for all videos
command_queue = queue.Queue()

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

'''def release_and_open_camera(new_index):
    global cap, cam_device_index
    try:
        if cap is not None:
            cap.release()
        cam_device_index = new_index
        cap = cv2.VideoCapture(cam_device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(f"Switched to camera device ID: {cam_device_index}")
    except Exception as e:
        print(f"Error switching camera: {e}. Resetting to default (0).")
        cam_device_index = 0
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        '''

def translate_mask(mask, dx, dy):
    """Translate mask by dx, dy pixels."""
    rows, cols = mask.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    translated = cv2.warpAffine(mask, M, (cols, rows), borderValue=0)
    return translated


# [MODIFIED reload_captures function]
def reload_captures():
    """
    Closes all current video captures and reloads them from the video_sources list.
    THIS FUNCTION MUST ONLY BE CALLED FROM THE MAIN THREAD.
    """
    global caps
    # Release all existing captures
    for cap in caps:
        cap.release()
    caps.clear()

    # Open all sources
    for source in video_sources:
        try:
            # --- MODIFICATION HERE ---
            if isinstance(source, int):
                # Use CAP_DSHOW for cameras, it's more stable on Windows
                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            else:
                # Use default for video files
                cap = cv2.VideoCapture(source)
            # --- END MODIFICATION ---

            if not cap.isOpened():
                print(f"Error: Could not open video source: {source}")
                continue
            
            if isinstance(source, int):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, STANDARD_HEIGHT)
            
            caps.append(cap)
            print(f"Successfully opened: {source}")
        except Exception as e:
            print(f"Failed to open {source}: {e}")
            
    
    if not caps:
        print("No valid video sources. Trying default camera 0...")
        video_sources.clear()
        video_sources.append(0)
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                caps.append(cap)
                print("Successfully opened default camera 0.")
            else:
                print("Error: Failed to open default camera 0.")
        except Exception as e:
            print(f"Exception opening default camera 0: {e}")
# Lock is automatically released here

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
                if mask[y, x] > 0:
                    selected_track_id = track_id
                    last_seen_time = time.time()
                    print(f"Selected object with track ID: {track_id}")
                    break

def update_highlight_color():
    global highlight_color
    r = r_slider.get()
    g = g_slider.get()
    b = b_slider.get()
    gray = gray_slider.get()
    highlight_color = [int(b * (gray / 255)), int(g * (gray / 255)), int(r * (gray / 255))]

# [NEW CODE - Replace your old gui_thread function]
def gui_thread():
    global edit_mode, selected_track_id, brush_size

    root = tk.Tk()
    root.title("Object Tracker Controls")
    root.geometry("400x700")

    # --- GUI-side functions that PUT COMMANDS into the queue ---
    
    def add_video_file():
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("MP4 files", "*.mp4"), ("AVI files", "*.avi"), ("All files", "*.*"))
        )
        if filepath:
            print(f"GUI: Queuing command to add video: {filepath}")
            command_queue.put(('add_video', filepath)) # Send command

    def switch_to_cam(idx):
        print(f"GUI: Queuing command to switch to cam {idx}")
        command_queue.put(('set_cam', idx)) # Send command

    def reset_to_laptop_cam():
        print("GUI: Queuing command to reset to laptop cam.")
        command_queue.put(('set_cam', 0)) # Send command
        
    def clear_all_videos():
        print("GUI: Queuing command to clear all sources.")
        command_queue.put(('clear_all', None)) # Send command

    # Buttons for actions
    ttk.Button(root, text="Quit", command=lambda: command_queue.put(('quit', None))).pack(pady=5) # Send quit command
    ttk.Button(root, text="Reset Track", command=lambda: reset_track()).pack(pady=5)
    ttk.Button(root, text="Clear Mask Edits", command=lambda: clear_mask_edits()).pack(pady=5)
    ttk.Button(root, text="Pause/Resume for Mask Editing", command=lambda: toggle_edit_mode()).pack(pady=5)
    
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
    global r_slider, g_slider, b_slider, gray_slider
    color_frame = ttk.LabelFrame(root, text="Color Controls")
    color_frame.pack(pady=10)

    # ... [Your slider code remains exactly the same] ...
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
        # Restore to original mask if available
        if selected_track_id in original_masks_dict:
            edited_masks_dict[selected_track_id] = original_masks_dict[selected_track_id].copy()
        else:
            del edited_masks_dict[selected_track_id]
        print(f"Cleared mask edits for track ID {selected_track_id}, restored to original")

'''def reset_to_laptop_cam():
    print("Resetting to laptop webcam (device 0).")
    release_and_open_camera(0)'''

def toggle_edit_mode():
    global edit_mode, selected_track_id, paused_frame
    if selected_track_id is not None:
        edit_mode = not edit_mode
        if not edit_mode:
            exit_edit_mode()

def exit_edit_mode():
    global edit_mode, paused_frame, edited_mask, selected_track_id
    # Exit edit mode and save changes with original center from bounding box
    edit_mode = False
    if edited_mask is not None and selected_track_id is not None:
        new_mask = (edited_mask / 255).astype(np.uint8)
        # Get original center from bounding box at edit time
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

'''def switch_camera(idx):
    print(f"Switching to camera device ID {idx}")
    release_and_open_camera(idx)'''

def adjust_brush(delta):
    global brush_size
    brush_size = max(1, min(50, brush_size + delta))

# [NEW CODE - Replace the start of your main loop]

# Start GUI in a separate thread
threading.Thread(target=gui_thread, daemon=True).start()

cv2.namedWindow('Object Tracker')
cv2.setMouseCallback('Object Tracker', mouse_callback)

# --- Initial load of captures ---
reload_captures() 
print(f"Starting with video sources: {video_sources}")

# [MODIFIED Main Loop]

# ... [Your code for starting the GUI thread and cv2.namedWindow] ...
# ... [This code is all the same] ...

# Initial load of captures
reload_captures() 
print(f"Starting with video sources: {video_sources}")

# --- Main Application Loop ---
# --- Main Application Loop ---
while True:
    
    # --- 1. Check for commands from the GUI thread (This part was perfect) ---
    try:
        command, data = command_queue.get(block=False)
        
        if command == 'add_video':
            print(f"MAIN: Received command to add video: {data}")
            if len(video_sources) == 1 and video_sources[0] == 0:
                video_sources.clear() # Remove default cam if adding a file
            video_sources.append(data)
            reload_captures() # Called safely from main thread

        elif command == 'set_cam':
            print(f"MAIN: Received command to set camera: {data}")
            video_sources.clear()
            video_sources.append(data)
            reload_captures() # Called safely from main thread

        elif command == 'clear_all':
            print("MAIN: Received command to clear all sources.")
            video_sources.clear()
            video_sources.append(0) # Default back to cam 0
            reload_captures() # Called safely from main thread

        elif command == 'quit':
            print("MAIN: Received quit command. Exiting.")
            break # Exit the main loop

    except queue.Empty:
        pass # No commands, just continue
    
    
    # [FIX 2: Initialize 'overlay' to None so it always exists]
    overlay = None
    
    # --- 2. Your existing 'edit_mode' or 'tracking' logic ---
    if not edit_mode:
        current_caps = caps 
        current_sources = video_sources

        if not current_caps:
            print("No video captures are open. Waiting...")
            time.sleep(0.1) # Sleep a bit
            # [FIX 1: Changed 'continue' to 'pass' to allow waitKey to run]
            pass

        else: # We have caps, so try to read
            frames = []
            all_frames_read = True
            
            # 1. Read one frame from each source
            for i, cap in enumerate(current_caps):
                ret, frame = cap.read()
                if not ret:
                    if not isinstance(current_sources[i], int):
                        print(f"Video file {current_sources[i]} ended. Looping.")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                    
                    if not ret:
                        all_frames_read = False
                        break 
                
                # 2. Resize all frames
                if ret: 
                    try:
                        h, w, _ = frame.shape
                        scale = STANDARD_HEIGHT / h
                        new_w = int(w * scale)
                        frame_resized = cv2.resize(frame, (new_w, STANDARD_HEIGHT), interpolation=cv2.INTER_AREA)
                        frames.append(frame_resized)
                    except Exception as e:
                        print(f"Error resizing frame from {current_sources[i]}: {e}")
                        all_frames_read = False
                        break
            
            if not all_frames_read or not frames:
                # [FIX 1: Changed 'continue' to 'pass' to allow waitKey to run]
                pass
            
            else: # We have frames, so try to stack
                try:
                    combined_frame = np.hstack(frames)
                except ValueError as e:
                    print(f"Error stacking frames: {e}")
                    # [FIX 1: Changed 'continue' to 'pass' to allow waitKey to run]
                    pass
                else:
                    # --- 4. SUCCESS! Run your EXISTING logic on the combined frame ---
                    results = model.track(combined_frame, persist=True, tracker="bytetrack.yaml")
                    
                    overlay = combined_frame.copy() # 'overlay' is now defined
                    masks_data = []

                    if results[0].masks is not None:
                        masks = results[0].masks.data.cpu().numpy()
                        track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else np.arange(len(masks))
                        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else None

                        for i, mask in enumerate(masks):
                            track_id = int(track_ids[i]) if len(track_ids) > i else i
                            mask_resized = cv2.resize(mask, (combined_frame.shape[1], combined_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                            mask_binary = (mask_resized > 0.5).astype(np.uint8)

                            current_center = None
                            if boxes is not None and i < len(boxes):
                                x1, y1, x2, y2 = boxes[i]
                                current_center = ((x1 + x2) / 2, (y1 + y2) / 2)

                            if track_id in edited_masks_dict and current_center is not None:
                                original_center = edited_masks_dict[track_id]['center']
                                dx = current_center[0] - original_center[0]
                                dy = current_center[1] - original_center[1]
                                mask_binary = translate_mask(edited_masks_dict[track_id]['mask'], dx, dy)

                            if selected_track_id is None or track_id == selected_track_id:
                                if selected_track_id is None:
                                    color = np.random.randint(0,255,3).tolist()
                                else:
                                    color = highlight_color
                                    last_seen_time = time.time()

                                
                
                                # --- 1. Draw the transparent color fill ---
                                # We can re-enable this now that the 'continue' bug is fixed
                                colored_mask = np.zeros_like(combined_frame)
                                colored_mask[mask_binary == 1] = color
                                overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)
                                
                                # --- 2. Draw the outline with the *same color* ---
                                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                # We use 'color' here instead of hard-coded red
                                cv2.drawContours(overlay, contours, -1, color, 2)

                            masks_data.append((mask_binary, track_id))

                    if selected_track_id is not None and last_seen_time and (time.time() - last_seen_time) > RESET_TIMEOUT:
                        print(f"Selected object lost for {RESET_TIMEOUT}s. Resetting to multi-object mode.")
                        selected_track_id = None
                        last_seen_time = None

                    instruction_text = "Click to track. Use GUI for other controls."
                    cam_text = f"Sources: {len(video_sources)}"
                    cv2.putText(overlay, instruction_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    cv2.putText(overlay, cam_text, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    else:
        # --- Your edit_mode logic (This was fine) ---
        if paused_frame is None:
            print("Entering edit mode...")
            # We need to use the last good 'combined_frame' to pause
            # Let's check if 'combined_frame' exists from the last loop
            if 'combined_frame' in locals():
                paused_frame = combined_frame.copy()
            else:
                print("No frame to pause. Exiting edit mode.")
                edit_mode = False
                continue # Use continue here, we'll hit waitKey at the end
            
            current_mask_index = None
            for idx, (mask, track_id) in enumerate(masks_data):
                if track_id == selected_track_id:
                    current_mask_index = idx
                    break
            
            if current_mask_index is None:
                print(f"No mask found for selected track id: {selected_track_id}")
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
        overlay = cv2.addWeighted(paused_frame, 1, colored_mask, 0.5, 0) # 'overlay' is now defined

        cv2.putText(overlay, "Edit Mode: LMB paint, RMB erase, +/- brush size, 'e' exit edit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.putText(overlay, f"Brush size: {brush_size}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    
    # --- 3. Universal Display and Key Handler ---
    
    # [FIX 2: Only show the window if 'overlay' was successfully created]
    if overlay is not None:
        cv2.imshow("Object Tracker", overlay)
    
    # [FIX 1: This waitKey is no longer skipped, so the window will not freeze]
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break # This will also work
    elif key == ord('r'):
        reset_track()
    elif key == ord('c'):
        clear_mask_edits()
    # [I've removed 'b' and '0-9' as they are now in your GUI]
    elif key == ord('e'):
        toggle_edit_mode()
        if not edit_mode:
            exit_edit_mode()
    elif key == ord('+') or key == ord('='):
        adjust_brush(1)
    elif key == ord('-') or key == ord('_'):
        adjust_brush(-1)

# --- 4. Cleanup ---
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
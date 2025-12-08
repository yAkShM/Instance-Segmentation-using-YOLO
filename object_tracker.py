import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import queue
import torchreid
from scipy.spatial import distance
import torch
from PIL import Image


model = YOLO('yolov8n-seg.pt')
print("Loading Person Re-ID model...")


reid_model = torchreid.models.build_model(
    name="osnet_x1_0",  
    num_classes = 1501,
    pretrained=True    
)

reid_model = reid_model.cuda()
reid_model.eval()

reid_transform,_ = torchreid.data.transforms.build_transforms(
    height=256,
    width=128,
    is_test=True  # Use test-mode transforms
)
print("Re-ID model loaded successfully.")

target_fingerprints = {}
highlight_color = [0, 255, 0]

selected_track_ids = set() 
active_edit_track_id = None 

RESET_TIMEOUT = 5 
masks_data = []


reid_targets = {}  
reid_threshold = 0.25
current_matches = {}  


video_sources = [0] 
caps = [] 
STANDARD_HEIGHT = 480 
command_queue = queue.Queue()

edit_mode = False
drawing = False
erasing = False
brush_size = 10
edited_mask = None
paused_frame = None
current_mask_index = None

boxes = None
combined_frame = None

edited_masks_dict = {}
original_masks_dict = {}

track_history = {} 
TRAIL_LENGTH = 30 


def translate_mask(mask, dx, dy):
    rows, cols = mask.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    translated = cv2.warpAffine(mask, M, (cols, rows), borderValue=0)
    return translated


def get_feature_vector(crop):
    """
    Takes a single image crop (NumPy array) of a person
    and returns its 512-dimension feature vector.
    """

    img_pil = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_pil)
    img_tensor = reid_transform(img_pil)
    
    # 2. Add a "batch" dimension and send to the GPU
    img_tensor = img_tensor.unsqueeze(0).cuda()

    # 3. Get the fingerprint!
    with torch.no_grad(): # Disable gradient calculations
        features = reid_model(img_tensor)
    
    # 4. Return the fingerprint (as a simple NumPy array)
    return features.cpu().numpy()[0]

def reload_captures():
    global caps
    for cap in caps:
        cap.release()
    caps.clear()

    for source in video_sources:
        try:
            if isinstance(source, int):
                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(source)

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



def mouse_callback(event, x, y, flags, param):
    global masks_data, edit_mode, drawing, erasing
    global edited_mask, current_mask_index, active_edit_track_id
    global boxes, combined_frame, reid_targets, current_matches

    if edit_mode and edited_mask is not None:
        # [ ... Brush logic remains exactly the same ... ]
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
            if combined_frame is None or boxes is None or not masks_data:
                print("Click Error: Data not ready.")
                return

            print(f"Click at ({x},{y}).")

            for i, (mask, track_id) in enumerate(masks_data):
                # Check if click is within mask bounds
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                    if mask[y, x] > 0:
                        # Found the clicked person!
                        if boxes is not None and i < len(boxes):
                            x1, y1, x2, y2 = map(int, boxes[i])
                            cropped_person = combined_frame[max(0, y1):min(combined_frame.shape[0], y2),
                                                            max(0, x1):min(combined_frame.shape[1], x2)]
                            
                            if cropped_person.size > 0:
                                # 1. Get fingerprint of person we just clicked
                                click_fp = get_feature_vector(cropped_person)
                                
                                # 2. Check if this person is ALREADY in our targets
                                match_found_id = None
                                for t_id, t_data in reid_targets.items():
                                    dist = distance.cosine(t_data['fp'], click_fp)
                                    if dist < reid_threshold:
                                        match_found_id = t_id
                                        break
                                
                                # 3. Toggle Logic
                                if match_found_id is not None:
                                    # We already track them -> REMOVE (Deselect)
                                    del reid_targets[match_found_id]
                                    print(f"Removed target {match_found_id} (Deselected).")
                                else:
                                    # New person -> ADD (Select) with a random unique color
                                    new_color = np.random.randint(50, 255, 3).tolist()
                                    reid_targets[track_id] = {'fp': click_fp, 'color': new_color}
                                    print(f"Added new target {track_id} with color {new_color}.")
                            else:
                                print("Error: Crop empty.")
                        break

def gui_thread():
    global edit_mode, brush_size 

    root = tk.Tk()
    root.title("Object Tracker Controls")
    root.geometry("400x700")

    
    def add_video_file():
        filepath = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(("MP4 files", ".mp4"), ("AVI files", ".avi"), ("All files", "."))
        )
        if filepath:
            print(f"GUI: Queuing command to add video: {filepath}")
            command_queue.put(('add_video', filepath)) 

    def switch_to_cam(idx):
        print(f"GUI: Queuing command to switch to cam {idx}")
        command_queue.put(('set_cam', idx)) 

    def reset_to_laptop_cam():
        print("GUI: Queuing command to reset to laptop cam.")
        command_queue.put(('set_cam', 0)) 
        
    def clear_all_videos():
        print("GUI: Queuing command to clear all sources.")
        command_queue.put(('clear_all', None)) 

    ttk.Button(root, text="Quit", command=lambda: command_queue.put(('quit', None))).pack(pady=5) 
    ttk.Button(root, text="Reset/Clear Selection", command=lambda: reset_track()).pack(pady=5)
    ttk.Button(root, text="Clear Mask Edit (for selected)", command=lambda: clear_mask_edits()).pack(pady=5)
    ttk.Button(root, text="Pause/Resume for Mask Editing", command=lambda: toggle_edit_mode()).pack(pady=5)
    
    file_frame = ttk.LabelFrame(root, text="Video Sources")
    file_frame.pack(pady=10)
    ttk.Button(file_frame, text="Add Video File", command=add_video_file).pack(pady=5)
    ttk.Button(file_frame, text="Reset to Laptop Cam (0)", command=reset_to_laptop_cam).pack(pady=5)
    ttk.Button(file_frame, text="Clear All Sources", command=clear_all_videos).pack(pady=5)
    
    cam_frame = ttk.LabelFrame(root, text="Switch to Camera Index")
    cam_frame.pack(pady=10)
    for i in range(1, 10):
        ttk.Button(cam_frame, text=str(i), command=lambda idx=i: switch_to_cam(idx)).grid(row=(i-1)//5, column=(i-1)%5, padx=5, pady=5)

    brush_frame = ttk.LabelFrame(root, text="Brush Size")
    brush_frame.pack(pady=10)
    ttk.Button(brush_frame, text="+", command=lambda: adjust_brush(1)).grid(row=0, column=0, padx=5)
    ttk.Button(brush_frame, text="-", command=lambda: adjust_brush(-1)).grid(row=0, column=1, padx=5)

    

    root.mainloop()



def reset_track():
    global selected_track_ids, edit_mode, paused_frame, track_history
    # --- ADD THESE GLOBALS ---
    
    global reid_targets, current_matches

    
    track_history.clear()
    reid_targets.clear()
    current_matches.clear()
    
    reid_target_track_id = None
    reid_target_fingerprint = None

    if edit_mode:
        edit_mode = False
        paused_frame = None
        cv2.setMouseCallback('Object Tracker', mouse_callback)
    print("Cleared.")

def clear_mask_edits():
    global selected_track_ids, edit_mode, paused_frame 

    if len(selected_track_ids) != 1:
        print("Please select exactly ONE object to clear its mask edit.")
        return

    track_id_to_clear = next(iter(selected_track_ids)) 

    if track_id_to_clear in edited_masks_dict:
        del edited_masks_dict[track_id_to_clear]
        print(f"Cleared mask edit for track ID {track_id_to_clear}. Resuming live detection.")
    else:
        print(f"No mask edit found for track ID {track_id_to_clear}.")


    if edit_mode:
        edit_mode = False
        paused_frame = None
        cv2.setMouseCallback('Object Tracker', mouse_callback) 
        print("Exiting edit mode, resuming live tracking.")


def toggle_edit_mode():
    global edit_mode, selected_track_ids, paused_frame, active_edit_track_id
    
    if len(selected_track_ids) == 1:
        edit_mode = not edit_mode
        if edit_mode:
            active_edit_track_id = next(iter(selected_track_ids))
        else:
            exit_edit_mode()
    elif edit_mode:
         edit_mode = False
         exit_edit_mode()
    else:
        print("Please select exactly ONE object to enter edit mode.")

def exit_edit_mode():
    global edit_mode, paused_frame, edited_mask, active_edit_track_id
    edit_mode = False
    if edited_mask is not None and active_edit_track_id is not None:
        new_mask = (edited_mask / 255).astype(np.uint8)
        original_center = None
        if boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else np.arange(len(boxes))
            for i, tid in enumerate(track_ids):
                if int(tid) == active_edit_track_id: 
                    x1, y1, x2, y2 = boxes[i]
                    original_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    break
        if original_center is not None:
            edited_masks_dict[active_edit_track_id] = {'mask': new_mask, 'center': original_center} 
    
    paused_frame = None
    edited_mask = None
    active_edit_track_id = None 
    cv2.setMouseCallback('Object Tracker', mouse_callback)


def adjust_brush(delta):
    global brush_size
    brush_size = max(1, min(50, brush_size + delta))


threading.Thread(target=gui_thread, daemon=True).start()

cv2.namedWindow('Object Tracker')
cv2.setMouseCallback('Object Tracker', mouse_callback)

reload_captures() 
print(f"Starting with video sources: {video_sources}")


reload_captures() 
print(f"Starting with video sources: {video_sources}")


while True:
    
    try:
        command, data = command_queue.get(block=False)
        
        if command == 'add_video':
            print(f"MAIN: Received command to add video: {data}")
            if len(video_sources) == 1 and video_sources[0] == 0:
                video_sources.clear() 
            video_sources.append(data)
            reload_captures() 

        elif command == 'set_cam':
            print(f"MAIN: Received command to set camera: {data}")
            video_sources.clear()
            video_sources.append(data)
            reload_captures() 

        elif command == 'clear_all':
            print("MAIN: Received command to clear all sources.")
            video_sources.clear()
            video_sources.append(0) 
            reload_captures() 

        elif command == 'quit':
            print("MAIN: Received quit command. Exiting.")
            break 
            
    except queue.Empty:
        pass 
    
    
    overlay = None
    
    if not edit_mode:
        current_caps = caps 
        current_sources = video_sources

        if not current_caps:
            print("No video captures are open. Waiting...")
            time.sleep(0.1) 
            pass

        else: 
            frames = []
            all_frames_read = True
            
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
                pass
            
            else: 
                try:
                    
                    combined_frame = np.hstack(frames)
                except ValueError as e:
                    print(f"Error stacking frames: {e}")
                    pass
                else:
                    results = model.track(combined_frame, persist=True, tracker="bytetrack.yaml", verbose=False)
                    
                    overlay = combined_frame.copy() 
                    #masks_data = [] # <--- DO NOT RESET IT HERE
                    
                    new_masks_data = [] # <--- Create a *new, temporary* list
                    current_track_ids = set()


                if results[0].masks is not None:
                    masks = results[0].masks.data.cpu().numpy()
                    
                    # --- Calculate boxes (Keep this fix!) ---
                    if results[0].boxes is not None and results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        track_ids = results[0].boxes.id.cpu().numpy()
                    else:
                        boxes = None
                        track_ids = np.arange(len(masks))
                    
                    # --- 1. MULTI-TARGET RE-ID MATCHING ---
                    current_matches = {} # Reset current frame matches
                    
                    # First, assume any "Original" target ID that is still on screen is a match
                    for i, t_id in enumerate(track_ids):
                        tid_int = int(t_id)
                        if tid_int in reid_targets:
                            current_matches[tid_int] = reid_targets[tid_int]['color']

                    # Now, check everyone else using Re-ID
                    if reid_targets:
                        # Collect unknown people (those not already matched by ID)
                        crops_to_process = []
                        track_ids_to_process = []
                        
                        for i, mask in enumerate(masks):
                            track_id = int(track_ids[i]) if len(track_ids) > i else i
                            
                            if track_id not in current_matches: # Only check if not already known
                                if boxes is not None and i < len(boxes):
                                    x1, y1, x2, y2 = map(int, boxes[i])

                                    # Calculate height
                                    person_height = y2 - y1
                                    
                                    # [FIX 1] Ignore people who are too small (e.g., < 100 pixels tall)
                                    if person_height < 100:
                                        continue

                                    # [FIX 3] Tighter Crop
                                    # Crop 5% from edges to remove background noise
                                    h, w = y2 - y1, x2 - x1
                                    crop_y1 = int(y1 + h * 0.05)
                                    crop_y2 = int(y2 - h * 0.05)
                                    crop_x1 = int(x1 + w * 0.05)
                                    crop_x2 = int(x2 - w * 0.05)

                                    
                                    cropped_person = combined_frame[max(0, y1):min(combined_frame.shape[0], y2),
                                                                   max(0, x1):min(combined_frame.shape[1], x2)]
                                    if cropped_person.size > 0:
                                        crops_to_process.append(cropped_person)
                                        track_ids_to_process.append(track_id)

                        # Batch Inference
                        if crops_to_process:
                            batch_tensors = []
                            for crop in crops_to_process:
                                img_pil = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                                img_pil = Image.fromarray(img_pil)
                                batch_tensors.append(reid_transform(img_pil))
                            
                            batch_input = torch.stack(batch_tensors).cuda()
                            with torch.no_grad():
                                batch_features = reid_model(batch_input) 
                            batch_fingerprints = batch_features.cpu().numpy()

                            
                            # [REPLACE THE OLD 'for j' LOOP WITH THIS]
                            
                            # Compare each person on screen against ALL targets
                            for j, current_fp in enumerate(batch_fingerprints):
                                current_track_id = track_ids_to_process[j]
                                
                                # Start with a "None" match
                                best_dist = 1.0 
                                best_color = None
                                
                               
                                if current_track_id in track_history:
                                    current_threshold = 0.35  # Lenient for existing tracks
                                else:
                                    current_threshold = 0.20  # Strict for new tracks
                                
                                # Check against all targets
                                for t_id, t_data in reid_targets.items():
                                    dist = distance.cosine(t_data['fp'], current_fp)
                                    
                                    # Uncomment for debugging:
                                    # print(f"DEBUG: ID {current_track_id} vs Target {t_id} = {dist:.4f} (Thresh: {current_threshold})")

                                    # Check if it matches our dynamic threshold AND is the best match so far
                                    if dist < current_threshold and dist < best_dist:
                                        best_dist = dist
                                        best_color = t_data['color']
                                
                                # If we found a valid match after checking all targets
                                if best_color is not None:
                                    current_matches[current_track_id] = best_color

                    # --- 2. DRAWING LOOP ---
                    new_masks_data = [] # Temp list for click fix

                    for i, mask in enumerate(masks):
                        track_id = int(track_ids[i]) if len(track_ids) > i else i
                        
                        current_track_ids.add(track_id)

                        mask_resized = cv2.resize(mask, (combined_frame.shape[1], combined_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                        mask_binary = (mask_resized > 0.5).astype(np.uint8)

                        current_center = None
                        if boxes is not None and i < len(boxes):
                            x1, y1, x2, y2 = boxes[i]
                            current_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                            
                            if track_id not in track_history:
                                track_history[track_id] = []
                            track_history[track_id].append((int(current_center[0]), int(current_center[1])))
                            if len(track_history[track_id]) > TRAIL_LENGTH:
                                track_history[track_id].pop(0) 

                        if track_id in edited_masks_dict and current_center is not None:
                            original_center = edited_masks_dict[track_id]['center']
                            dx = current_center[0] - original_center[0]
                            dy = current_center[1] - original_center[1]
                            mask_binary = translate_mask(edited_masks_dict[track_id]['mask'], dx, dy)

                        # --- Color Logic ---
                        color_to_draw = None
                        
                        # Is this person in our 'current_matches' list?
                        if track_id in current_matches:
                            color_to_draw = current_matches[track_id]
                        else:
                            # If they are not a match, DO NOT DRAW THEM.
                            color_to_draw = None
                        
                        if color_to_draw is not None:
                            colored_mask = np.zeros_like(combined_frame)
                            colored_mask[mask_binary == 1] = color_to_draw
                            overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0)
                            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(overlay, contours, -1, color_to_draw, 2)

                        new_masks_data.append((mask_binary, track_id))
                    
                    # --- Update Global Mask Data (The Flicker Fix) ---
                    if new_masks_data:
                        masks_data = new_masks_data
                    
                    # --- Cleanup History ---
                    for old_track_id in list(track_history.keys()):
                        if old_track_id not in current_track_ids:
                            del track_history[old_track_id]

                    
                    
    # [ ... your 'if not edit_mode:' block ends here ... ]

    else:
        # --- THIS IS THE NEW, CORRECTED "EDIT MODE" BLOCK ---
        if paused_frame is None:
            print("Entering edit mode...")
            
            # --- THE FIX ---
            # Check the GLOBAL variable, not the 'local' one
            if combined_frame is not None:
                paused_frame = combined_frame.copy()
            else:
            # ---------------
                print("No frame to pause. Exiting edit mode.")
                edit_mode = False
                continue 
            
            current_mask_index = None
            for idx, (mask, track_id) in enumerate(masks_data):
                if track_id == active_edit_track_id: 
                    current_mask_index = idx
                    break
            
            if current_mask_index is None:
                print(f"No mask found for selected track id: {active_edit_track_id}") 
                edit_mode = False
                paused_frame = None
                continue

            # --- This logic below is unchanged, but now it will run ---
            original_mask = masks_data[current_mask_index][0]
            ys, xs = np.where(original_mask > 0)
            if len(xs) > 0 and len(ys) > 0:
                original_center = (np.mean(xs), np.mean(ys))
                original_masks_dict[active_edit_track_id] = {'mask': original_mask.copy(), 'center': original_center} 

            if active_edit_track_id in edited_masks_dict: 
                edited_mask = (edited_masks_dict[active_edit_track_id]['mask'] * 255).astype(np.uint8) 
            else:
                edited_mask = (masks_data[current_mask_index][0] * 255).astype(np.uint8)
            
            edited_mask = cv2.threshold(edited_mask, 127, 255, cv2.THRESH_BINARY)[1]
            cv2.setMouseCallback('Object Tracker', mouse_callback)
        
        # This part will now work because paused_frame is set
        colored_mask = np.zeros_like(paused_frame)
        colored_mask[edited_mask == 255] = np.array(highlight_color, dtype=np.uint8)
        overlay = cv2.addWeighted(paused_frame, 1, colored_mask, 0.5, 0) 

        cv2.putText(overlay, f"Editing ID: {active_edit_track_id} | LMB paint, RMB erase, +/- brush", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        cv2.putText(overlay, f"Brush size: {brush_size} | 'e' or GUI to exit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    
    
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
    elif key == ord('+') or key == ord('='):
        adjust_brush(1)
    elif key == ord('-') or key == ord('_'):
        adjust_brush(-1)

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
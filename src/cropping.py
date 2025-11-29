"""
Module for saliency-aware vertical reframing (9:16) with Active Speaker Detection.
"""
import cv2
import mediapipe as mp
import numpy as np
import math
from collections import deque

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """
        min_cutoff: Decrease to reduce jitter (e.g., 0.01 - 1.0)
        beta: Increase to reduce lag during fast movement (e.g., 0.0 - 1.0)
        d_cutoff: Cutoff frequency for derivative
        """
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        """
        t: Current timestamp
        x: Current noisy value (e.g., face center x-coordinate)
        """
        t_e = t - self.t_prev
        if t_e <= 0: return self.x_prev # Avoid divide by zero

        # Filter the derivative (speed of change)
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)

        # Adjust cutoff based on speed (this is the magic of One Euro)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

class FaceTracker:
    def __init__(self, max_history=15, width=1920):
        self.tracks = {} # id -> {center_x, mar_history, last_seen_frame}
        self.next_id = 0
        self.max_history = max_history
        self.width = width

    def update(self, faces, frame_idx):
        # faces: list of {'center_x': float, 'mar': float}
        used_track_ids = set()
        
        for face in faces:
            center_x = face['center_x']
            mar = face['mar']
            
            # Find closest existing track
            best_dist = float('inf')
            best_id = -1
            
            for tid, track in self.tracks.items():
                if tid in used_track_ids:
                    continue
                dist = abs(track['center_x'] - center_x)
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid
            
            # Threshold for matching (e.g., 15% of width)
            MATCH_THRESHOLD = self.width * 0.15
            
            if best_id != -1 and best_dist < MATCH_THRESHOLD:
                # Update existing
                self.tracks[best_id]['center_x'] = center_x
                self.tracks[best_id]['mar_history'].append(mar)
                self.tracks[best_id]['last_seen_frame'] = frame_idx
                used_track_ids.add(best_id)
            else:
                # Create new
                self.tracks[self.next_id] = {
                    'center_x': center_x,
                    'mar_history': deque([mar], maxlen=self.max_history),
                    'last_seen_frame': frame_idx
                }
                used_track_ids.add(self.next_id)
                self.next_id += 1
                
        # Cleanup old tracks
        to_remove = []
        for tid, track in self.tracks.items():
            if frame_idx - track['last_seen_frame'] > 30: # Lost for ~1 sec
                to_remove.append(tid)
        for tid in to_remove:
            del self.tracks[tid]

    def get_score(self, tid):
        if tid not in self.tracks or not self.tracks[tid]['mar_history']:
            return 0.0
        return sum(self.tracks[tid]['mar_history']) / len(self.tracks[tid]['mar_history'])

    def get_center(self, tid):
        if tid in self.tracks:
            return self.tracks[tid]['center_x']
        return None

    def get_active_speaker_id(self, current_id=None, hysteresis_factor=1.2, silence_threshold=0.01):
        best_score = -1
        best_id = -1
        
        # Find the track with the highest average MAR
        for tid in self.tracks:
            score = self.get_score(tid)
            if score > best_score:
                best_score = score
                best_id = tid
        
        # If everyone is silent, stick to current or return None
        if best_score < silence_threshold:
            return current_id if current_id is not None else best_id
            
        # If we have a current speaker, apply hysteresis
        if current_id is not None and current_id in self.tracks:
            current_score = self.get_score(current_id)
            # Only switch if new best is significantly better
            if best_id != current_id:
                if best_score > (current_score * hysteresis_factor):
                    return best_id
                else:
                    return current_id
        
        return best_id

def get_mouth_aspect_ratio(landmarks, width, height):
    """
    Calculates the Mouth Aspect Ratio (MAR) to detect speaking.
    """
    # Indices for mouth landmarks (MediaPipe Face Mesh)
    # Upper lip: 13, Lower lip: 14, Left corner: 78, Right corner: 308
    
    # Helper to get coords
    def get_point(idx):
        return np.array([landmarks[idx].x * width, landmarks[idx].y * height])

    upper = get_point(13)
    lower = get_point(14)
    left  = get_point(78)
    right = get_point(308)
    
    vertical_dist = np.linalg.norm(upper - lower)
    horizontal_dist = np.linalg.norm(left - right)
    
    if horizontal_dist == 0:
        return 0
    return vertical_dist / horizontal_dist

def calculate_crop_coordinates(video_path, start_time, end_time, output_width=1080, output_height=1920, debug=True):
    """
    Calculates dynamic crop coordinates using Active Speaker Detection (MAR).
    
    Args:
        video_path (str): Path to the input video.
        start_time (float): Start time of the clip.
        end_time (float): End time of the clip.
        output_width (int): Target width.
        output_height (int): Target height.
        debug (bool): If True, saves a debug video with overlays.
        
    Returns:
        list: List of (frame_time, x_crop, y_crop) tuples.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Use Face Mesh for better landmarks (mouth)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=5, # Increased to handle more people
        refine_landmarks=True, # Better lip landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    crop_coordinates = []
    
    # Initialize One Euro Filter
    one_euro_filter = None
    
    # Active Speaker Tracking State
    tracker = FaceTracker(max_history=15, width=width) # ~0.5s history at 30fps
    current_speaker_id = None
    last_valid_center_x = width / 2
    
    # Debug Video Writer
    debug_writer = None
    if debug:
        import os
        debug_path = os.path.join("temp", "debug_tracking.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        debug_writer = cv2.VideoWriter(debug_path, fourcc, fps, (width, height))
        print(f"Debug mode enabled. Saving visualization to {debug_path}")
    
    print(f"Analyzing frames {start_frame} to {end_frame} for active speaker tracking...")
    
    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = frame_idx / fps
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        detected_faces = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Calculate bounding box center
                x_coords = [lm.x for lm in face_landmarks.landmark]
                y_coords = [lm.y for lm in face_landmarks.landmark]
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                
                center_x = (min_x + max_x) / 2 * width
                center_y = (min_y + max_y) / 2 * height
                
                mar = get_mouth_aspect_ratio(face_landmarks.landmark, width, height)
                detected_faces.append({
                    'center_x': center_x, 
                    'mar': mar,
                    'bbox': (int(min_x*width), int(min_y*height), int(max_x*width), int(max_y*height))
                })
        
        # Update tracker
        tracker.update(detected_faces, frame_idx)
        
        # Determine active speaker
        # Hysteresis: New speaker must be 20% louder (visually) to switch
        # Silence Threshold: 0.015 (below this, assume silence)
        active_id = tracker.get_active_speaker_id(
            current_id=current_speaker_id, 
            hysteresis_factor=1.2, 
            silence_threshold=0.015
        )
        
        if active_id is not None:
            current_speaker_id = active_id
            center = tracker.get_center(active_id)
            if center is not None:
                last_valid_center_x = center
        
        target_center_x = last_valid_center_x

        # Apply One Euro Filter
        if one_euro_filter is None:
            one_euro_filter = OneEuroFilter(current_time, target_center_x, min_cutoff=0.05, beta=0.5)
            smoothed_center_x = target_center_x
        else:
            smoothed_center_x = one_euro_filter(current_time, target_center_x)
        
        # Calculate crop x (top-left corner)
        crop_h = height
        crop_w = int(height * (9/16))
        
        crop_x = int(smoothed_center_x - crop_w / 2)
        
        # Clamp to boundaries
        crop_x = max(0, min(crop_x, width - crop_w))
        crop_y = 0 
        
        crop_coordinates.append({
            'time': current_time,
            'x': crop_x,
            'y': crop_y,
            'w': crop_w,
            'h': crop_h
        })
        
        # --- Debug Visualization ---
        if debug and debug_writer:
            debug_frame = frame.copy()
            
            # Draw all tracked faces
            for tid, track in tracker.tracks.items():
                # Find which detected face corresponds to this track (approximate by center_x)
                # This is just for visualization, so simple matching is fine
                matched_face = None
                for face in detected_faces:
                    if abs(face['center_x'] - track['center_x']) < width * 0.05:
                        matched_face = face
                        break
                
                score = tracker.get_score(tid)
                color = (0, 255, 0) if tid == current_speaker_id else (0, 0, 255)
                
                if matched_face:
                    x1, y1, x2, y2 = matched_face['bbox']
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(debug_frame, f"ID: {tid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.putText(debug_frame, f"MAR: {matched_face['mar']:.3f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(debug_frame, f"Avg: {score:.3f}", (x1, y2 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    # Draw a circle where we think the track is
                    cx = int(track['center_x'])
                    cv2.circle(debug_frame, (cx, height//2), 20, color, -1)
                    cv2.putText(debug_frame, f"ID: {tid} (Lost)", (cx, height//2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Draw Crop Area
            cv2.rectangle(debug_frame, (int(crop_x), 0), (int(crop_x + crop_w), height), (255, 255, 0), 4)
            cv2.putText(debug_frame, "CROP AREA", (int(crop_x), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            debug_writer.write(debug_frame)
        # ---------------------------
        
    cap.release()
    if debug_writer:
        debug_writer.release()
    face_mesh.close()
    return crop_coordinates


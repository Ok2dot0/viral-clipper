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

def calculate_crop_coordinates(video_path, start_time, end_time, output_width=1080, output_height=1920):
    """
    Calculates dynamic crop coordinates using Active Speaker Detection (MAR).
    
    Args:
        video_path (str): Path to the input video.
        start_time (float): Start time of the clip.
        end_time (float): End time of the clip.
        output_width (int): Target width.
        output_height (int): Target height.
        
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
        max_num_faces=3,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    crop_coordinates = []
    
    # Initialize One Euro Filter
    one_euro_filter = None
    
    # Active Speaker Tracking State
    last_active_center_x = width / 2
    speaker_history = deque(maxlen=10) # Store recent MARs to smooth switching
    
    print(f"Analyzing frames {start_frame} to {end_frame} for active speaker tracking...")
    
    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = frame_idx / fps
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        
        target_center_x = last_active_center_x
        
        if results.multi_face_landmarks:
            best_mar = 0
            active_face_center = None
            
            # Find the face with the highest Mouth Aspect Ratio (MAR)
            for face_landmarks in results.multi_face_landmarks:
                # Calculate bounding box center
                x_coords = [lm.x for lm in face_landmarks.landmark]
                # y_coords = [lm.y for lm in face_landmarks.landmark]
                min_x, max_x = min(x_coords), max(x_coords)
                center_x = (min_x + max_x) / 2 * width
                
                mar = get_mouth_aspect_ratio(face_landmarks.landmark, width, height)
                
                if mar > best_mar:
                    best_mar = mar
                    active_face_center = center_x
            
            # Simple logic: If MAR is above a threshold, update target.
            # Otherwise, stay on last known position (or the face with highest MAR if it's significant)
            
            # Threshold for "speaking" (approximate, depends on face)
            SPEAKING_THRESHOLD = 0.02 
            
            if best_mar > SPEAKING_THRESHOLD:
                target_center_x = active_face_center
                last_active_center_x = active_face_center
            else:
                # If no one is clearly speaking, stick to the last known speaker position
                # But check if the last known position still has a face nearby?
                # For simplicity, we just hold the last position.
                target_center_x = last_active_center_x

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
        
    cap.release()
    face_mesh.close()
    return crop_coordinates


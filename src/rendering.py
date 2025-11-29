"""
Module for final video composition and rendering using FFmpeg.
"""
import ffmpeg
import os
import math

def time_to_ass_format(seconds):
    """Converts seconds to ASS time format (H:MM:SS.cc)."""
    centiseconds = int((seconds - int(seconds)) * 100)
    seconds = int(seconds)
    minutes = seconds // 60
    hours = minutes // 60
    minutes %= 60
    seconds %= 60
    return f"{hours}:{minutes:02}:{seconds:02}.{centiseconds:02}"

def generate_ass_karaoke(clip_words, output_path):
    """
    Generates an ASS subtitle file with karaoke highlighting.
    """
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,80,&H00FFFF,&HFFFFFF,&H000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,20,20,150,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    
    events = []
    
    # Group words into lines (simple logic: max 4 words or 25 chars)
    lines = []
    current_line = []
    current_len = 0
    
    for w in clip_words:
        word_len = len(w['word'])
        if len(current_line) >= 4 or (current_len + word_len > 25):
            lines.append(current_line)
            current_line = [w]
            current_len = word_len
        else:
            current_line.append(w)
            current_len += word_len + 1
            
    if current_line:
        lines.append(current_line)
        
    for line in lines:
        if not line: continue
        start_time = line[0]['start']
        end_time = line[-1]['end']
        
        # Construct karaoke text
        text = ""
        for w in line:
            duration = w['end'] - w['start']
            cs = int(duration * 100) # centiseconds
            # {\kXX} highlights the text in PrimaryColour (Yellow), starting from Secondary (White)
            # We add a space after the word if it's not the last one
            text += f"{{\\k{cs}}}{w['word'].strip()} "
            
        events.append(f"Dialogue: 0,{time_to_ass_format(start_time)},{time_to_ass_format(end_time)},Default,,0,0,0,,{text.strip()}")
        
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header)
        for e in events:
            f.write(e + "\n")

def generate_crop_expression(crop_data, duration, start_time_offset):
    """
    Generates an FFmpeg expression for dynamic cropping (x-coordinate).
    Uses piecewise linear interpolation.
    """
    print(f"[DEBUG] Generating crop expression. Total points: {len(crop_data) if crop_data else 0}")
    print(f"[DEBUG] Clip Start: {start_time_offset}, Duration: {duration}")

    if not crop_data:
        print("[DEBUG] No crop data provided. Using default 0.")
        return "0"
        
    # Downsample to avoid huge expressions (1 point every 0.2s for smoother tracking)
    points = []
    last_t = -1
    
    # Filter points relevant to this clip and normalize time
    valid_points = []
    for c in crop_data:
        # Normalize time to start from 0 for the clip
        rel_time = c['time'] - start_time_offset
        if 0 <= rel_time <= duration:
            valid_points.append({**c, 'time': rel_time})
            
    print(f"[DEBUG] Valid points within clip duration: {len(valid_points)}")
    if valid_points:
        print(f"[DEBUG] First valid point: {valid_points[0]}")
        print(f"[DEBUG] Last valid point: {valid_points[-1]}")

    if not valid_points:
        print(f"[DEBUG] No valid points found in range. Using first crop point: {crop_data[0]['x']}")
        return str(crop_data[0]['x'])

    for c in valid_points:
        if c['time'] - last_t >= 0.2: # Increased resolution to 0.2s
            points.append(c)
            last_t = c['time']
            
    # Ensure start and end points
    if points[0]['time'] > 0:
        points.insert(0, {**points[0], 'time': 0})
    if points[-1]['time'] < duration:
        points.append({**points[-1], 'time': duration})
        
    # Build expression: sum of segments
    # We use summation instead of nested if() to avoid recursion depth limits and reduce string length overhead.
    # Logic: sum( (t >= t0 and t < t1) * lerp(...) )
    segments = []
    
    for i in range(len(points)-1):
        p0 = points[i]
        p1 = points[i+1]
        t0 = p0['time']
        t1 = p1['time']
        x0 = p0['x']
        x1 = p1['x']
        
        # Avoid division by zero
        if t1 <= t0: continue
        
        # Condition: Active during this segment
        # For the last segment, we include the end time (lte) to ensure the final frame is covered.
        # For others, we use strictly less than (lt) to avoid double-counting at the boundary where t == t1.
        if i == len(points) - 2:
            cond = f"between(t,{t0},{t1})" # inclusive [t0, t1]
        else:
            cond = f"(gte(t,{t0})*lt(t,{t1}))" # [t0, t1)
            
        # Interpolation
        # We wrap lerp in parentheses just in case
        segment_expr = f"lerp({x0},{x1},(t-{t0})/{t1-t0})"
        
        segments.append(f"{cond}*{segment_expr}")
        
    if not segments:
        return str(crop_data[0]['x'])
        
    # Join all segments with +
    expr = "+".join(segments)
        
    print(f"[DEBUG] Generated Expression Length: {len(expr)}")
    # print(f"[DEBUG] Expression Preview: {expr[:200]}...")
    return expr

def render_clip(video_path, start_time, end_time, crop_data, transcript_segments, output_path, b_roll_paths=None):
    """
    Renders the final vertical video with dynamic cropping, karaoke captions, and B-roll.
    """
    duration = end_time - start_time
    
    # 1. Generate Karaoke ASS
    ass_path = output_path.replace(".mp4", ".ass")
    
    clip_words = []
    for seg in transcript_segments:
        if seg['end'] > start_time and seg['start'] < end_time:
            if 'words' in seg:
                for w in seg['words']:
                    if w['end'] > start_time and w['start'] < end_time:
                        new_w = w.copy()
                        new_w['start'] = max(0, w['start'] - start_time)
                        new_w['end'] = min(duration, w['end'] - start_time)
                        clip_words.append(new_w)
                        
    generate_ass_karaoke(clip_words, ass_path)
    
    # 2. Calculate Dynamic Crop Expression
    if crop_data:
        crop_w = crop_data[0]['w']
        crop_h = crop_data[0]['h']
        # Pass start_time to normalize timestamps
        x_expr = generate_crop_expression(crop_data, duration, start_time)
    else:
        crop_w = 608
        crop_h = 1080
        x_expr = "0"
        
    print(f"[DEBUG] Final Crop Expression: {x_expr[:500]}...")

    # 3. Build FFmpeg command
    try:
        input_stream = ffmpeg.input(video_path, ss=start_time, t=duration)
        
        # Background Layer
        background = (
            input_stream
            .filter('scale', 1080, 1920, force_original_aspect_ratio='increase')
            .filter('crop', 1080, 1920)
            .filter('boxblur', 20)
            .filter('setsar', 1)
        )
        
        # Foreground Layer (Dynamic Crop)
        # IMPORTANT: The crop filter in ffmpeg evaluates expressions per frame.
        # We must ensure 't' is relative to the start of the filter chain.
        # When using ss/t in input, the timestamps in the filter chain usually start from 0.
        foreground = (
            input_stream
            .crop(x=x_expr, y=0, width=int(crop_w), height=int(crop_h))
            .filter('scale', 1080, -1)
        )
        
        # Combine Background and Foreground
        # We use 'overlay' filter.
        # Note: x_expr is for the CROP filter, not the overlay.
        # The overlay is centered: x=(W-w)/2
        main_video = ffmpeg.overlay(background, foreground, x='(W-w)/2', y='(H-h)/2')
        
        # 4. B-Roll Overlay (if available)
        if b_roll_paths:
            # Simple logic: Overlay the first B-roll clip at 1/3rd of the duration for 3 seconds
            b_roll_start = duration / 3
            b_roll_duration = 3.0
            
            # Load B-roll
            b_roll_input = ffmpeg.input(b_roll_paths[0], t=b_roll_duration)
            # Scale B-roll to fit width (1080) and maintain aspect, then crop to 1080x1920 or just center
            # Usually B-roll should fill the screen or be a cutaway.
            # Let's make it fill the screen (1080x1920)
            b_roll_processed = (
                b_roll_input
                .filter('scale', 1080, 1920, force_original_aspect_ratio='increase')
                .filter('crop', 1080, 1920)
            )
            
            # Overlay B-roll
            # enable='between(t, start, end)'
            main_video = ffmpeg.overlay(
                main_video, 
                b_roll_processed, 
                x=0, y=0, 
                enable=f"between(t,{b_roll_start},{b_roll_start+b_roll_duration})"
            )
        
        # 5. Apply Karaoke Subtitles
        video_stream = main_video.filter('ass', ass_path)
        audio_stream = input_stream.audio
        
        stream = (
            ffmpeg.output(video_stream, audio_stream, output_path, vcodec='libx264', acodec='aac', audio_bitrate='192k', crf=18, preset='slow')
            .overwrite_output()
        )
        
        print(f"Rendering clip to {output_path} with dynamic features...")
        # Enable verbose logging for debugging
        stream.run(quiet=False) 
        print("Rendering complete.")
        
    except ffmpeg.Error as e:
        error_message = e.stderr.decode('utf8') if e.stderr else "Unknown FFmpeg error (no stderr)"
        print(f"Error rendering clip: {error_message}")
        raise


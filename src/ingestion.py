"""
Module for video ingestion and audio extraction.
"""
import os
import ffmpeg
import yt_dlp

def download_youtube_video(url, output_dir):
    """
    Downloads a video from YouTube using yt-dlp.
    
    Args:
        url (str): YouTube video URL.
        output_dir (str): Directory to save the downloaded video.
        
    Returns:
        str: Path to the downloaded video file.
    """
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'noplaylist': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_path = ydl.prepare_filename(info_dict)
        
    return video_path

def extract_audio(video_path, output_audio_path):
    """
    Extracts audio from the input video file using FFmpeg.
    
    Args:
        video_path (str): Path to the input video file.
        output_audio_path (str): Path to save the extracted audio (e.g., .wav or .mp3).
        
    Returns:
        str: Path to the extracted audio file if successful, None otherwise.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        print(f"Extracting audio from {video_path} to {output_audio_path}...")
        (
            ffmpeg
            .input(video_path)
            .output(output_audio_path, acodec='pcm_s16le', ac=1, ar='16k') # 16kHz mono for Whisper
            .overwrite_output()
            .run(quiet=True)
        )
        print("Audio extraction complete.")
        return output_audio_path
    except ffmpeg.Error as e:
        print(f"Error extracting audio: {e.stderr.decode('utf8')}")
        raise


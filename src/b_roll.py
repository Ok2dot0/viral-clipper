"""
Module for B-Roll acquisition using RAKE keyword extraction and Pexels API.
"""
from rake_nltk import Rake
from pexels_api import API
import os
import requests
import random

def extract_keywords(text, num_keywords=3):
    """
    Extracts key phrases from text using RAKE.
    """
    r = Rake()
    r.extract_keywords_from_text(text)
    # Get ranked phrases
    phrases = r.get_ranked_phrases()
    # Filter for length (avoid too long/short)
    clean_phrases = [p for p in phrases if 3 < len(p) < 30]
    return clean_phrases[:num_keywords]

def download_video(url, output_path):
    """Downloads a video from a URL."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        return output_path
    return None

def get_b_roll(text, duration, output_dir, pexels_api_key):
    """
    Fetches and downloads B-roll video based on text content.
    
    Args:
        text (str): The transcript text to analyze.
        duration (float): Target duration (not strictly used for search, but for context).
        output_dir (str): Directory to save downloaded videos.
        pexels_api_key (str): API Key for Pexels.
        
    Returns:
        str: Path to the downloaded video file, or None if failed.
    """
    if not pexels_api_key:
        print("Pexels API key missing.")
        return None
        
    keywords = extract_keywords(text)
    if not keywords:
        keywords = ["abstract background", "nature", "city"]
        
    print(f"Searching Pexels for keywords: {keywords}")
    
    api = API(pexels_api_key)
    
    # Try keywords in order
    video_url = None
    for query in keywords:
        try:
            # Search for portrait videos
            api.search(query, page=1, results_per_page=5)
            videos = api.get_videos()
            
            if videos:
                # Find a video with portrait orientation if possible, or just take one
                # Pexels API python wrapper might not expose orientation easily in all versions,
                # but we can check width/height if available, or just pick one.
                # We'll pick a random one to vary content
                video = random.choice(videos)
                
                # Get the highest quality video file
                # The library usually returns a list of video objects
                # We need to access the video files
                # Note: The pexels-api-py library structure might vary, 
                # assuming standard structure or direct dict access if it returns dicts.
                # Let's assume 'video' is an object with 'video_files' attribute
                
                # Safety check for library structure
                if hasattr(video, 'video_files'):
                    files = video.video_files
                elif isinstance(video, dict) and 'video_files' in video:
                    files = video['video_files']
                else:
                    continue
                    
                # Sort by resolution (width * height) to get HD
                # We prefer 1080x1920 (portrait) or high res
                best_file = None
                for vf in files:
                    # We want portrait if possible
                    if vf.get('width') and vf.get('height'):
                        if vf['height'] > vf['width']: # Portrait
                            best_file = vf
                            break
                
                if not best_file and files:
                    best_file = files[0] # Fallback
                    
                if best_file:
                    video_url = best_file.get('link')
                    break
        except Exception as e:
            print(f"Error searching Pexels for '{query}': {e}")
            continue
            
    if video_url:
        filename = f"broll_{random.randint(1000,9999)}.mp4"
        output_path = os.path.join(output_dir, filename)
        print(f"Downloading B-roll: {video_url}")
        return download_video(video_url, output_path)
    
    return None

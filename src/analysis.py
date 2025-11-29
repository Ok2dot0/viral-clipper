"""
Module for viral moment detection using VADER and LLM.
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
import json

def analyze_transcript(transcript_segments, min_duration=15, max_duration=60, sentiment_threshold=0.75):
    """
    Analyzes the transcript to find viral moments using VADER sentiment analysis.
    
    Args:
        transcript_segments (list): List of transcript segments from WhisperX.
        min_duration (int): Minimum clip duration in seconds.
        max_duration (int): Maximum clip duration in seconds.
        sentiment_threshold (float): Threshold for sentiment intensity (absolute compound score).
        
    Returns:
        list: List of potential clips (dictionaries with start, end, score, text).
    """
    analyzer = SentimentIntensityAnalyzer()
    potential_clips = []
    
    # Flatten words to a continuous timeline for easier sliding window analysis
    # (Simplified approach: iterate segments and combine)
    
    # Basic sliding window over segments
    n = len(transcript_segments)
    for i in range(n):
        current_duration = 0
        current_text = ""
        start_time = transcript_segments[i]['start']
        segment_scores = []
        
        # Hook detection: Check if the first segment has a high sentiment or starts a sharp rise
        first_seg_score = abs(analyzer.polarity_scores(transcript_segments[i]['text'])['compound'])
        
        for j in range(i, n):
            seg = transcript_segments[j]
            current_duration = seg['end'] - start_time
            current_text += " " + seg['text']
            
            # Calculate sentiment for this accumulated segment
            score = analyzer.polarity_scores(seg['text'])
            segment_scores.append(abs(score['compound']))
            
            if current_duration >= min_duration:
                if current_duration <= max_duration:
                    # Check if this window is "viral"
                    # Metric: Average sentiment intensity of segments in window
                    avg_score = sum(segment_scores) / len(segment_scores)
                    
                    # Boost score if it starts with a "Hook" (high initial sentiment)
                    final_score = avg_score
                    if first_seg_score > 0.6:
                        final_score *= 1.2 # 20% boost for strong hooks
                    
                    if final_score >= sentiment_threshold:
                        potential_clips.append({
                            'start': start_time,
                            'end': seg['end'],
                            'score': final_score,
                            'text': current_text.strip()
                        })
                else:
                    # Exceeded max duration, stop expanding this window
                    break
    
    # Sort by score descending
    potential_clips.sort(key=lambda x: x['score'], reverse=True)
    return potential_clips

def select_best_clip_with_llm(potential_clips, llm_config):
    """
    Uses an LLM to select the best clip from the potential candidates.
    
    Args:
        potential_clips (list): List of potential clips.
        llm_config (dict): Configuration for the LLM (provider, base_url, model, etc.).
        
    Returns:
        dict: The selected best clip.
    """
    if not potential_clips:
        return None
        
    # Take top 5 candidates to save tokens
    candidates = potential_clips[:5]
    
    prompt = """You are an expert video editor for TikTok, Instagram Reels, and YouTube Shorts. Your goal is to identify the single most viral segment from the provided candidates.

A viral segment MUST have:
1. **The Hook (0-3s)**: Does it start with a surprising statement, a question, a loud reaction, or a controversial opinion? (Crucial for retention).
2. **Emotional Intensity**: Is there laughter, shock, anger, debate, or intense storytelling?
3. **Completeness**: Does the clip tell a coherent micro-story or end on a perfect cliffhanger?
4. **Relatability/Shareability**: Is this something people would tag their friends in?

Input: A list of candidate clips with timestamps and text.

Output: Return ONLY a JSON object with the following structure (no markdown formatting):
{
    "selected_clip_index": int,
    "virality_score": float (0-10),
    "reason": "string explaining the hook and emotional value",
    "category": "Funny" | "Educational" | "Debate" | "Story" | "Shocking",
    "title": "A catchy, clickbait-style title for the clip"
}

Candidates:
"""
    for i, clip in enumerate(candidates):
        prompt += f"Option {i+1}: {json.dumps(clip)}\n"
        
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that outputs raw JSON."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2, # Lower temperature for more deterministic selection
        "stream": False
    }
    
    if llm_config.get("model"):
        payload["model"] = llm_config["model"]

    try:
        response = requests.post(
            f"{llm_config['base_url']}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload
        )
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # Try to parse JSON from the response (it might be wrapped in markdown)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        selected_data = json.loads(content)
        
        # Map back to the original candidate
        idx = selected_data.get("selected_clip_index", 1) - 1
        if 0 <= idx < len(candidates):
            best_candidate = candidates[idx]
            # Merge LLM insights
            best_candidate.update(selected_data)
            # Optionally update the main score to the LLM's score (normalized or raw)
            # best_candidate['score'] = selected_data.get('virality_score', best_candidate['score'])
            return best_candidate
            
        return candidates[0]
        
    except Exception as e:
        print(f"LLM selection failed: {e}. Falling back to highest VADER score.")
        return candidates[0]


# Main orchestrator for the AI Video Repurposing Engine
import os
import sys
import yaml
import json
from dotenv import load_dotenv
from src import ingestion, transcription, analysis, cropping, rendering, b_roll

# Load environment variables
load_dotenv()

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("AI Video Repurposing Engine Started")
    
    # 1. Load Config
    config = load_config()
    
    input_dir = config['paths']['input_dir']
    output_dir = config['paths']['output_dir']
    temp_dir = config['paths']['temp_dir']
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # 2. Process Videos
    video_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp4', '.mov', '.mkv'))]
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        print(f"\nProcessing {video_file}...")
        
        try:
            # Step 1: Ingestion
            audio_path = os.path.join(temp_dir, f"{os.path.splitext(video_file)[0]}.wav")
            if not os.path.exists(audio_path):
                ingestion.extract_audio(video_path, audio_path)
            
            # Step 2: Transcription
            # Check if transcript already exists to save time
            transcript_path = os.path.join(temp_dir, f"{os.path.splitext(video_file)[0]}_transcript.json")
            if os.path.exists(transcript_path):
                print("Loading existing transcript...")
                with open(transcript_path, 'r') as f:
                    transcript_result = json.load(f)
            else:
                transcript_result = transcription.transcribe_audio(
                    audio_path, 
                    model_size=config['whisper']['model_size'],
                    device=config['whisper']['device'],
                    compute_type=config['whisper']['compute_type'],
                    batch_size=config['whisper']['batch_size']
                )
                # Save transcript
                with open(transcript_path, 'w') as f:
                    json.dump(transcript_result, f)
            
            # Step 3: Analysis
            print("Analyzing for viral moments...")
            potential_clips = analysis.analyze_transcript(
                transcript_result['segments'],
                min_duration=config['analysis']['min_clip_duration'],
                max_duration=config['analysis']['max_clip_duration'],
                sentiment_threshold=config['analysis']['sentiment_threshold']
            )
            
            if not potential_clips:
                print("No suitable clips found based on criteria.")
                continue
                
            # Select best clip (LLM or Top Score)
            if config['llm']['provider'] != "none":
                print("Selecting best clip with LLM...")
                best_clip = analysis.select_best_clip_with_llm(potential_clips, config['llm'])
            else:
                best_clip = potential_clips[0]
                
            print(f"Selected Clip: {best_clip['text'][:50]}... (Score: {best_clip['score']:.2f})")
            print(f"Time: {best_clip['start']} - {best_clip['end']}")
            
            # Step 4: Cropping
            print("Calculating crop coordinates...")
            crop_data = cropping.calculate_crop_coordinates(
                video_path, 
                best_clip['start'], 
                best_clip['end']
            )
            
            # Step 4.5: B-Roll Acquisition
            b_roll_paths = []
            if config.get('b_roll', {}).get('enabled', True):
                print("Fetching B-Roll...")
                pexels_key = os.getenv("PEXELS_API_KEY")
                if pexels_key:
                    b_roll_path = b_roll.get_b_roll(
                        best_clip['text'], 
                        best_clip['end'] - best_clip['start'], 
                        temp_dir, 
                        pexels_key
                    )
                    if b_roll_path:
                        b_roll_paths.append(b_roll_path)
                else:
                    print("PEXELS_API_KEY not found. Skipping B-Roll.")

            # Step 5: Rendering
            output_filename = f"{os.path.splitext(video_file)[0]}_short.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            rendering.render_clip(
                video_path,
                best_clip['start'],
                best_clip['end'],
                crop_data,
                transcript_result['segments'],
                output_path,
                b_roll_paths=b_roll_paths
            )
            
            print(f"Successfully created {output_path}")
            
        except Exception as e:
            print(f"Failed to process {video_file}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()


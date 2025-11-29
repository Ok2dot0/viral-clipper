import gradio as gr
import os
import yaml
import json
from dotenv import load_dotenv
from src import ingestion, transcription, analysis, cropping, rendering

# Load environment variables
load_dotenv()

# Load Config
def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

def process_video(video_file, youtube_url, min_duration, max_duration, sentiment_threshold, use_llm):
    """
    Process a single video file and return the path to the generated short.
    """
    status_log = []
    def log(msg):
        status_log.append(msg)
        return "\n".join(status_log)

    video_path = None
    
    # Handle Input Source
    if youtube_url and youtube_url.strip():
        yield None, log(f"Downloading video from YouTube: {youtube_url}...")
        try:
            video_path = ingestion.download_youtube_video(youtube_url, config['paths']['input_dir'])
            yield None, log(f"Downloaded: {os.path.basename(video_path)}")
        except Exception as e:
            yield None, log(f"Error downloading YouTube video: {e}")
            return None, "\n".join(status_log)
    elif video_file is not None:
        video_path = video_file.name
    else:
        return None, "No video file or YouTube URL provided."

    temp_dir = config['paths']['temp_dir']
    output_dir = config['paths']['output_dir']
    
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        yield None, log(f"Processing {os.path.basename(video_path)}...")
        
        # Step 1: Ingestion
        audio_path = os.path.join(temp_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}.wav")
        if not os.path.exists(audio_path):
            yield None, log("Extracting audio...")
            ingestion.extract_audio(video_path, audio_path)
        
        # Step 2: Transcription
        transcript_path = os.path.join(temp_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_transcript.json")
        if os.path.exists(transcript_path):
            yield None, log("Loading existing transcript...")
            with open(transcript_path, 'r') as f:
                transcript_result = json.load(f)
        else:
            yield None, log("Transcribing audio (this may take a while)...")
            transcript_result = transcription.transcribe_audio(
                audio_path, 
                model_size=config['whisper']['model_size'],
                device=config['whisper']['device'],
                compute_type=config['whisper']['compute_type'],
                batch_size=config['whisper']['batch_size']
            )
            with open(transcript_path, 'w') as f:
                json.dump(transcript_result, f)
        
        # Step 3: Analysis
        yield None, log("Analyzing for viral moments...")
        potential_clips = analysis.analyze_transcript(
            transcript_result['segments'],
            min_duration=min_duration,
            max_duration=max_duration,
            sentiment_threshold=sentiment_threshold
        )
        
        if not potential_clips:
            yield None, log("No suitable clips found based on criteria.")
            return None, "\n".join(status_log)
            
        # Select best clip
        if use_llm and config['llm']['provider'] != "none":
            yield None, log("Selecting best clip with LLM...")
            best_clip = analysis.select_best_clip_with_llm(potential_clips, config['llm'])
        else:
            best_clip = potential_clips[0]
            
        yield None, log(f"Selected Clip: {best_clip['text'][:50]}... (Score: {best_clip['score']:.2f})")
        
        # Step 4: Cropping
        yield None, log("Calculating crop coordinates...")
        crop_data = cropping.calculate_crop_coordinates(
            video_path, 
            best_clip['start'], 
            best_clip['end']
        )
        
        # Step 5: Rendering
        output_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_short.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        yield None, log(f"Rendering clip to {output_path}...")
        rendering.render_clip(
            video_path,
            best_clip['start'],
            best_clip['end'],
            crop_data,
            transcript_result['segments'],
            output_path
        )
        
        yield output_path, log("Processing complete!")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield None, log(f"Error: {str(e)}")

# Gradio Interface
with gr.Blocks(title="AI Video Repurposing Engine") as app:
    gr.Markdown("# 🎬 AI Video Repurposing Engine")
    gr.Markdown("Upload a long-form video or provide a YouTube URL to automatically generate viral shorts with face tracking and captions.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.File(label="Upload Video", file_types=[".mp4", ".mov", ".mkv"])
            youtube_url_input = gr.Textbox(label="Or Enter YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
            
            with gr.Accordion("Advanced Settings", open=False):
                min_duration = gr.Slider(minimum=5, maximum=60, value=15, step=1, label="Min Clip Duration (s)")
                max_duration = gr.Slider(minimum=15, maximum=120, value=60, step=1, label="Max Clip Duration (s)")
                sentiment_threshold = gr.Slider(minimum=0.0, maximum=1.0, value=0.6, step=0.1, label="Sentiment Threshold")
                use_llm = gr.Checkbox(label="Use LLM for Selection", value=True)
            
            generate_btn = gr.Button("Generate Short", variant="primary")
        
        with gr.Column():
            video_output = gr.Video(label="Generated Short")
            status_output = gr.Textbox(label="Status Log", lines=10)

    generate_btn.click(
        fn=process_video,
        inputs=[video_input, youtube_url_input, min_duration, max_duration, sentiment_threshold, use_llm],
        outputs=[video_output, status_output]
    )

if __name__ == "__main__":
    app.queue().launch(server_name="0.0.0.0", server_port=7860)

# Local AI Video Repurposing Engine

A high-quality, local, and free implementation of an AI video repurposing tool (similar to Ssemble/OpusClip). It takes long-form videos, identifies viral moments using AI (Sentiment Analysis + LLM), and automatically crops them into vertical (9:16) shorts with face tracking and captions.

## Features

- **AI Clipping:** Detects viral moments using VADER sentiment analysis and optional LLM refinement.
- **High-Quality Transcription:** Uses **WhisperX** for accurate ASR and word-level timestamps.
- **Smart Cropping:** Uses **MediaPipe** for face detection to keep the speaker centered in a 9:16 frame.
- **Automated Rendering:** Uses **FFmpeg** to compose the final video with captions.
- **Local & Free:** Runs entirely on your machine (requires GPU for best performance).

## Prerequisites

- **Python 3.10+**
- **NVIDIA GPU** (Recommended for WhisperX and faster processing) with CUDA installed.
- **FFmpeg** installed and added to system PATH.

## Installation

1.  **Clone the repository** (or navigate to the workspace).

2.  **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    *Note: You may need to install PyTorch with CUDA support first if the default pip install doesn't pick the right version for your GPU.*
    ```bash
    # Install PyTorch (adjust command for your CUDA version, e.g., cu118 or cu121)
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Install other requirements
    pip install -r requirements.txt
    ```

4.  **Configuration:**
    *   Copy `.env.example` to `.env` and add your API keys (if using OpenAI/OpenRouter).
    *   Edit `config.yaml` to adjust settings (paths, model sizes, thresholds).

## Usage

### Command Line Interface (CLI)
1.  Place your long-form videos (mp4, mkv, mov) in the `input` folder.
2.  Run the main script:
    ```bash
    python main.py
    ```
3.  The processed shorts will be saved in the `output` folder.

### Graphical User Interface (GUI)
1.  Run the Gradio app:
    ```bash
    python app.py
    ```
2.  Open your browser and navigate to `http://localhost:7860`.
3.  Upload a video and click "Generate Short".

## Configuration (`config.yaml`)

- **whisper**: Adjust `model_size` (e.g., `large-v2`, `medium`) and `device`.
- **analysis**: Set `sentiment_threshold` to control how "emotional" a clip needs to be.
- **llm**: Configure your LLM provider (e.g., `lm_studio` for local LLM, or `openai`).
- **cropping**: Toggle face tracking.

## Troubleshooting

- **WhisperX Issues:** Ensure you have the correct CUDA toolkit installed for your PyTorch version.
- **FFmpeg Errors:** Make sure `ffmpeg` is accessible from your terminal.

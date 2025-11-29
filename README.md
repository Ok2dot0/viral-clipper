# AI Video Repurposing Engine (Viral Clipper)

A powerful, local, and free AI tool designed to repurpose long-form videos into engaging, viral short-form content (Shorts, Reels, TikToks). This engine automates the entire production pipeline: from downloading and transcription to viral moment detection, smart cropping, and dynamic subtitle generation.

## 🚀 Features

*   **Multi-Source Ingestion:**
    *   Supports local video files (`.mp4`, `.mov`, `.mkv`).
    *   Directly downloads and processes videos from **YouTube URLs** using `yt-dlp`.
*   **High-Precision Transcription:**
    *   Utilizes **WhisperX** for state-of-the-art automatic speech recognition (ASR).
    *   Provides accurate **word-level timestamps** essential for precise clipping and subtitle animation.
*   **Intelligent Viral Detection:**
    *   **Sentiment Analysis:** Uses **VADER** to detect emotional peaks and "hooks" in the audio using a sliding window approach.
    *   **LLM Integration:** Optionally uses Large Language Models (OpenAI, Local LLMs via LM Studio) to contextually select the most shareable moments from the top candidates.
*   **Cinematic Smart Cropping:**
    *   **Face Tracking:** Uses **MediaPipe** to detect faces and keep the speaker centered in a 9:16 vertical frame.
    *   **Smooth Camera Movement:** Implements a **One Euro Filter** to eliminate jitter and create professional, smooth camera pans.
    *   **Dynamic Backgrounds:** Automatically fills empty space with blurred versions of the video for a polished look.
*   **Engaging Visuals:**
    *   **Karaoke-Style Subtitles:** Generates "pop-in" word-level subtitles (`.ass` format) to maximize viewer retention.
    *   **Automated B-Roll:** (Optional) Fetches and overlays relevant stock footage (e.g., from Pexels) to break up visual monotony.
*   **Dual Interfaces:**
    *   **CLI (Command Line Interface):** For batch processing and automation.
    *   **Web UI (Gradio):** For an easy-to-use, interactive experience.

## 🧠 How It Works

The pipeline consists of five distinct stages:

1.  **Ingestion (`src/ingestion.py`):**
    *   Downloads video from YouTube (if URL provided) or loads local file.
    *   Extracts audio to `.wav` format for processing.

2.  **Transcription (`src/transcription.py`):**
    *   The audio is passed to **WhisperX**, which performs ASR.
    *   Crucially, it performs **forced alignment** to generate precise start and end times for every single word. This is required for the "karaoke" subtitle effect.

3.  **Analysis (`src/analysis.py`):**
    *   The transcript is analyzed using a sliding window (e.g., 15-60 seconds).
    *   **VADER** calculates a sentiment score for each window. High emotion (positive or negative) = higher viral potential.
    *   **LLM Refinement (Optional):** Top scoring clips are sent to an LLM (like GPT-4 or a local Llama 3) to pick the one with the best narrative hook.

4.  **Cropping (`src/cropping.py`):**
    *   The video is analyzed frame-by-frame using **MediaPipe** to find the speaker's face.
    *   Raw face coordinates are noisy. We apply a **One Euro Filter** (a low-pass filter adapted for human movement) to smooth out the camera motion, simulating a professional cameraman.
    *   The video is cropped to 9:16. If the video is horizontal, a blurred background is generated to fill the vertical frame.

5.  **Rendering (`src/rendering.py`):**
    *   **FFmpeg** is used to compose the final video.
    *   It combines the cropped video, the blurred background, and the generated `.ass` subtitles.
    *   B-roll footage is overlaid if enabled.

## 🛠️ Prerequisites

Before installing, ensure you have the following:

*   **Python 3.10+** installed.
*   **FFmpeg** installed and added to your system's PATH.
    *   *Windows:* `winget install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/).
    *   *Linux:* `sudo apt install ffmpeg`
    *   *macOS:* `brew install ffmpeg`
*   **NVIDIA GPU** (Highly Recommended):
    *   Required for efficient WhisperX transcription and faster video processing.
    *   Ensure **CUDA Toolkit** is installed and matches your PyTorch version.

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Ok2dot0/viral-clipper.git
    cd viral-clipper
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/macOS
    source venv/bin/activate
    ```

3.  **Install PyTorch with CUDA Support**
    *   *Crucial Step:* The default `pip install torch` might not install the CUDA-enabled version. Visit [pytorch.org](https://pytorch.org/get-started/locally/) to get the correct command for your system.
    *   Example for CUDA 11.8:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```

4.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Configuration Guide

The project uses `config.yaml` for extensive customization. Here is a detailed breakdown:

### `whisper`
*   `model_size`: The size of the Whisper model. `large-v3` is most accurate but requires ~10GB VRAM. Use `medium` or `small` for lower VRAM.
*   `device`: `cuda` (GPU) or `cpu`.
*   `compute_type`: `float16` (faster on GPU) or `int8` (less memory).

### `analysis`
*   `min_clip_duration`: Minimum length of a generated short (seconds).
*   `max_clip_duration`: Maximum length (seconds).
*   `sentiment_threshold`: (0.0 to 1.0). How "emotional" the text must be to be considered. Higher = stricter.

### `llm`
*   `provider`: `openai`, `lm_studio` (for local models), or `none`.
*   `base_url`: URL for the API (e.g., `http://localhost:1234/v1` for LM Studio).
*   `model`: The model identifier string.

### `cropping`
*   `face_tracking`: Enable/disable face detection.
*   `min_cutoff`: (One Euro Filter) Lower values = smoother but more lag. Higher = more responsive but jittery. Default `0.05` is good for talking heads.
*   `beta`: (One Euro Filter) Speed coefficient.

## 🖥️ Usage

### Option 1: Web UI (Recommended)
Launch the interactive Gradio interface:
```bash
python app.py
```
*   Open the provided local URL (usually `http://127.0.0.1:7860`) in your browser.
*   Paste a YouTube URL or upload a video file.
*   Adjust settings and click "Submit".

### Option 2: Command Line Interface (CLI)
1.  Place your video files (`.mp4`, `.mkv`, `.mov`) inside the `input/` folder.
2.  Run the main script:
    ```bash
    python main.py
    ```
3.  The script will process all videos in the input folder and save the results to `output/`.

## ❓ Troubleshooting

*   **`RuntimeError: CUDA error: no kernel image is available...`**
    *   Your PyTorch version does not match your installed CUDA driver. Reinstall PyTorch with the correct CUDA version (see Installation step 3).
*   **`FileNotFoundError: [WinError 2] The system cannot find the file specified`**
    *   FFmpeg is likely not in your system PATH. Verify by running `ffmpeg -version` in a terminal.
*   **`AttributeError: module 'whisper' has no attribute 'load_model'`**
    *   Ensure you installed `whisperx` and not just `openai-whisper`. The requirements file handles this, but conflicts can occur.
*   **Slow Processing?**
    *   Ensure `device: cuda` is set in `config.yaml` and your GPU is actually being used (check Task Manager).

## 📂 Project Structure

```
├── input/              # Drop source videos here
├── output/             # Generated shorts are saved here
├── temp/               # Intermediate files (audio, transcripts)
├── src/
│   ├── ingestion.py    # Video downloading and audio extraction
│   ├── transcription.py# WhisperX integration
│   ├── analysis.py     # Viral moment detection logic
│   ├── cropping.py     # Face detection and smart cropping
│   ├── rendering.py    # FFmpeg composition and subtitles
│   └── b_roll.py       # B-roll fetching and overlay
├── app.py              # Gradio Web UI entry point
├── main.py             # CLI entry point
├── config.yaml         # Configuration settings
└── requirements.txt    # Python dependencies
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

[MIT License](LICENSE)
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

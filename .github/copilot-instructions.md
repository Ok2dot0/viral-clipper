# AI Video Repurposing Engine - Project Plan

- [x] **Project Setup**
    - [x] Create project structure (`src/`, `input/`, `output/`, `temp/`).
    - [x] Create `requirements.txt` with dependencies (WhisperX, FFmpeg, MediaPipe, Gradio, yt-dlp).
    - [x] Create `config.yaml` and `.env` template.

- [x] **Core Modules Implementation**
    - [x] `src/ingestion.py`: Video loading, YouTube download, and audio extraction.
    - [x] `src/transcription.py`: WhisperX integration for word-level timestamps.
    - [x] `src/analysis.py`: Viral moment detection (VADER + LLM).
    - [x] `src/cropping.py`: Face detection and dynamic cropping.
    - [x] `src/rendering.py`: FFmpeg composition and subtitle generation.
    - [x] `main.py`: CLI Orchestrator.

- [x] **User Interface**
    - [x] Create `app.py` for Gradio Web UI.
    - [x] Add YouTube URL support to Gradio UI.
    - [x] Update `README.md` with usage instructions.

- [x] **Quality Improvements (Completed)**
    - [x] **Smooth Face Tracking**: Implement One Euro Filter in `src/cropping.py` to reduce camera jitter.
    - [x] **Cinematic Rendering**: Update `src/rendering.py` to support:
        - [x] Blurred background padding for 9:16 conversion.
        - [x] "Pop-in" word-level subtitles (Karaoke style).
        - [x] Dynamic Panning (Lerp expressions).
    - [x] **Enhanced Analysis**: Verify LLM prompts for better viral clip selection.
    - [x] **B-Roll Integration**: Automated B-Roll fetching and overlay.

- [ ] **Final Polish**
    - [ ] Verify full pipeline execution.
    - [ ] Clean up temporary files.

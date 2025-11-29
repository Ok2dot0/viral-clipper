"""
Module for ASR and word-level timestamp generation using WhisperX.
"""
import torch

# Monkey patch torch.load to fix PyTorch 2.6+ breaking change with pyannote/whisperx
# This forces weights_only=False to allow loading legacy checkpoints
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # Force weights_only=False to handle legacy checkpoints containing omegaconf objects
    # This overrides any explicit weights_only=True passed by libraries
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import whisperx
import gc

def transcribe_audio(audio_path, model_size="large-v3", device="cuda", compute_type="float16", batch_size=16):
    """
    Transcribes audio and aligns it to get word-level timestamps using WhisperX.
    
    Args:
        audio_path (str): Path to the audio file.
        model_size (str): Whisper model size (e.g., "base", "small", "medium", "large-v3").
        device (str): Device to run on ("cuda" or "cpu").
        compute_type (str): Compute type ("float16" or "int8").
        batch_size (int): Batch size for transcription.
        
    Returns:
        dict: Transcription result with word-level timestamps.
    """
    print(f"Loading WhisperX model: {model_size} on {device}...")
    
    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    
    print("Transcribing audio...")
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size)
    
    # Free up memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    print("Aligning timestamps...")
    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    # Free up memory
    del model_a
    gc.collect()
    torch.cuda.empty_cache()
    
    print("Transcription and alignment complete.")
    return result

def transcribe_and_diarize(audio_path, hf_token, model_size="large-v3", device="cuda", compute_type="float16", batch_size=16):
    """
    Transcribes, aligns, and diarizes audio using WhisperX.
    Requires a Hugging Face token for pyannote.audio.
    """
    print(f"Loading WhisperX model: {model_size} on {device}...")
    
    # 1. Transcribe
    model = whisperx.load_model(model_size, device, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size)
    
    # Free up memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. Align
    print("Aligning timestamps...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    
    del model_a
    gc.collect()
    torch.cuda.empty_cache()
    
    # 3. Diarize
    print("Diarizing speakers...")
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
    diarize_segments = diarize_model(audio)
    
    # 4. Assign Speakers
    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    print("Diarization complete.")
    return result


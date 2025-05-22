"""
FastAPI Server for Dia TTS Model - SillyTavern Compatible
"""

import io
import os
import re
import tempfile
import time
import json
import uuid
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, Header

from dia.model import Dia


# Request/Response Models
class TTSRequest(BaseModel):
    model: str = Field(default="dia", description="Model name (required for OpenAI compatibility)")
    input: str = Field(..., max_length=4096, description="Text to convert to speech")
    voice: str = Field(default="alloy", description="Voice identifier")
    response_format: str = Field(default="wav", description="Audio format (wav, mp3)")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Speech speed")


class VoiceInfo(BaseModel):
    name: str
    voice_id: str
    preview_url: Optional[str] = None


class TTSGenerateRequest(BaseModel):
    """Alternative format for SillyTavern-Extras compatibility"""
    text: str = Field(..., max_length=4096)
    speaker: str = Field(default="alloy")


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


class VoiceMapping(BaseModel):
    style: str
    primary_speaker: str
    audio_prompt: Optional[str] = None


class VoiceMappingUpdate(BaseModel):
    voice_id: str
    style: Optional[str] = None
    primary_speaker: Optional[str] = None
    audio_prompt: Optional[str] = None


class ServerConfig(BaseModel):
    debug_mode: bool = False
    save_outputs: bool = False
    show_prompts: bool = False
    output_retention_hours: int = 24


class GenerationLog(BaseModel):
    id: str
    timestamp: datetime
    text: str
    processed_text: str
    voice: str
    audio_prompt_used: bool
    generation_time: float
    file_path: Optional[str] = None
    file_size: Optional[int] = None


# Initialize FastAPI app
app = FastAPI(
    title="Dia TTS Server",
    description="FastAPI server for Dia text-to-speech model, compatible with SillyTavern",
    version="1.0.0"
)

# Add CORS middleware for web compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model: Optional[Dia] = None

# Security (optional, accepts any bearer token)
security = HTTPBearer(auto_error=False)

# Voice mapping (Dia uses speaker tags [S1]/[S2], we'll map common voice names)
VOICE_MAPPING: Dict[str, Dict[str, Any]] = {
    "alloy": {"style": "neutral", "primary_speaker": "S1", "audio_prompt": None},
    "echo": {"style": "calm", "primary_speaker": "S1", "audio_prompt": None}, 
    "fable": {"style": "expressive", "primary_speaker": "S2", "audio_prompt": None},
    "nova": {"style": "friendly", "primary_speaker": "S1", "audio_prompt": None},
    "onyx": {"style": "deep", "primary_speaker": "S2", "audio_prompt": None},
    "shimmer": {"style": "bright", "primary_speaker": "S1", "audio_prompt": None},
}

# Store uploaded audio prompts
AUDIO_PROMPTS: Dict[str, np.ndarray] = {}

# Server configuration
SERVER_CONFIG = ServerConfig()

# Generation logs
GENERATION_LOGS: Dict[str, GenerationLog] = {}

# Output directory for saved files
OUTPUT_DIR = "audio_outputs"


def load_model():
    """Load the Dia model on startup"""
    global model
    
    if model is not None:
        return
    
    print("Loading Dia model...")
    try:
        # Determine device
        if torch.cuda.is_available():
            device = torch.device("cuda")
            compute_dtype = "float16"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps") 
            compute_dtype = "float16"
        else:
            device = torch.device("cpu")
            compute_dtype = "float32"
        
        print(f"Using device: {device}, compute_dtype: {compute_dtype}")
        
        model = Dia.from_pretrained(
            "nari-labs/Dia-1.6B", 
            compute_dtype=compute_dtype,
            device=device
        )
        print("Dia model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading Dia model: {e}")
        raise RuntimeError(f"Failed to load Dia model: {e}")


def preprocess_text(text: str, voice: str) -> str:
    """Preprocess text for Dia model requirements"""
    # Remove asterisks (common in SillyTavern)
    text = re.sub(r'\*+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Ensure text has speaker tags for Dia
    if not ('[S1]' in text or '[S2]' in text):
        # Use voice mapping to determine primary speaker
        voice_config = VOICE_MAPPING.get(voice, VOICE_MAPPING["alloy"])
        primary_speaker = voice_config["primary_speaker"]
        
        # Add speaker tag at the beginning
        text = f"[{primary_speaker}] {text}"
    
    return text


def can_use_torch_compile() -> bool:
    """Check if torch.compile can be used safely"""
    try:
        # Only try torch.compile on CUDA with proper compiler setup
        if not torch.cuda.is_available():
            return False
        
        # Check if we're on Windows and don't have proper compiler
        import platform
        if platform.system() == "Windows":
            # On Windows, torch.compile often fails without proper MSVC setup
            return False
        
        # Try a simple compilation test
        @torch.compile
        def test_fn(x):
            return x + 1
        
        test_tensor = torch.tensor([1.0])
        test_fn(test_tensor)
        return True
    except Exception:
        return False


def ensure_output_dir():
    """Ensure output directory exists"""
    if SERVER_CONFIG.save_outputs and not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)


def cleanup_old_files():
    """Remove files older than retention period"""
    if not SERVER_CONFIG.save_outputs or not os.path.exists(OUTPUT_DIR):
        return
    
    cutoff_time = datetime.now() - timedelta(hours=SERVER_CONFIG.output_retention_hours)
    
    # Clean up files
    for filename in os.listdir(OUTPUT_DIR):
        file_path = os.path.join(OUTPUT_DIR, filename)
        if os.path.isfile(file_path):
            file_time = datetime.fromtimestamp(os.path.getctime(file_path))
            if file_time < cutoff_time:
                try:
                    os.remove(file_path)
                    print(f"Cleaned up old file: {filename}")
                except Exception as e:
                    print(f"Failed to clean up {filename}: {e}")
    
    # Clean up logs for deleted files
    logs_to_remove = []
    for log_id, log in GENERATION_LOGS.items():
        if log.file_path and not os.path.exists(log.file_path):
            logs_to_remove.append(log_id)
    
    for log_id in logs_to_remove:
        del GENERATION_LOGS[log_id]


def generate_audio_from_text(text: str, voice: str = "alloy", speed: float = 1.0) -> tuple[np.ndarray, str]:
    """Generate audio using Dia model and return (audio, log_id)"""
    if model is None:
        raise RuntimeError("Model not loaded")
    
    start_time = time.time()
    log_id = str(uuid.uuid4())
    
    # Preprocess text
    processed_text = preprocess_text(text, voice)
    
    # Get audio prompt if available
    voice_config = VOICE_MAPPING.get(voice, VOICE_MAPPING["alloy"])
    audio_prompt = None
    audio_prompt_used = False
    if voice_config.get("audio_prompt"):
        audio_prompt_data = AUDIO_PROMPTS.get(voice_config["audio_prompt"])
        if audio_prompt_data is not None:
            # Convert numpy array to torch tensor for Dia model
            audio_prompt = torch.from_numpy(audio_prompt_data).float()
            if hasattr(model, 'device'):
                audio_prompt = audio_prompt.to(model.device)
            audio_prompt_used = True
    
    # Debug logging
    if SERVER_CONFIG.debug_mode or SERVER_CONFIG.show_prompts:
        print(f"\n=== Generation Request ===")
        print(f"ID: {log_id}")
        print(f"Original text: {text}")
        print(f"Processed text: {processed_text}")
        print(f"Voice: {voice}")
        print(f"Audio prompt used: {audio_prompt_used}")
        print(f"Speed: {speed}")
    
    # Generate audio
    try:
        # Check if torch.compile is safe to use
        use_compile = can_use_torch_compile()
        
        # Generate with proper audio prompt handling
        audio_output = model.generate(
            processed_text,
            audio_prompt=audio_prompt,  # Dia will handle the batching internally
            use_torch_compile=use_compile,
            temperature=1.2,
            cfg_scale=3.0,
            top_p=0.95,
            verbose=SERVER_CONFIG.debug_mode
        )
        
        # Apply speed adjustment if needed
        if speed != 1.0 and audio_output is not None:
            # Simple speed adjustment by resampling
            original_len = len(audio_output)
            target_len = int(original_len / speed)
            if target_len > 0:
                x_original = np.arange(original_len)
                x_resampled = np.linspace(0, original_len - 1, target_len)
                audio_output = np.interp(x_resampled, x_original, audio_output)
        
        generation_time = time.time() - start_time
        
        # Save output file if enabled
        file_path = None
        file_size = None
        if SERVER_CONFIG.save_outputs and audio_output is not None:
            ensure_output_dir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{log_id[:8]}_{voice}.wav"
            file_path = os.path.join(OUTPUT_DIR, filename)
            
            # Save audio file
            sf.write(file_path, audio_output, 44100, format='WAV', subtype='PCM_16')
            file_size = os.path.getsize(file_path)
        
        # Create log entry
        log_entry = GenerationLog(
            id=log_id,
            timestamp=datetime.now(),
            text=text,
            processed_text=processed_text,
            voice=voice,
            audio_prompt_used=audio_prompt_used,
            generation_time=generation_time,
            file_path=file_path,
            file_size=file_size
        )
        GENERATION_LOGS[log_id] = log_entry
        
        if SERVER_CONFIG.debug_mode:
            print(f"Generation completed in {generation_time:.2f}s")
            if file_path:
                print(f"Saved to: {file_path}")
        
        return audio_output, log_id
        
    except Exception as e:
        # If torch.compile fails, try again without it
        if "Compiler:" in str(e) and "not found" in str(e):
            print("Torch compile failed, retrying without compilation...")
            try:
                audio_output = model.generate(
                    processed_text,
                    audio_prompt=audio_prompt,
                    use_torch_compile=False,  # Disable compilation
                    temperature=1.2,
                    cfg_scale=3.0,
                    top_p=0.95,
                    verbose=SERVER_CONFIG.debug_mode
                )
                
                # Apply speed adjustment if needed
                if speed != 1.0 and audio_output is not None:
                    original_len = len(audio_output)
                    target_len = int(original_len / speed)
                    if target_len > 0:
                        x_original = np.arange(original_len)
                        x_resampled = np.linspace(0, original_len - 1, target_len)
                        audio_output = np.interp(x_resampled, x_original, audio_output)
                
                generation_time = time.time() - start_time
                
                # Save and log as above
                file_path = None
                file_size = None
                if SERVER_CONFIG.save_outputs and audio_output is not None:
                    ensure_output_dir()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{timestamp}_{log_id[:8]}_{voice}.wav"
                    file_path = os.path.join(OUTPUT_DIR, filename)
                    sf.write(file_path, audio_output, 44100, format='WAV', subtype='PCM_16')
                    file_size = os.path.getsize(file_path)
                
                log_entry = GenerationLog(
                    id=log_id,
                    timestamp=datetime.now(),
                    text=text,
                    processed_text=processed_text,
                    voice=voice,
                    audio_prompt_used=audio_prompt_used,
                    generation_time=generation_time,
                    file_path=file_path,
                    file_size=file_size
                )
                GENERATION_LOGS[log_id] = log_entry
                
                return audio_output, log_id
            except Exception as retry_e:
                print(f"Retry without compilation also failed: {retry_e}")
                raise HTTPException(status_code=500, detail=f"Audio generation failed: {retry_e}")
        
        print(f"Error generating audio: {e}")
        print(f"Text: {processed_text}")
        print(f"Voice: {voice}")
        print(f"Audio prompt available: {audio_prompt is not None}")
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {e}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()
    
    # Start cleanup task
    def cleanup_task():
        while True:
            cleanup_old_files()
            time.sleep(3600)  # Run every hour
    
    cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
    cleanup_thread.start()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Dia TTS Server is running", "status": "healthy"}


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI compatible)"""
    return ModelsResponse(
        data=[
            ModelInfo(
                id="dia",
                created=int(time.time()),
                owned_by="nari-labs"
            ),
            ModelInfo(
                id="tts-1",
                created=int(time.time()),
                owned_by="nari-labs"
            ),
            ModelInfo(
                id="tts-1-hd",
                created=int(time.time()),
                owned_by="nari-labs"
            )
        ]
    )


@app.get("/v1/voices")
async def list_voices():
    """List available voices (OpenAI compatible)"""
    voices = []
    for voice_id, config in VOICE_MAPPING.items():
        voices.append(VoiceInfo(
            name=voice_id,
            voice_id=voice_id,
            preview_url=f"/preview/{voice_id}"
        ))
    return voices


@app.get("/voices")
async def list_voices_alt():
    """Alternative voices endpoint"""
    return await list_voices()


@app.post("/v1/audio/speech")
async def generate_speech(
    request: TTSRequest, 
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Main TTS endpoint (OpenAI compatible)"""
    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    
    try:
        # Generate audio
        audio_data, log_id = generate_audio_from_text(
            request.input, 
            request.voice, 
            request.speed
        )
        
        if audio_data is None:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
        
        # Convert to bytes and create streaming response
        def generate_audio_stream():
            with io.BytesIO() as buffer:
                # Ensure audio is in the right format
                if audio_data.dtype != np.float32:
                    audio_data_processed = audio_data.astype(np.float32)
                else:
                    audio_data_processed = audio_data
                
                # Write audio file
                sf.write(buffer, audio_data_processed, 44100, format='WAV', subtype='PCM_16')
                buffer.seek(0)
                
                # Stream in chunks
                chunk_size = 8192
                while True:
                    chunk = buffer.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
        
        # Determine media type and filename
        if request.response_format.lower() == "mp3":
            media_type = "audio/wav"  # Still return WAV for now
            filename = "speech.wav"
        else:
            media_type = "audio/wav"
            filename = "speech.wav"
        
        response_headers = {
            "Content-Disposition": f"attachment; filename={filename}",
            "Transfer-Encoding": "chunked"
        }
        
        # Add log ID header if debug mode
        if SERVER_CONFIG.debug_mode:
            response_headers["X-Generation-ID"] = log_id
        
        return StreamingResponse(
            generate_audio_stream(),
            media_type=media_type,
            headers=response_headers
        )
        
    except Exception as e:
        print(f"Error in generate_speech: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tts/generate")
async def generate_speech_alt(request: TTSGenerateRequest):
    """Alternative TTS endpoint for SillyTavern-Extras compatibility"""
    tts_request = TTSRequest(
        input=request.text,
        voice=request.speaker
    )
    return await generate_speech(tts_request)


@app.get("/api/tts/speakers")
async def list_speakers():
    """List speakers (alternative format)"""
    return list(VOICE_MAPPING.keys())


@app.get("/preview/{voice_id}")
async def get_voice_preview(voice_id: str):
    """Generate a preview sample for a voice"""
    if voice_id not in VOICE_MAPPING:
        raise HTTPException(status_code=404, detail="Voice not found")
    
    preview_text = f"[S1] Hello, this is a preview of the {voice_id} voice. [S2] How does this sound to you?"
    
    try:
        audio_data = generate_audio_from_text(preview_text, voice_id)
        
        with io.BytesIO() as buffer:
            sf.write(buffer, audio_data, 44100, format='WAV', subtype='PCM_16')
            buffer.seek(0)
            audio_bytes = buffer.getvalue()
        
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={"Content-Disposition": f"attachment; filename=preview_{voice_id}.wav"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": time.time()
    }


# Voice Management Endpoints

@app.get("/v1/voice_mappings")
async def get_voice_mappings():
    """Get current voice mappings"""
    return VOICE_MAPPING


@app.put("/v1/voice_mappings/{voice_id}")
async def update_voice_mapping(voice_id: str, update: VoiceMappingUpdate):
    """Update voice mapping configuration"""
    if voice_id not in VOICE_MAPPING:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
    
    # Update voice configuration
    if update.style is not None:
        VOICE_MAPPING[voice_id]["style"] = update.style
    if update.primary_speaker is not None:
        VOICE_MAPPING[voice_id]["primary_speaker"] = update.primary_speaker
    if update.audio_prompt is not None:
        VOICE_MAPPING[voice_id]["audio_prompt"] = update.audio_prompt
    
    return {"message": f"Voice '{voice_id}' updated successfully", "voice_config": VOICE_MAPPING[voice_id]}


@app.post("/v1/voice_mappings")
async def create_voice_mapping(mapping: VoiceMappingUpdate):
    """Create new voice mapping"""
    if not mapping.voice_id:
        raise HTTPException(status_code=400, detail="voice_id is required")
    
    VOICE_MAPPING[mapping.voice_id] = {
        "style": mapping.style or "neutral",
        "primary_speaker": mapping.primary_speaker or "S1",
        "audio_prompt": mapping.audio_prompt
    }
    
    return {"message": f"Voice '{mapping.voice_id}' created successfully", "voice_config": VOICE_MAPPING[mapping.voice_id]}


@app.delete("/v1/voice_mappings/{voice_id}")
async def delete_voice_mapping(voice_id: str):
    """Delete voice mapping (only custom voices, not defaults)"""
    default_voices = {"alloy", "echo", "fable", "nova", "onyx", "shimmer"}
    
    if voice_id in default_voices:
        raise HTTPException(status_code=400, detail=f"Cannot delete default voice '{voice_id}'")
    
    if voice_id not in VOICE_MAPPING:
        raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")
    
    # Remove associated audio prompt
    if VOICE_MAPPING[voice_id].get("audio_prompt"):
        audio_prompt_id = VOICE_MAPPING[voice_id]["audio_prompt"]
        AUDIO_PROMPTS.pop(audio_prompt_id, None)
    
    del VOICE_MAPPING[voice_id]
    return {"message": f"Voice '{voice_id}' deleted successfully"}


@app.post("/v1/audio_prompts/upload")
async def upload_audio_prompt(
    prompt_id: str = Form(...),
    audio_file: UploadFile = File(...)
):
    """Upload audio file to use as voice prompt"""
    # Basic validation
    if not prompt_id or not prompt_id.strip():
        raise HTTPException(status_code=400, detail="prompt_id cannot be empty")
    
    if not audio_file or not audio_file.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # Content type validation (more flexible)
    valid_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
    file_ext = os.path.splitext(audio_file.filename.lower())[1]
    
    if file_ext not in valid_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported: {', '.join(valid_extensions)}"
        )
    
    temp_file_path = None
    try:
        # Read audio file data
        audio_data = await audio_file.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Audio file is empty")
        
        # Create temporary file with proper extension
        temp_file_fd, temp_file_path = tempfile.mkstemp(suffix=file_ext)
        
        try:
            # Write data to temp file
            with os.fdopen(temp_file_fd, 'wb') as temp_file:
                temp_file.write(audio_data)
                temp_file.flush()
                os.fsync(temp_file.fileno())  # Ensure data is written to disk
            
            # Load audio with error handling
            try:
                audio_array, sample_rate = sf.read(temp_file_path)
            except Exception as sf_error:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Cannot read audio file. Please check the file format: {str(sf_error)}"
                )
            
            # Validate audio data
            if len(audio_array) == 0:
                raise HTTPException(status_code=400, detail="Audio file contains no audio data")
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Validate duration (3-30 seconds recommended)
            duration = len(audio_array) / sample_rate
            if duration < 0.5:
                raise HTTPException(status_code=400, detail="Audio file too short (minimum 0.5 seconds)")
            if duration > 60:
                raise HTTPException(status_code=400, detail="Audio file too long (maximum 60 seconds)")
            
            # Resample to 44.1kHz if needed
            if sample_rate != 44100:
                resample_ratio = 44100 / sample_rate
                new_length = int(len(audio_array) * resample_ratio)
                if new_length > 0:
                    audio_array = np.interp(
                        np.linspace(0, len(audio_array) - 1, new_length),
                        np.arange(len(audio_array)),
                        audio_array
                    )
            
            # Normalize audio to prevent clipping
            if np.max(np.abs(audio_array)) > 0:
                audio_array = audio_array / np.max(np.abs(audio_array)) * 0.95
            
            # Store the audio prompt
            AUDIO_PROMPTS[prompt_id] = audio_array.astype(np.float32)
            
            return {
                "message": f"Audio prompt '{prompt_id}' uploaded successfully",
                "duration": len(audio_array) / 44100,
                "sample_rate": 44100,
                "original_sample_rate": sample_rate,
                "channels": "mono"
            }
            
        except HTTPException:
            raise
        except Exception as process_error:
            raise HTTPException(
                status_code=500, 
                detail=f"Error processing audio file: {str(process_error)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error during file upload: {str(e)}"
        )
    finally:
        # Always clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                print(f"Warning: Failed to clean up temp file {temp_file_path}: {cleanup_error}")


@app.get("/v1/audio_prompts")
async def list_audio_prompts():
    """List available audio prompts"""
    prompts = {}
    for prompt_id, audio_data in AUDIO_PROMPTS.items():
        prompts[prompt_id] = {
            "duration": len(audio_data) / 44100,
            "sample_rate": 44100
        }
    return prompts


@app.delete("/v1/audio_prompts/{prompt_id}")
async def delete_audio_prompt(prompt_id: str):
    """Delete audio prompt"""
    if prompt_id not in AUDIO_PROMPTS:
        raise HTTPException(status_code=404, detail=f"Audio prompt '{prompt_id}' not found")
    
    # Check if any voices are using this prompt
    using_voices = [voice_id for voice_id, config in VOICE_MAPPING.items() 
                   if config.get("audio_prompt") == prompt_id]
    
    if using_voices:
        return {
            "warning": f"Audio prompt '{prompt_id}' is used by voices: {using_voices}",
            "message": "Remove from voice mappings first before deleting"
        }
    
    del AUDIO_PROMPTS[prompt_id]
    return {"message": f"Audio prompt '{prompt_id}' deleted successfully"}


# Debug and Configuration Endpoints

@app.get("/v1/config")
async def get_server_config():
    """Get current server configuration"""
    return SERVER_CONFIG


@app.put("/v1/config")
async def update_server_config(config: ServerConfig):
    """Update server configuration"""
    global SERVER_CONFIG
    SERVER_CONFIG = config
    return {"message": "Configuration updated successfully", "config": SERVER_CONFIG}


@app.get("/v1/logs")
async def get_generation_logs(
    limit: int = Query(default=50, le=500),
    voice: Optional[str] = Query(default=None)
):
    """Get generation logs"""
    logs = list(GENERATION_LOGS.values())
    
    # Filter by voice if specified
    if voice:
        logs = [log for log in logs if log.voice == voice]
    
    # Sort by timestamp (newest first)
    logs.sort(key=lambda x: x.timestamp, reverse=True)
    
    # Limit results
    logs = logs[:limit]
    
    return {
        "logs": logs,
        "total": len(GENERATION_LOGS),
        "filtered": len(logs)
    }


@app.get("/v1/logs/{log_id}")
async def get_generation_log(log_id: str):
    """Get specific generation log"""
    if log_id not in GENERATION_LOGS:
        raise HTTPException(status_code=404, detail=f"Log '{log_id}' not found")
    
    return GENERATION_LOGS[log_id]


@app.get("/v1/logs/{log_id}/download")
async def download_generation_output(log_id: str):
    """Download the audio file for a specific generation"""
    if log_id not in GENERATION_LOGS:
        raise HTTPException(status_code=404, detail=f"Log '{log_id}' not found")
    
    log = GENERATION_LOGS[log_id]
    if not log.file_path or not os.path.exists(log.file_path):
        raise HTTPException(status_code=404, detail="Audio file not found or has been cleaned up")
    
    filename = os.path.basename(log.file_path)
    return FileResponse(
        log.file_path,
        media_type="audio/wav",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.delete("/v1/logs")
async def clear_generation_logs():
    """Clear all generation logs"""
    global GENERATION_LOGS
    GENERATION_LOGS = {}
    return {"message": "All generation logs cleared"}


@app.delete("/v1/logs/{log_id}")
async def delete_generation_log(log_id: str):
    """Delete specific generation log and its file"""
    if log_id not in GENERATION_LOGS:
        raise HTTPException(status_code=404, detail=f"Log '{log_id}' not found")
    
    log = GENERATION_LOGS[log_id]
    
    # Delete file if it exists
    if log.file_path and os.path.exists(log.file_path):
        try:
            os.remove(log.file_path)
        except Exception as e:
            print(f"Failed to delete file {log.file_path}: {e}")
    
    # Delete log
    del GENERATION_LOGS[log_id]
    
    return {"message": f"Log '{log_id}' and associated file deleted"}


@app.post("/v1/cleanup")
async def manual_cleanup():
    """Manually trigger cleanup of old files"""
    cleanup_old_files()
    return {"message": "Cleanup completed"}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Dia TTS FastAPI server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--save-outputs", action="store_true", help="Save audio outputs to files")
    parser.add_argument("--show-prompts", action="store_true", help="Show prompts in console")
    parser.add_argument("--retention-hours", type=int, default=24, help="File retention hours")
    
    args = parser.parse_args()
    
    # Update server config from command line args
    if args.debug:
        SERVER_CONFIG.debug_mode = True
    if args.save_outputs:
        SERVER_CONFIG.save_outputs = True
    if args.show_prompts:
        SERVER_CONFIG.show_prompts = True
    SERVER_CONFIG.output_retention_hours = args.retention_hours
    
    print(f"Starting Dia TTS Server on {args.host}:{args.port}")
    print("Make sure you have set the HF_TOKEN environment variable!")
    print(f"SillyTavern endpoint: http://{args.host}:{args.port}/v1/audio/speech")
    print(f"Configuration API: http://{args.host}:{args.port}/v1/config")
    print(f"Generation logs: http://{args.host}:{args.port}/v1/logs")
    print(f"Debug mode: {SERVER_CONFIG.debug_mode}")
    print(f"Save outputs: {SERVER_CONFIG.save_outputs}")
    print(f"Show prompts: {SERVER_CONFIG.show_prompts}")
    print(f"Retention: {SERVER_CONFIG.output_retention_hours} hours")
    
    uvicorn.run(
        "fastapi_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
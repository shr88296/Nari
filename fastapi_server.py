"""
FastAPI Server for Dia TTS Model - SillyTavern Compatible
"""

import io
import os
import re
import tempfile
import time
import json
from typing import Optional, Dict, Any

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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


def generate_audio_from_text(text: str, voice: str = "alloy", speed: float = 1.0) -> np.ndarray:
    """Generate audio using Dia model"""
    if model is None:
        raise RuntimeError("Model not loaded")
    
    # Preprocess text
    processed_text = preprocess_text(text, voice)
    
    # Get audio prompt if available
    voice_config = VOICE_MAPPING.get(voice, VOICE_MAPPING["alloy"])
    audio_prompt = None
    if voice_config.get("audio_prompt"):
        audio_prompt = AUDIO_PROMPTS.get(voice_config["audio_prompt"])
    
    # Generate audio
    try:
        # Use faster settings for API server
        use_compile = torch.cuda.is_available()  # Only use compile on CUDA
        
        audio_output = model.generate(
            processed_text,
            audio_prompt=audio_prompt,
            use_torch_compile=use_compile,
            temperature=1.2,
            cfg_scale=3.0,
            top_p=0.95,
            verbose=False
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
        
        return audio_output
        
    except Exception as e:
        print(f"Error generating audio: {e}")
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {e}")


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


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
        audio_data = generate_audio_from_text(
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
        
        return StreamingResponse(
            generate_audio_stream(),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Transfer-Encoding": "chunked"
            }
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
    if not audio_file.content_type or not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    try:
        # Read audio file
        audio_data = await audio_file.read()
        
        # Save to temporary file and load with soundfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_data)
            temp_file.flush()
            
            # Load audio
            audio_array, sample_rate = sf.read(temp_file.name)
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            # Resample to 44.1kHz if needed (Dia's expected sample rate)
            if sample_rate != 44100:
                # Simple resampling - in production you'd want proper resampling
                resample_ratio = 44100 / sample_rate
                new_length = int(len(audio_array) * resample_ratio)
                audio_array = np.interp(
                    np.linspace(0, len(audio_array) - 1, new_length),
                    np.arange(len(audio_array)),
                    audio_array
                )
            
            # Store the audio prompt
            AUDIO_PROMPTS[prompt_id] = audio_array.astype(np.float32)
            
            # Clean up temp file
            os.unlink(temp_file.name)
        
        return {
            "message": f"Audio prompt '{prompt_id}' uploaded successfully",
            "duration": len(audio_array) / 44100,
            "sample_rate": 44100
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process audio file: {e}")


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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Dia TTS FastAPI server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print(f"Starting Dia TTS Server on {args.host}:{args.port}")
    print("Make sure you have set the HF_TOKEN environment variable!")
    print(f"SillyTavern endpoint: http://{args.host}:{args.port}/v1/audio/speech")
    
    uvicorn.run(
        "fastapi_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
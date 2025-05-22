# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dia is a 1.6B parameter text-to-speech model that directly generates realistic dialogue from text transcripts. The model supports emotion/tone control through audio prompts and can produce nonverbal communications like laughter, coughing, etc.

## Development Commands

### Environment Setup
```bash
# Using uv (recommended)
uv run app.py

# Or traditional Python
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### Running the Application
```bash
# Start Gradio UI
python app.py
# With custom device: python app.py --device cuda
# With sharing: python app.py --share

# CLI usage
python cli.py "your text here" --output output.wav

# FastAPI Server (SillyTavern compatible)
python fastapi_server.py
# With custom host/port: python fastapi_server.py --host 0.0.0.0 --port 8000
# Development mode: python fastapi_server.py --reload
```

### Code Quality
```bash
# Linting (configured in pyproject.toml)
ruff check
ruff format

# The project uses ruff for linting with custom rules:
# - Ignores line length violations (E501)
# - Ignores complexity (C901), naming (E741), regex (W605)
# - Line length set to 119 characters
```

## Architecture Overview

### Core Components

1. **DiaModel** (`dia/layers.py`): Main transformer architecture with encoder-decoder structure
   - Encoder: Processes text input with standard transformer layers
   - Decoder: Generates audio tokens using grouped-query attention and cross-attention to encoder

2. **Audio Processing** (`dia/audio.py`): Handles delay patterns for multi-channel audio generation
   - `apply_audio_delay()`: Applies channel-specific delays during generation
   - `revert_audio_delay()`: Reverts delays for final audio reconstruction

3. **Configuration System** (`dia/config.py`): Pydantic-based config management
   - DataConfig: Text/audio lengths, padding, delay patterns
   - ModelConfig: Architecture parameters (layers, heads, dimensions)
   - DiaConfig: Master configuration combining all components

4. **State Management** (`dia/state.py`): Handles inference state for encoder/decoder
   - EncoderInferenceState: Manages text encoding and padding
   - DecoderInferenceState: Manages KV caches and cross-attention during generation
   - DecoderOutput: Tracks generated tokens and manages prefill/generation phases

### Generation Process

1. **Text Encoding**: Text is byte-encoded with special tokens [S1]/[S2] replaced by \x01/\x02
2. **Audio Prompt Processing**: Optional audio prompts are encoded using DAC (Descript Audio Codec)
3. **Dual Path Generation**: Uses classifier-free guidance with conditional/unconditional paths
4. **Delay Pattern Application**: Multi-channel audio uses staggered generation with channel-specific delays
5. **Token Sampling**: Supports temperature, top-p, and top-k sampling with EOS handling
6. **Audio Reconstruction**: Generated codes are converted back to waveforms via DAC decoder

### Key Design Patterns

- **Batched Generation**: Supports generating multiple audio sequences simultaneously
- **Torch Compilation**: Optional torch.compile support for faster inference (use_torch_compile=True)
- **Device Flexibility**: Auto-detects CUDA, MPS (Apple Silicon), or CPU
- **Memory Efficiency**: Uses grouped-query attention and optional float16/bfloat16 precision

## Text Format Requirements

- Always start with `[S1]` and alternate between `[S1]` and `[S2]` speakers
- Keep input length moderate (5-20 seconds of audio equivalent)
- Supported nonverbals: `(laughs)`, `(coughs)`, `(clears throat)`, `(sighs)`, `(gasps)`, etc.
- For voice cloning: provide transcript of source audio before generation text

## Model Loading Patterns

```python
# From Hugging Face Hub (default)
model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16")

# From local files
model = Dia.from_local("config.json", "model.pth", compute_dtype="float16")

# With custom device
model = Dia.from_pretrained("nari-labs/Dia-1.6B", device=torch.device("cuda:0"))
```

## Environment Variables

- `HF_TOKEN`: Required for downloading models from Hugging Face Hub
- `GRADIO_SERVER_NAME`: Override Gradio server host (use "0.0.0.0" for Docker)
- `GRADIO_SERVER_PORT`: Override Gradio server port

## Hardware Requirements

- GPU: ~10GB VRAM for float16/bfloat16, ~13GB for float32
- CPU: Supported but significantly slower
- Apple Silicon: Use `use_torch_compile=False` (MPS doesn't support torch.compile)

## FastAPI Server (SillyTavern Integration)

The `fastapi_server.py` provides OpenAI-compatible TTS API endpoints for integration with SillyTavern and other applications.

### Key Endpoints
- `POST /v1/audio/speech` - Main TTS generation (OpenAI compatible)
- `GET /v1/voices` - List available voices
- `POST /api/tts/generate` - Alternative TTS endpoint (SillyTavern-Extras style)
- `GET /health` - Server health check

### SillyTavern Configuration

**Setup Instructions:**

1. **Start the FastAPI Server:**
   ```bash
   python fastapi_server.py
   # Or with custom settings:
   python fastapi_server.py --host 0.0.0.0 --port 8000
   ```

2. **Configure SillyTavern:**
   - Navigate to Settings → Text-to-Speech
   - Set TTS Provider: **OpenAI Compatible**
   - Model: **dia** (or tts-1, tts-1-hd)
   - API Key: **sk-anything** (any value starting with "sk-")
   - Endpoint URL: **http://localhost:7860/v1/audio/speech**
   - Voice: Choose from alloy, echo, fable, nova, onyx, shimmer

3. **Available Endpoints:**
   - `GET /v1/models` - List available models
   - `POST /v1/audio/speech` - Main TTS endpoint
   - `GET /v1/voices` - List available voices
   - `GET /health` - Server health check

**Troubleshooting:**
- Ensure HF_TOKEN environment variable is set
- Check server logs for model loading status
- Test with: `curl http://localhost:7860/health`

## Voice Cloning and Custom Voices

The FastAPI server supports custom voice creation through audio prompts:

### Upload Audio Prompt
```bash
curl -X POST "http://localhost:7860/v1/audio_prompts/upload" \
  -F "prompt_id=my_voice" \
  -F "audio_file=@voice_sample.wav"
```

### Create Custom Voice Mapping
```bash
curl -X POST "http://localhost:7860/v1/voice_mappings" \
  -H "Content-Type: application/json" \
  -d '{
    "voice_id": "custom_voice",
    "style": "friendly",
    "primary_speaker": "S1",
    "audio_prompt": "my_voice"
  }'
```

### Voice Management Endpoints
- `GET /v1/voice_mappings` - List all voice configurations
- `PUT /v1/voice_mappings/{voice_id}` - Update voice configuration
- `POST /v1/voice_mappings` - Create new voice mapping
- `DELETE /v1/voice_mappings/{voice_id}` - Delete custom voice
- `POST /v1/audio_prompts/upload` - Upload audio prompt file
- `GET /v1/audio_prompts` - List uploaded audio prompts
- `DELETE /v1/audio_prompts/{prompt_id}` - Delete audio prompt

### Audio Prompt Requirements
- Supported formats: WAV, MP3, M4A, etc.
- Automatically resampled to 44.1kHz mono
- Recommended: 3-10 seconds of clean speech
- For best results: Include the speaker's voice saying the target text or similar content

## Debug and Logging Features

The FastAPI server includes comprehensive logging and debugging capabilities:

### Command Line Options
```bash
# Enable debug mode with file saving
python fastapi_server.py --debug --save-outputs --show-prompts

# Custom retention period
python fastapi_server.py --save-outputs --retention-hours 48

# Development mode with all features
python fastapi_server.py --debug --save-outputs --show-prompts --reload
```

### Configuration API
```bash
# Get current configuration
curl http://localhost:7860/v1/config

# Update configuration
curl -X PUT "http://localhost:7860/v1/config" \
  -H "Content-Type: application/json" \
  -d '{
    "debug_mode": true,
    "save_outputs": true,
    "show_prompts": true,
    "output_retention_hours": 24
  }'
```

### Generation Logs API
```bash
# Get recent generation logs
curl http://localhost:7860/v1/logs

# Get logs for specific voice
curl "http://localhost:7860/v1/logs?voice=alloy&limit=10"

# Get specific log details
curl http://localhost:7860/v1/logs/{log_id}

# Download generated audio file
curl http://localhost:7860/v1/logs/{log_id}/download -o output.wav

# Clear all logs
curl -X DELETE http://localhost:7860/v1/logs

# Manual cleanup of old files
curl -X POST http://localhost:7860/v1/cleanup
```

### Features
- **Prompt Logging**: Shows original and processed text for each request
- **Audio File Saving**: Saves all generated audio with timestamps
- **Generation Metrics**: Tracks generation time and file sizes
- **Automatic Cleanup**: Removes files older than retention period (default 24h)
- **Debug Headers**: Includes generation ID in response headers
- **Voice Tracking**: Logs which voice and audio prompts were used

### Voice Mapping
The server maps common voice names to Dia's speaker system:
- `alloy`, `nova`, `shimmer` → Primarily use [S1] speaker
- `echo`, `fable`, `onyx` → Primarily use [S2] speaker

### API Request Format
```json
{
  "model": "dia",
  "input": "Text to convert to speech",
  "voice": "alloy",
  "response_format": "wav",
  "speed": 1.0
}
```

## Docker Support

```bash
# GPU build
docker build -f docker/Dockerfile.gpu -t dia-gpu .

# CPU build  
docker build -f docker/Dockerfile.cpu -t dia-cpu .
```
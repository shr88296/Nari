# Dia TTS API Integration Guide

## Overview

Dia TTS FastAPI server provides OpenAI-compatible text-to-speech API with voice cloning capabilities. This document covers integration patterns for SillyTavern and custom applications.

## Quick Start

### 1. Start the Server
```bash
python fastapi_server.py --host 0.0.0.0 --port 8000
```

### 2. Basic TTS Request
```bash
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-anything" \
  -d '{
    "model": "dia",
    "input": "[S1] Hello, this is a test message.",
    "voice": "alloy",
    "response_format": "wav"
  }' \
  --output speech.wav
```

## Core API Endpoints

### Text-to-Speech
- **POST** `/v1/audio/speech` - Generate speech from text
- **GET** `/v1/models` - List available models
- **GET** `/v1/voices` - List available voices

### Voice Management
- **GET** `/v1/voice_mappings` - List voice configurations
- **POST** `/v1/voice_mappings` - Create custom voice
- **PUT** `/v1/voice_mappings/{voice_id}` - Update voice
- **DELETE** `/v1/voice_mappings/{voice_id}` - Delete custom voice

### Audio Prompts
- **POST** `/v1/audio_prompts/upload` - Upload voice sample
- **GET** `/v1/audio_prompts` - List uploaded prompts
- **DELETE** `/v1/audio_prompts/{prompt_id}` - Delete prompt

## SillyTavern Integration

### Configuration
1. Navigate to **Settings → Text-to-Speech**
2. Set **TTS Provider**: `OpenAI Compatible`
3. Set **Model**: `dia`
4. Set **API Key**: `sk-anything`
5. Set **Endpoint URL**: `http://localhost:8000/v1/audio/speech`
6. Choose **Voice**: `alloy`, `echo`, `fable`, `nova`, `onyx`, or `shimmer`

### Custom Voice Setup
```bash
# 1. Upload voice sample
curl -X POST "http://localhost:8000/v1/audio_prompts/upload" \
  -F "prompt_id=character_voice" \
  -F "audio_file=@sample.wav"

# 2. Create voice mapping
curl -X POST "http://localhost:8000/v1/voice_mappings" \
  -H "Content-Type: application/json" \
  -d '{
    "voice_id": "my_character",
    "style": "expressive",
    "primary_speaker": "S1", 
    "audio_prompt": "character_voice"
  }'

# 3. Use in SillyTavern by setting Voice: "my_character"
```

## Request/Response Formats

### TTS Request
```json
{
  "model": "dia",
  "input": "[S1] Your text here",
  "voice": "alloy",
  "response_format": "wav",
  "speed": 1.0
}
```

### Voice Mapping
```json
{
  "voice_id": "custom_voice",
  "style": "neutral",
  "primary_speaker": "S1",
  "audio_prompt": "prompt_id"
}
```

## Text Format Guidelines

### Speaker Tags
- Always start with `[S1]` or `[S2]`
- Alternate speakers: `[S1] Hello! [S2] Hi there!`
- Single speaker: `[S1] This is a monologue.`

### Nonverbal Sounds
- `(laughs)`, `(coughs)`, `(sighs)`, `(gasps)`
- `(clears throat)`, `(whispers)`, `(shouts)`

### Best Practices
- Keep input under 4096 characters
- Use clear, natural speech patterns
- For voice cloning: provide 3-10 seconds of clean audio

## Error Handling

### Common Status Codes
- `400` - Invalid request format
- `404` - Voice/prompt not found
- `500` - Generation or server error

### Example Error Response
```json
{
  "detail": "Voice 'unknown_voice' not found"
}
```

## Voice Cloning Workflow

### 1. Prepare Audio Sample
- Format: WAV, MP3, M4A (auto-converted)
- Duration: 3-10 seconds recommended
- Quality: Clear speech, minimal background noise

### 2. Upload and Configure
```python
import requests

# Upload audio prompt
with open("voice_sample.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/v1/audio_prompts/upload",
        data={"prompt_id": "my_voice"},
        files={"audio_file": f}
    )

# Create voice mapping
requests.post(
    "http://localhost:8000/v1/voice_mappings",
    json={
        "voice_id": "cloned_voice",
        "style": "natural",
        "primary_speaker": "S1",
        "audio_prompt": "my_voice"
    }
)
```

### 3. Generate Speech
```python
response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "dia",
        "input": "[S1] Hello, this is my cloned voice!",
        "voice": "cloned_voice"
    }
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

## Integration Examples

### Python Client
```python
import requests

class DiaTTSClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def generate_speech(self, text, voice="alloy"):
        response = requests.post(
            f"{self.base_url}/v1/audio/speech",
            json={
                "model": "dia",
                "input": text,
                "voice": voice
            }
        )
        return response.content
    
    def upload_voice(self, prompt_id, audio_file_path):
        with open(audio_file_path, "rb") as f:
            return requests.post(
                f"{self.base_url}/v1/audio_prompts/upload",
                data={"prompt_id": prompt_id},
                files={"audio_file": f}
            ).json()

# Usage
client = DiaTTSClient()
audio = client.generate_speech("[S1] Hello world!", "alloy")
```

### JavaScript/Node.js
```javascript
const FormData = require('form-data');
const fs = require('fs');

class DiaTTSClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async generateSpeech(text, voice = 'alloy') {
        const response = await fetch(`${this.baseUrl}/v1/audio/speech`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model: 'dia',
                input: text,
                voice: voice
            })
        });
        return response.arrayBuffer();
    }
}
```

## Performance Considerations

- **Model Loading**: ~30 seconds on first request
- **Generation Speed**: ~2-5x real-time on GPU
- **Memory Usage**: ~10GB VRAM (GPU) or ~16GB RAM (CPU)
- **Concurrent Requests**: Single model instance (consider load balancing)

## Troubleshooting

### Common Issues
1. **Model not loading**: Check `HF_TOKEN` environment variable
2. **Audio quality**: Ensure input text has proper speaker tags
3. **Slow generation**: Use GPU with CUDA for faster inference
4. **Voice not found**: Check `/v1/voice_mappings` for available voices

### Health Check
```bash
curl http://localhost:8000/health
```

### Logs
Monitor server logs for detailed error information and model loading status.
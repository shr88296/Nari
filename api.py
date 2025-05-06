import time
import logging
import io
import os
import torch
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
except:
    pass

from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import soundfile as sf
import numpy as np

from dia.model import Dia

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app):
    global model
    try:
        logger.info("Starting up: loading model...")
        load_model()
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
    
    yield 
    
    if model is not None:
        logger.info("Shutting down: releasing model resources")
        model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
app = FastAPI(
    title="Dia TTS API",
    description="Speech generation API for Dia text-to-speech model",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
last_request_time = time.time()
cached_results = {}

class TextToSpeechRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    cfg_scale: float = Field(3.0, description="Classifier-free guidance scale (1.0-5.0)")
    temperature: float = Field(1.3, description="Temperature for sampling (0.5-2.0)")
    top_p: float = Field(0.95, description="Top-p sampling threshold (0.5-1.0)")
    use_torch_compile: bool = Field(True, description="Whether to use torch.compile")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

class VoiceCloneRequest(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    cfg_scale: float = Field(3.0, description="Classifier-free guidance scale")

def load_model():
    global model
    if model is None:
        try:
            logger.info("Loading Dia model...")
            start_time = time.time()
            
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if torch.backends.mps.is_available():
                logger.info("MPS backend detected, but using CPU for better compatibility")
                device = "cpu"
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            
            logger.info(f"Using device: {device}")
            model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16", device=device)
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")

@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "healthy", "model": "dia-1.6b"}

@app.post("/generate")
async def generate_speech(request: TextToSpeechRequest):
    global model, last_request_time, cached_results
    
    if model is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")
    
    cache_key = f"{request.text}_{request.cfg_scale}_{request.temperature}_{request.top_p}_{request.seed}"
    if cache_key in cached_results:
        logger.info(f"Returning cached result for: {request.text[:50]}...")
        audio_data = cached_results[cache_key]
    else:
        try:
            if request.seed is not None:
                torch.manual_seed(request.seed)
            
            start_time = time.time()
            logger.info(f"Processing TTS request: '{request.text[:100]}...'")
            
            use_compile = request.use_torch_compile
            if torch.backends.mps.is_available():
                logger.info("MPS backend detected, disabling torch.compile to avoid errors")
                use_compile = False
            
            audio = model.generate(
                text=request.text,
                cfg_scale=request.cfg_scale,
                temperature=request.temperature,
                top_p=request.top_p,
                use_torch_compile=use_compile
            )
            
            buf = io.BytesIO()
            sf.write(buf, audio, 44100, format="WAV")
            buf.seek(0)
            audio_data = buf.getvalue()
            
            MAX_CACHE_SIZE = 100
            if len(cached_results) >= MAX_CACHE_SIZE:
                oldest_key = next(iter(cached_results))
                del cached_results[oldest_key]
            cached_results[cache_key] = audio_data
            
            process_time = time.time() - start_time
            logger.info(f"Generated audio in {process_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    last_request_time = time.time()
    
    return StreamingResponse(
        io.BytesIO(audio_data),
        media_type="audio/wav",
        headers={
            "Content-Disposition": "attachment; filename=dia_generated_audio.wav"
        }
    )


@app.post("/voice-clone")
async def voice_clone(
    text: str = Form(...),
    audio_file: UploadFile = File(...),
    cfg_scale: float = Form(3.0),
    temperature: float = Form(1.3),
    top_p: float = Form(0.95)
):
    global model
    
    if model is None:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")
    
    try:
        start_time = time.time()
        logger.info(f"Processing voice clone request: '{text[:100]}...'")
        
        audio_content = await audio_file.read()
        audio_path = f"/tmp/dia_voice_clone_{int(time.time())}.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_content)
        
        audio = model.generate(
            text=text,
            audio_prompt_path=audio_path,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p
        )
        
        try:
            os.remove(audio_path)
        except:
            pass
        
        buf = io.BytesIO()
        sf.write(buf, audio, 44100, format="WAV")
        buf.seek(0)
        
        process_time = time.time() - start_time
        logger.info(f"Generated cloned voice in {process_time:.2f} seconds")
        
        return StreamingResponse(
            buf,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=dia_cloned_voice.wav"
            }
        )
        
    except Exception as e:
        logger.error(f"Voice cloning failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")


class BatchRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to convert to speech")
    cfg_scale: float = Field(3.0, description="Classifier-free guidance scale")
    temperature: float = Field(1.3, description="Temperature for sampling")

class BatchJobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    completed_count: int
    total_count: int

batch_jobs = {}

def process_batch(job_id: str, texts: List[str], cfg_scale: float, temperature: float):
    global model, batch_jobs
    
    if model is None:
        try:
            load_model()
        except Exception as e:
            batch_jobs[job_id] = {
                "status": "failed",
                "error": f"Model not available: {str(e)}",
                "progress": 0.0,
                "completed_count": 0,
                "total_count": len(texts),
                "results": []
            }
            return
    
    batch_jobs[job_id]["status"] = "processing"
    results = []
    
    for i, text in enumerate(texts):
        try:
            audio = model.generate(
                text=text,
                cfg_scale=cfg_scale,
                temperature=temperature
            )
            
            buf = io.BytesIO()
            sf.write(buf, audio, 44100, format="WAV")
            buf.seek(0)
            
            import base64
            audio_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            results.append({
                "text": text,
                "audio_data": audio_b64
            })
            
        except Exception as e:
            results.append({
                "text": text,
                "error": str(e)
            })
        
        batch_jobs[job_id]["completed_count"] = i + 1
        batch_jobs[job_id]["progress"] = (i + 1) / len(texts)
    
    batch_jobs[job_id]["status"] = "completed"
    batch_jobs[job_id]["progress"] = 1.0
    batch_jobs[job_id]["results"] = results

@app.post("/batch")
async def batch_generate(request: BatchRequest, background_tasks: BackgroundTasks):
    job_id = f"batch_{int(time.time())}"
    
    batch_jobs[job_id] = {
        "status": "queued",
        "progress": 0.0,
        "completed_count": 0,
        "total_count": len(request.texts),
        "results": []
    }
    
    background_tasks.add_task(
        process_batch,
        job_id,
        request.texts,
        request.cfg_scale,
        request.temperature
    )
    
    return {"job_id": job_id}

@app.get("/batch/{job_id}")
async def get_batch_status(job_id: str):
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = batch_jobs[job_id]
    status = BatchJobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        completed_count=job["completed_count"],
        total_count=job["total_count"]
    )
    
    return status

@app.get("/batch/{job_id}/results")
async def get_batch_results(job_id: str):
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = batch_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not completed: {job['status']}")
    
    return {"results": job["results"]}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)

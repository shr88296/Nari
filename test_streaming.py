#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the Dia model with streaming audio generation.
This script demonstrates the difference between regular generation and streaming generation.
"""

import argparse
import time
import os
import wave
import numpy as np
import pygame
from dia import Dia
from dia.config import DiaConfig

def write_audio_to_wav(audio_data, filename, sample_rate=44100):
    """Write audio data to a WAV file."""
    with wave.open(filename, 'wb') as wav_file:
        # Set parameters
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes (16 bits) per sample
        wav_file.setframerate(sample_rate)
        
        # Convert float32 audio to int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wav_file.writeframes(audio_int16.tobytes())
    
    print(f"Saved audio to {filename}")

def play_audio(audio_data, sample_rate=44100):
    """Play audio using pygame."""
    pygame.mixer.init(frequency=sample_rate, size=-16, channels=1)
    # Convert audio to int16
    audio_int16 = (audio_data * 32767).astype(np.int16)
    sound = pygame.sndarray.make_sound(audio_int16)
    sound.play()
    pygame.time.wait(int(1000 * len(audio_data) / sample_rate))

def test_regular_generation(model, text, max_tokens=1000):
    """Test regular (non-streaming) audio generation."""
    print(f"Generating audio for text: '{text}'")
    start_time = time.time()
    audio = model.generate(
        text=text,
        max_tokens=max_tokens,
        cfg_scale=3.0,
        temperature=1.3,
        top_p=0.95,
    )
    total_time = time.time() - start_time
    
    print(f"Generated {len(audio)} audio samples in {total_time:.2f} seconds")
    return audio

def test_streaming_generation(model, text, max_tokens=1000, chunk_size=100, overlap=10, play=False):
    """Test streaming audio generation."""
    print(f"Generating streaming audio for text: '{text}'")
    print(f"Chunk size: {chunk_size}, Overlap: {overlap}")
    
    audio_chunks = []
    start_time = time.time()
    chunk_times = []
    
    # Use the streaming generator
    for i, audio_chunk in enumerate(model.generate_streaming(
        text=text,
        max_tokens=max_tokens,
        cfg_scale=3.0,
        temperature=1.3,
        top_p=0.95,
        chunk_size=chunk_size,
        overlap=overlap
    )):
        chunk_time = time.time() - start_time
        chunk_times.append(chunk_time)
        
        # Store the chunk
        audio_chunks.append(audio_chunk)
        
        print(f"Received chunk {i+1}: {len(audio_chunk)} samples after {chunk_time:.2f}s")
        
        # Optionally play the audio chunk in real-time
        if play:
            play_audio(audio_chunk)
        
        # Reset start time to measure individual chunk generation time
        start_time = time.time()
    
    # Combine all chunks
    # The full audio stream should ideally be handled in real-time in a real application
    full_audio = np.concatenate(audio_chunks)
    
    print(f"Total audio length: {len(full_audio)} samples")
    print(f"Generated {len(chunk_times)} chunks with average time {sum(chunk_times)/len(chunk_times):.2f}s per chunk")
    
    return full_audio, audio_chunks

def main():
    parser = argparse.ArgumentParser(description="Test Dia model with streaming audio generation")
    parser.add_argument("--text", type=str, default="Hello world, this is a test of streaming audio generation.", 
                        help="Text prompt to generate audio from")
    parser.add_argument("--model", type=str, default="nari-labs/Dia-1.6B",
                        help="Model name or path to local model files")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.json (only needed if using local model)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (only needed if using local model)")
    parser.add_argument("--max_tokens", type=int, default=1000,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--chunk_size", type=int, default=100,
                        help="Number of tokens to generate before yielding audio (for streaming)")
    parser.add_argument("--overlap", type=int, default=10,
                        help="Overlap between chunks to prevent boundary artifacts (for streaming)")
    parser.add_argument("--streaming", action="store_true",
                        help="Use streaming generation mode")
    parser.add_argument("--play", action="store_true",
                        help="Play audio chunks as they're generated (for streaming)")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Output file for the generated audio")
    parser.add_argument("--output_dir", type=str, default="chunks",
                        help="Directory to save individual audio chunks (for streaming)")
    
    args = parser.parse_args()
    
    # Initialize model
    print("Loading model...")
    if args.config and args.checkpoint:
        model = Dia.from_local(args.config, args.checkpoint)
    else:
        model = Dia.from_pretrained(args.model)
    
    print("Model loaded successfully!")
    
    # Test based on mode
    if args.streaming:
        # Create output directory for chunks if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        # Run streaming test
        full_audio, audio_chunks = test_streaming_generation(
            model, 
            args.text, 
            max_tokens=args.max_tokens,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            play=args.play
        )
        
        # Save full audio
        write_audio_to_wav(full_audio, args.output)
        
        # Save individual chunks
        for i, chunk in enumerate(audio_chunks):
            chunk_file = os.path.join(args.output_dir, f"chunk_{i+1}.wav")
            write_audio_to_wav(chunk, chunk_file)
    else:
        # Run regular test
        audio = test_regular_generation(model, args.text, max_tokens=args.max_tokens)
        write_audio_to_wav(audio, args.output)
        
        # Play the audio if requested
        if args.play:
            print("Playing generated audio...")
            play_audio(audio)
    
    print("Done!")

if __name__ == "__main__":
    main() 
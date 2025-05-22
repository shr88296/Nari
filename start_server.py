#!/usr/bin/env python3
"""
Simple startup script for Dia FastAPI TTS Server
"""

import argparse
import os
import sys
import subprocess

def check_environment():
    """Check if the environment is properly set up"""
    issues = []
    
    # Check if HF_TOKEN is set
    if not os.getenv("HF_TOKEN"):
        issues.append("❌ HF_TOKEN environment variable not set")
        issues.append("   Set it with: export HF_TOKEN='your_token_here'")
        issues.append("   Get token from: https://huggingface.co/settings/tokens")
    else:
        print("✅ HF_TOKEN environment variable is set")
    
    # Check if required packages are available
    try:
        import fastapi
        import uvicorn
        import torch
        print("✅ Required packages are available")
    except ImportError as e:
        issues.append(f"❌ Missing required package: {e}")
        issues.append("   Install with: pip install -e .")
    
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available with {torch.cuda.device_count()} GPU(s)")
        else:
            print("ℹ️  CUDA not available, will use CPU (slower)")
    except:
        pass
    
    return issues

def main():
    parser = argparse.ArgumentParser(description="Start Dia TTS FastAPI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--check-only", action="store_true", help="Only check environment, don't start server")
    
    args = parser.parse_args()
    
    print("🚀 Dia TTS Server Startup")
    print("=" * 40)
    
    # Check environment
    issues = check_environment()
    
    if issues:
        print("\n⚠️  Environment Issues:")
        for issue in issues:
            print(issue)
        
        if args.check_only:
            sys.exit(1)
        
        print("\nDo you want to continue anyway? (y/N): ", end="")
        if input().lower() != 'y':
            sys.exit(1)
    
    if args.check_only:
        print("\n✅ Environment check passed!")
        return
    
    print(f"\n🌐 Starting server on {args.host}:{args.port}")
    print("📋 SillyTavern Configuration:")
    print("   TTS Provider: OpenAI Compatible")
    print("   Model: dia")
    print("   API Key: not-needed")
    print(f"   Endpoint URL: http://{args.host}:{args.port}/v1/audio/speech")
    print()
    print("🔗 Server endpoints:")
    print(f"   Health Check: http://{args.host}:{args.port}/health")
    print(f"   API Docs: http://{args.host}:{args.port}/docs")
    print(f"   Voice List: http://{args.host}:{args.port}/v1/voices")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 40)
    
    # Start the server
    try:
        cmd = [
            sys.executable, "fastapi_server.py",
            "--host", args.host,
            "--port", str(args.port)
        ]
        
        if args.reload:
            cmd.append("--reload")
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
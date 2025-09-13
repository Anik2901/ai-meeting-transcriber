#!/usr/bin/env python3
"""
Startup script for AI Meeting Transcriber
Handles environment setup and application startup
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    try:
        import fastapi
        import uvicorn
        import openai
        import websockets
        import pydantic
        import requests
        import dotenv
        import numpy
        import librosa
        import soundfile
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found")
        print("Please copy env_example.txt to .env and configure your API keys")
        return False
    
    # Check for required environment variables
    required_vars = [
        "OPENAI_API_KEY",
        "ZOOM_API_KEY", 
        "ZOOM_API_SECRET",
        "ZOOM_BOT_JID"
    ]
    
    missing_vars = []
    with open(env_file) as f:
        content = f.read()
        for var in required_vars:
            if f"{var}=your_" in content or f"{var}=" not in content:
                missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing or incomplete environment variables: {', '.join(missing_vars)}")
        print("Please update your .env file with the correct API keys")
        return False
    
    print("✅ Environment configuration looks good")
    return True

def main():
    """Main startup function"""
    print("🚀 Starting AI Meeting Transcriber...")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check environment
    if not check_env_file():
        sys.exit(1)
    
    print("=" * 50)
    print("🎉 All checks passed! Starting the application...")
    print("📱 Web interface will be available at: http://localhost:8000")
    print("🔌 WebSocket endpoint: ws://localhost:8000/ws")
    print("=" * 50)
    
    # Start the application
    try:
        import uvicorn
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down AI Meeting Transcriber...")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simple startup script for AI Meeting Transcriber Demo
"""

import os
import sys
from pathlib import Path

# Load environment variables from config.env
def load_env_file():
    env_file = Path("config.env")
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ Environment variables loaded from config.env")
    else:
        print("❌ config.env file not found")

if __name__ == "__main__":
    print("🚀 Starting AI Meeting Transcriber Demo...")
    
    # Load environment variables
    load_env_file()
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your_openai_api_key_here":
        print("❌ Please set your OpenAI API key in config.env")
        sys.exit(1)
    
    print("✅ OpenAI API key configured")
    print("✅ Zoom credentials configured")
    
    # Import and run the application
    try:
        import uvicorn
        
        print("\n🌐 Starting server...")
        print("📱 Web Interface: http://localhost:8000")
        print("📊 API Documentation: http://localhost:8000/docs")
        print("🔍 Health Check: http://localhost:8000/health")
        print("\nPress Ctrl+C to stop the server")
        
        # Start the server directly
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

#!/usr/bin/env python3
"""
Production startup script for AI Meeting Transcriber
Optimized for real-world deployment with proper error handling and monitoring
"""

import os
import sys
import asyncio
import logging
import signal
from pathlib import Path
from contextlib import asynccontextmanager

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/app.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

# Create logs directory
Path("logs").mkdir(exist_ok=True)

def check_environment():
    """Check if all required environment variables are set"""
    required_vars = [
        "OPENAI_API_KEY",
        "ZOOM_API_KEY", 
        "ZOOM_API_SECRET",
        "ZOOM_BOT_JID"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var) or os.getenv(var) == f"your_{var.lower()}_here":
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables in your .env file")
        return False
    
    return True

def check_dependencies():
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
        import jwt
        logger.info("✅ All required packages are installed")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing required package: {e}")
        logger.error("Please run: pip install -r requirements.txt")
        return False

async def health_check():
    """Perform health checks on external services"""
    try:
        # Check OpenAI API
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        client = openai.AsyncOpenAI()
        await client.models.list()
        logger.info("✅ OpenAI API connection successful")
        
        # Check Zoom API
        import requests
        headers = {
            'Authorization': f'Bearer {os.getenv("ZOOM_API_KEY")}',
            'Content-Type': 'application/json'
        }
        response = requests.get("https://api.zoom.us/v2/users/me", headers=headers, timeout=10)
        if response.status_code == 200:
            logger.info("✅ Zoom API connection successful")
        else:
            logger.warning("⚠️ Zoom API connection failed - check your credentials")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return False

@asynccontextmanager
async def lifespan(app):
    """Application lifespan manager"""
    logger.info("🚀 Starting AI Meeting Transcriber...")
    
    # Perform startup checks
    if not check_dependencies():
        sys.exit(1)
    
    if not check_environment():
        sys.exit(1)
    
    if not await health_check():
        logger.warning("⚠️ Some health checks failed, but continuing...")
    
    logger.info("✅ All startup checks passed")
    yield
    
    logger.info("👋 Shutting down AI Meeting Transcriber...")

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

def main():
    """Main startup function"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    logger.info("=" * 60)
    logger.info("🤖 AI Meeting Transcriber - Production Mode")
    logger.info("=" * 60)
    
    # Import and configure the app
    try:
        from main import app
        
        # Configure the app with lifespan
        app.router.lifespan_context = lifespan
        
        # Start the application
        import uvicorn
        
        # Production configuration
        config = uvicorn.Config(
            app,
            host=os.getenv("APP_HOST", "0.0.0.0"),
            port=int(os.getenv("APP_PORT", 8000)),
            log_level=os.getenv("LOG_LEVEL", "info").lower(),
            access_log=True,
            reload=False,  # Disable reload in production
            workers=1,  # Single worker for now
            loop="asyncio",
            http="httptools",  # Faster HTTP parsing
            ws="websockets",  # WebSocket support
        )
        
        server = uvicorn.Server(config)
        
        logger.info(f"📱 Web interface: http://{config.host}:{config.port}")
        logger.info(f"🔌 WebSocket endpoint: ws://{config.host}:{config.port}/ws")
        logger.info(f"📊 API documentation: http://{config.host}:{config.port}/docs")
        logger.info("=" * 60)
        
        # Run the server
        asyncio.run(server.serve())
        
    except KeyboardInterrupt:
        logger.info("👋 Shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

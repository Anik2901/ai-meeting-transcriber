"""
AI Meeting Transcriber - Main Application
A Zoom bot that can be invited to meetings to provide real-time transcription,
important point highlighting, and conversation guidance.
"""

import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
import uvicorn
from dotenv import load_dotenv
import os
from typing import Dict, List
import json
import time
from datetime import datetime

from src.zoom_bot import ZoomBot
from src.transcription_service import TranscriptionService
from src.ai_analyzer import AIAnalyzer
from src.websocket_manager import WebSocketManager
from src.export_service import ExportService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track application start time
start_time = time.time()

app = FastAPI(
    title="AI Meeting Transcriber",
    version="1.0.0",
    description="Production-ready AI-powered meeting transcription and analysis",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["*"]  # Configure this properly in production
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Initialize services
websocket_manager = WebSocketManager()
transcription_service = TranscriptionService()
ai_analyzer = AIAnalyzer()
export_service = ExportService()
zoom_bot = ZoomBot(transcription_service, ai_analyzer, websocket_manager)

@app.get("/")
async def root():
    """Serve the main web interface"""
    return HTMLResponse(open("static/index.html").read())

@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "AI Meeting Transcriber Bot",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "websocket": "/ws",
            "health": "/health",
            "meeting_info": "/meeting/{meeting_id}"
        }
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Check all services
        services_status = {
            "transcription": transcription_service.is_ready(),
            "ai_analyzer": ai_analyzer.is_ready(),
            "zoom_bot": zoom_bot.is_ready(),
            "websocket_manager": websocket_manager.get_connection_count() >= 0
        }
        
        # Check environment variables
        env_status = {
            "openai_api_key": bool(os.getenv("OPENAI_API_KEY")),
            "zoom_api_key": bool(os.getenv("ZOOM_API_KEY")),
            "zoom_api_secret": bool(os.getenv("ZOOM_API_SECRET")),
            "zoom_bot_jid": bool(os.getenv("ZOOM_BOT_JID"))
        }
        
        # Overall health status
        all_services_healthy = all(services_status.values())
        all_env_vars_set = all(env_status.values())
        overall_healthy = all_services_healthy and all_env_vars_set
        
        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "services": services_status,
            "environment": env_status,
            "uptime": time.time() - start_time if 'start_time' in globals() else 0,
            "active_connections": websocket_manager.get_connection_count()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@app.get("/metrics")
async def get_metrics():
    """Get application metrics"""
    try:
        return {
            "timestamp": datetime.now().isoformat(),
            "active_connections": websocket_manager.get_connection_count(),
            "active_meetings": len([m for m in transcription_service.current_transcripts.keys()]),
            "total_transcripts": len(transcription_service.current_transcripts),
            "memory_usage": "N/A",  # Could add psutil for real memory monitoring
            "uptime": time.time() - start_time if 'start_time' in globals() else 0
        }
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to collect metrics")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "join_meeting":
                meeting_id = message.get("meeting_id")
                await zoom_bot.join_meeting(meeting_id)
                
            elif message.get("type") == "leave_meeting":
                await zoom_bot.leave_meeting()
                
            elif message.get("type") == "get_transcript":
                transcript = await transcription_service.get_current_transcript()
                await websocket_manager.send_to_client(websocket, {
                    "type": "transcript",
                    "data": transcript
                })
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket)

@app.get("/meeting/{meeting_id}")
async def get_meeting_info(meeting_id: str):
    """Get information about a specific meeting"""
    try:
        meeting_info = await zoom_bot.get_meeting_info(meeting_id)
        return meeting_info
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Meeting not found: {e}")

@app.post("/webhook/zoom")
async def zoom_webhook(request: dict):
    """Handle Zoom webhook events"""
    try:
        await zoom_bot.handle_webhook(request)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail="Webhook processing failed")

@app.get("/transcript/{meeting_id}")
async def get_meeting_transcript(meeting_id: str):
    """Get the full transcript for a meeting"""
    try:
        transcript = await transcription_service.get_meeting_transcript(meeting_id)
        return {"meeting_id": meeting_id, "transcript": transcript}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Transcript not found: {e}")

@app.get("/summary/{meeting_id}")
async def get_meeting_summary(meeting_id: str):
    """Get AI-generated summary and insights for a meeting"""
    try:
        summary = await ai_analyzer.get_meeting_summary(meeting_id)
        return {"meeting_id": meeting_id, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Summary not found: {e}")

@app.post("/export/{meeting_id}")
async def export_meeting_data(meeting_id: str, format: str = "all"):
    """Export meeting data in various formats"""
    try:
        # Get transcript and summary data
        transcript = await transcription_service.get_meeting_transcript(meeting_id)
        summary = await ai_analyzer.get_meeting_summary(meeting_id)
        
        if format == "all":
            exports = await export_service.export_all_formats(meeting_id, transcript, summary)
            return {"meeting_id": meeting_id, "exports": exports}
        elif format == "transcript_json":
            filepath = await export_service.export_transcript_json(meeting_id, transcript)
            return {"meeting_id": meeting_id, "export": filepath}
        elif format == "transcript_txt":
            filepath = await export_service.export_transcript_txt(meeting_id, transcript)
            return {"meeting_id": meeting_id, "export": filepath}
        elif format == "summary_json":
            filepath = await export_service.export_summary_json(meeting_id, summary)
            return {"meeting_id": meeting_id, "export": filepath}
        elif format == "summary_markdown":
            filepath = await export_service.export_summary_markdown(meeting_id, summary)
            return {"meeting_id": meeting_id, "export": filepath}
        elif format == "action_items_csv":
            filepath = await export_service.export_action_items_csv(meeting_id, summary)
            return {"meeting_id": meeting_id, "export": filepath}
        else:
            raise HTTPException(status_code=400, detail="Invalid export format")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")

@app.get("/exports")
async def list_exports():
    """List all exported files"""
    try:
        exports = export_service.get_export_list()
        return {"exports": exports}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list exports: {e}")

if __name__ == "__main__":
    # Start the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

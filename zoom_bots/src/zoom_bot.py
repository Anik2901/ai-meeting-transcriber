"""
Zoom Bot Implementation
Handles joining meetings, audio capture, and real-time processing
"""

import asyncio
import logging
from typing import Dict, Optional, Any
import json
from datetime import datetime
import os
from .real_zoom_sdk import ZoomSDK

logger = logging.getLogger(__name__)

class ZoomBot:
    def __init__(self, transcription_service, ai_analyzer, websocket_manager):
        self.transcription_service = transcription_service
        self.ai_analyzer = ai_analyzer
        self.websocket_manager = websocket_manager
        self.current_meeting_id = None
        self.is_in_meeting = False
        self.meeting_participants = {}
        self.audio_stream = None
        
        # Initialize Zoom SDK
        self.zoom_sdk = ZoomSDK(
            api_key=os.getenv("ZOOM_API_KEY"),
            api_secret=os.getenv("ZOOM_API_SECRET"),
            bot_jid=os.getenv("ZOOM_BOT_JID")
        )
        
    async def join_meeting(self, meeting_id: str, password: str = None) -> bool:
        """Join a Zoom meeting as a bot participant"""
        try:
            logger.info(f"Attempting to join meeting: {meeting_id}")
            
            # Join the meeting
            join_result = await self.zoom_sdk.join_meeting(
                meeting_id=meeting_id,
                password=password,
                display_name="AI Transcriber Bot"
            )
            
            if join_result["success"]:
                self.current_meeting_id = meeting_id
                self.is_in_meeting = True
                
                # Start audio capture
                await self._start_audio_capture()
                
                # Notify connected clients
                await self.websocket_manager.broadcast({
                    "type": "meeting_joined",
                    "meeting_id": meeting_id,
                    "timestamp": datetime.now().isoformat()
                })
                
                logger.info(f"Successfully joined meeting: {meeting_id}")
                return True
            else:
                logger.error(f"Failed to join meeting: {join_result['error']}")
                return False
                
        except Exception as e:
            logger.error(f"Error joining meeting: {e}")
            return False
    
    async def leave_meeting(self) -> bool:
        """Leave the current meeting"""
        try:
            if self.is_in_meeting and self.current_meeting_id:
                # Stop audio capture
                await self._stop_audio_capture()
                
                # Leave the meeting
                await self.zoom_sdk.leave_meeting()
                
                # Notify connected clients
                await self.websocket_manager.broadcast({
                    "type": "meeting_left",
                    "meeting_id": self.current_meeting_id,
                    "timestamp": datetime.now().isoformat()
                })
                
                self.is_in_meeting = False
                self.current_meeting_id = None
                self.meeting_participants = {}
                
                logger.info("Successfully left meeting")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error leaving meeting: {e}")
            return False
    
    async def _start_audio_capture(self):
        """Start capturing audio from the meeting"""
        try:
            # Get audio stream from Zoom SDK
            self.audio_stream = await self.zoom_sdk.get_audio_stream()
            
            # Start processing audio in background
            asyncio.create_task(self._process_audio_stream())
            
        except Exception as e:
            logger.error(f"Error starting audio capture: {e}")
    
    async def _stop_audio_capture(self):
        """Stop capturing audio"""
        try:
            if self.audio_stream:
                await self.audio_stream.close()
                self.audio_stream = None
        except Exception as e:
            logger.error(f"Error stopping audio capture: {e}")
    
    async def _process_audio_stream(self):
        """Process incoming audio stream for transcription"""
        try:
            while self.is_in_meeting and self.audio_stream:
                # Read audio chunk
                audio_chunk = await self.audio_stream.read()
                
                if audio_chunk:
                    # Send to transcription service
                    transcription_result = await self.transcription_service.process_audio_chunk(
                        audio_chunk, self.current_meeting_id
                    )
                    
                    if transcription_result:
                        # Send to AI analyzer for processing
                        analysis_result = await self.ai_analyzer.analyze_transcript(
                            transcription_result, self.current_meeting_id
                        )
                        
                        # Broadcast results to connected clients
                        await self.websocket_manager.broadcast({
                            "type": "transcription_update",
                            "meeting_id": self.current_meeting_id,
                            "transcript": transcription_result,
                            "analysis": analysis_result,
                            "timestamp": datetime.now().isoformat()
                        })
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error processing audio stream: {e}")
    
    async def handle_webhook(self, webhook_data: Dict[str, Any]):
        """Handle Zoom webhook events"""
        try:
            event_type = webhook_data.get("event")
            
            if event_type == "meeting.participant_joined":
                participant = webhook_data.get("payload", {}).get("object", {}).get("participant", {})
                self.meeting_participants[participant["id"]] = participant
                
                await self.websocket_manager.broadcast({
                    "type": "participant_joined",
                    "participant": participant,
                    "timestamp": datetime.now().isoformat()
                })
                
            elif event_type == "meeting.participant_left":
                participant = webhook_data.get("payload", {}).get("object", {}).get("participant", {})
                if participant["id"] in self.meeting_participants:
                    del self.meeting_participants[participant["id"]]
                
                await self.websocket_manager.broadcast({
                    "type": "participant_left",
                    "participant": participant,
                    "timestamp": datetime.now().isoformat()
                })
                
            elif event_type == "meeting.ended":
                await self.leave_meeting()
                
        except Exception as e:
            logger.error(f"Error handling webhook: {e}")
    
    async def get_meeting_info(self, meeting_id: str) -> Dict[str, Any]:
        """Get information about a meeting"""
        try:
            meeting_info = await self.zoom_sdk.get_meeting_info(meeting_id)
            return {
                "meeting_id": meeting_id,
                "topic": meeting_info.get("topic"),
                "start_time": meeting_info.get("start_time"),
                "duration": meeting_info.get("duration"),
                "participants": list(self.meeting_participants.values()),
                "is_active": self.is_in_meeting and self.current_meeting_id == meeting_id
            }
        except Exception as e:
            logger.error(f"Error getting meeting info: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check if the bot is ready to join meetings"""
        return (
            self.zoom_sdk is not None and
            self.transcription_service.is_ready() and
            self.ai_analyzer.is_ready()
        )

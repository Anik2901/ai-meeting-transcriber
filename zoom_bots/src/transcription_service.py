"""
Transcription Service
Handles real-time speech-to-text conversion using OpenAI Whisper
"""

import asyncio
import logging
import io
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Any
import openai
import os
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class TranscriptionService:
    def __init__(self):
        # Set the API key for the global client
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = openai.AsyncOpenAI()
        self.current_transcripts = {}  # meeting_id -> transcript
        self.audio_buffer = {}  # meeting_id -> audio buffer
        self.is_ready_flag = False
        
        # Initialize the service
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialize the transcription service"""
        try:
            # Test OpenAI connection
            await self.openai_client.models.list()
            self.is_ready_flag = True
            logger.info("Transcription service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize transcription service: {e}")
            self.is_ready_flag = False
    
    async def process_audio_chunk(self, audio_data: bytes, meeting_id: str) -> Optional[Dict[str, Any]]:
        """Process a chunk of audio data and return transcription"""
        try:
            if not self.is_ready_flag:
                return None
            
            # Add audio to buffer
            if meeting_id not in self.audio_buffer:
                self.audio_buffer[meeting_id] = []
            
            self.audio_buffer[meeting_id].append(audio_data)
            
            # Process buffer when it reaches a certain size (e.g., 3 seconds of audio)
            if len(self.audio_buffer[meeting_id]) >= 30:  # Adjust based on sample rate
                return await self._transcribe_buffer(meeting_id)
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return None
    
    async def _transcribe_buffer(self, meeting_id: str) -> Optional[Dict[str, Any]]:
        """Transcribe the accumulated audio buffer"""
        try:
            if meeting_id not in self.audio_buffer or not self.audio_buffer[meeting_id]:
                return None
            
            # Combine audio chunks
            combined_audio = b''.join(self.audio_buffer[meeting_id])
            
            # Convert to numpy array and process
            audio_array = np.frombuffer(combined_audio, dtype=np.int16)
            
            # Convert to float32 and normalize
            audio_float = audio_array.astype(np.float32) / 32768.0
            
            # Resample if necessary (Whisper expects 16kHz)
            if len(audio_float) > 0:
                # Use librosa for proper resampling
                try:
                    audio_resampled = librosa.resample(audio_float, orig_sr=16000, target_sr=16000)
                except:
                    # Fallback to simple resampling
                    target_length = int(len(audio_float) * 16000 / 16000)
                    audio_resampled = np.interp(
                        np.linspace(0, len(audio_float) - 1, target_length),
                        np.arange(len(audio_float)),
                        audio_float
                    )
                
                # Convert to bytes for OpenAI API
                audio_bytes = (audio_resampled * 32767).astype(np.int16).tobytes()
                
                # Create audio file object with proper WAV format
                audio_file = io.BytesIO()
                
                # Write WAV header
                import wave
                with wave.open(audio_file, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(16000)  # 16kHz
                    wav_file.writeframes(audio_bytes)
                
                audio_file.seek(0)
                audio_file.name = "audio.wav"
                
                # Transcribe using OpenAI Whisper with optimized settings
                transcription = await self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="verbose_json",
                    timestamp_granularities=["word"],
                    language="en",  # Specify language for better accuracy
                    temperature=0.0  # More deterministic output
                )
                
                # Clear the buffer
                self.audio_buffer[meeting_id] = []
                
                # Process and return result
                result = {
                    "text": transcription.text.strip(),
                    "language": transcription.language,
                    "duration": transcription.duration,
                    "words": transcription.words if hasattr(transcription, 'words') else [],
                    "timestamp": datetime.now().isoformat(),
                    "meeting_id": meeting_id,
                    "confidence": getattr(transcription, 'confidence', 0.9)
                }
                
                # Only add to transcript if there's actual text
                if result["text"]:
                    self._add_to_transcript(meeting_id, result)
                    return result
            
            return None
            
        except Exception as e:
            logger.error(f"Error transcribing buffer: {e}")
            return None
    
    def _add_to_transcript(self, meeting_id: str, transcription_result: Dict[str, Any]):
        """Add transcription result to the meeting transcript"""
        if meeting_id not in self.current_transcripts:
            self.current_transcripts[meeting_id] = {
                "meeting_id": meeting_id,
                "start_time": datetime.now().isoformat(),
                "segments": [],
                "full_text": "",
                "participants": {}
            }
        
        # Add segment
        self.current_transcripts[meeting_id]["segments"].append(transcription_result)
        
        # Update full text
        if transcription_result["text"].strip():
            if self.current_transcripts[meeting_id]["full_text"]:
                self.current_transcripts[meeting_id]["full_text"] += " "
            self.current_transcripts[meeting_id]["full_text"] += transcription_result["text"]
    
    async def get_current_transcript(self, meeting_id: str = None) -> Dict[str, Any]:
        """Get the current transcript for a meeting"""
        if meeting_id:
            return self.current_transcripts.get(meeting_id, {})
        else:
            # Return the most recent transcript
            if self.current_transcripts:
                latest_meeting = max(
                    self.current_transcripts.keys(),
                    key=lambda k: self.current_transcripts[k]["start_time"]
                )
                return self.current_transcripts[latest_meeting]
            return {}
    
    async def get_meeting_transcript(self, meeting_id: str) -> Dict[str, Any]:
        """Get the complete transcript for a specific meeting"""
        return self.current_transcripts.get(meeting_id, {})
    
    async def save_transcript(self, meeting_id: str, file_path: str):
        """Save transcript to file"""
        try:
            transcript = self.current_transcripts.get(meeting_id)
            if transcript:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(transcript, f, indent=2, ensure_ascii=False)
                logger.info(f"Transcript saved to {file_path}")
            else:
                logger.warning(f"No transcript found for meeting {meeting_id}")
        except Exception as e:
            logger.error(f"Error saving transcript: {e}")
    
    def is_ready(self) -> bool:
        """Check if the transcription service is ready"""
        return self.is_ready_flag

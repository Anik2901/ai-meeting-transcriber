"""
AI Analyzer
Provides intelligent analysis of meeting transcripts including:
- Important point extraction
- Conversation guidance
- Question suggestions
- Action item identification
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import openai
import os
from datetime import datetime
import json
import re

logger = logging.getLogger(__name__)

class AIAnalyzer:
    def __init__(self):
        # Set the API key for the global client
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = openai.AsyncOpenAI()
        self.meeting_analyses = {}  # meeting_id -> analysis
        self.is_ready_flag = False
        
        # Initialize the service
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialize the AI analyzer"""
        try:
            # Test OpenAI connection
            await self.openai_client.models.list()
            self.is_ready_flag = True
            logger.info("AI Analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI Analyzer: {e}")
            self.is_ready_flag = False
    
    async def analyze_transcript(self, transcription_result: Dict[str, Any], meeting_id: str) -> Dict[str, Any]:
        """Analyze a transcription result and provide insights"""
        try:
            if not self.is_ready_flag or not transcription_result.get("text"):
                return {}
            
            text = transcription_result["text"]
            
            # Get current meeting context
            meeting_context = self.meeting_analyses.get(meeting_id, {
                "meeting_id": meeting_id,
                "start_time": datetime.now().isoformat(),
                "important_points": [],
                "action_items": [],
                "questions_suggested": [],
                "conversation_guidance": [],
                "topics_discussed": [],
                "sentiment_analysis": [],
                "full_context": ""
            })
            
            # Update full context
            if meeting_context["full_context"]:
                meeting_context["full_context"] += " "
            meeting_context["full_context"] += text
            
            # Analyze the new text
            analysis = await self._analyze_text_chunk(text, meeting_context)
            
            # Update meeting analysis
            self._update_meeting_analysis(meeting_id, analysis)
            self.meeting_analyses[meeting_id] = meeting_context
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing transcript: {e}")
            return {}
    
    async def _analyze_text_chunk(self, text: str, meeting_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a chunk of text and extract insights"""
        try:
            # Create analysis prompt
            prompt = f"""
            Analyze the following meeting transcript segment and provide insights:
            
            Previous context: {meeting_context.get('full_context', '')[-1000:]}
            
            New segment: "{text}"
            
            Please provide:
            1. Important points mentioned (if any)
            2. Action items or decisions made (if any)
            3. Questions that could be asked to further the conversation
            4. Conversation guidance (suggestions for where to take the discussion)
            5. Topics being discussed
            6. Sentiment analysis (positive, negative, neutral)
            
            Format your response as JSON with these keys:
            - important_points: array of important points
            - action_items: array of action items
            - suggested_questions: array of relevant questions
            - conversation_guidance: array of guidance suggestions
            - topics: array of topics mentioned
            - sentiment: object with overall sentiment and confidence
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI assistant that analyzes meeting transcripts to provide intelligent insights and guidance. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            # Parse the response
            analysis_text = response.choices[0].message.content
            analysis = json.loads(analysis_text)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing text chunk: {e}")
            return {}
    
    def _update_meeting_analysis(self, meeting_id: str, analysis: Dict[str, Any]):
        """Update the meeting analysis with new insights"""
        if meeting_id not in self.meeting_analyses:
            self.meeting_analyses[meeting_id] = {
                "meeting_id": meeting_id,
                "start_time": datetime.now().isoformat(),
                "important_points": [],
                "action_items": [],
                "questions_suggested": [],
                "conversation_guidance": [],
                "topics_discussed": [],
                "sentiment_analysis": [],
                "full_context": ""
            }
        
        meeting_analysis = self.meeting_analyses[meeting_id]
        
        # Add new insights
        if analysis.get("important_points"):
            meeting_analysis["important_points"].extend(analysis["important_points"])
        
        if analysis.get("action_items"):
            meeting_analysis["action_items"].extend(analysis["action_items"])
        
        if analysis.get("suggested_questions"):
            meeting_analysis["questions_suggested"].extend(analysis["suggested_questions"])
        
        if analysis.get("conversation_guidance"):
            meeting_analysis["conversation_guidance"].extend(analysis["conversation_guidance"])
        
        if analysis.get("topics"):
            meeting_analysis["topics_discussed"].extend(analysis["topics"])
        
        if analysis.get("sentiment"):
            meeting_analysis["sentiment_analysis"].append({
                "timestamp": datetime.now().isoformat(),
                "sentiment": analysis["sentiment"]
            })
    
    async def get_meeting_summary(self, meeting_id: str) -> Dict[str, Any]:
        """Get a comprehensive summary of the meeting"""
        try:
            meeting_analysis = self.meeting_analyses.get(meeting_id)
            if not meeting_analysis:
                return {"error": "No analysis found for this meeting"}
            
            # Generate comprehensive summary
            summary_prompt = f"""
            Based on the following meeting analysis, provide a comprehensive summary:
            
            Meeting Analysis: {json.dumps(meeting_analysis, indent=2)}
            
            Please provide:
            1. Executive summary (2-3 sentences)
            2. Key decisions made
            3. Action items with owners (if mentioned)
            4. Important topics discussed
            5. Next steps or follow-up items
            6. Overall meeting sentiment and effectiveness
            
            Format as JSON with these keys:
            - executive_summary: string
            - key_decisions: array
            - action_items: array with owner and due_date if mentioned
            - topics_discussed: array
            - next_steps: array
            - meeting_effectiveness: object with rating and notes
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI assistant that creates comprehensive meeting summaries. Always respond with valid JSON."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            summary_text = response.choices[0].message.content
            summary = json.loads(summary_text)
            
            # Add metadata
            summary["meeting_id"] = meeting_id
            summary["generated_at"] = datetime.now().isoformat()
            summary["analysis_data"] = meeting_analysis
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating meeting summary: {e}")
            return {"error": f"Failed to generate summary: {e}"}
    
    async def get_conversation_suggestions(self, meeting_id: str) -> Dict[str, Any]:
        """Get real-time conversation suggestions"""
        try:
            meeting_analysis = self.meeting_analyses.get(meeting_id)
            if not meeting_analysis:
                return {"suggestions": []}
            
            # Get recent context
            recent_context = meeting_analysis.get("full_context", "")[-2000:]  # Last 2000 characters
            
            suggestion_prompt = f"""
            Based on the recent meeting conversation, suggest:
            1. 3-5 relevant questions to ask
            2. 2-3 topics to explore further
            3. 2-3 potential action items to discuss
            4. Overall conversation direction suggestions
            
            Recent context: "{recent_context}"
            
            Format as JSON with these keys:
            - questions: array of suggested questions
            - topics_to_explore: array of topics
            - potential_actions: array of action items
            - conversation_direction: array of direction suggestions
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI assistant that provides real-time conversation guidance for meetings. Always respond with valid JSON."},
                    {"role": "user", "content": suggestion_prompt}
                ],
                temperature=0.4,
                max_tokens=800
            )
            
            suggestion_text = response.choices[0].message.content
            suggestions = json.loads(suggestion_text)
            
            suggestions["timestamp"] = datetime.now().isoformat()
            suggestions["meeting_id"] = meeting_id
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating conversation suggestions: {e}")
            return {"error": f"Failed to generate suggestions: {e}"}
    
    def is_ready(self) -> bool:
        """Check if the AI analyzer is ready"""
        return self.is_ready_flag

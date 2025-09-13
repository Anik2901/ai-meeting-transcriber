"""
WebSocket Manager
Handles WebSocket connections for real-time communication
"""

import asyncio
import logging
from typing import Dict, List, Set
from fastapi import WebSocket
import json

logger = logging.getLogger(__name__)

class WebSocketManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connection_metadata: Dict[WebSocket, Dict] = {}
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_metadata[websocket] = {
            "connected_at": asyncio.get_event_loop().time(),
            "last_activity": asyncio.get_event_loop().time()
        }
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            if websocket in self.connection_metadata:
                del self.connection_metadata[websocket]
            logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_to_client(self, websocket: WebSocket, message: Dict):
        """Send a message to a specific client"""
        try:
            if websocket in self.active_connections:
                await websocket.send_text(json.dumps(message))
                self.connection_metadata[websocket]["last_activity"] = asyncio.get_event_loop().time()
        except Exception as e:
            logger.error(f"Error sending message to client: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict):
        """Broadcast a message to all connected clients"""
        if not self.active_connections:
            return
        
        # Create a list of connections to avoid modification during iteration
        connections_to_send = list(self.active_connections)
        
        for websocket in connections_to_send:
            try:
                await websocket.send_text(json.dumps(message))
                self.connection_metadata[websocket]["last_activity"] = asyncio.get_event_loop().time()
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                self.disconnect(websocket)
    
    async def broadcast_to_meeting(self, meeting_id: str, message: Dict):
        """Broadcast a message to clients connected to a specific meeting"""
        # This would require tracking which clients are connected to which meetings
        # For now, we'll broadcast to all clients
        await self.broadcast(message)
    
    def get_connection_count(self) -> int:
        """Get the number of active connections"""
        return len(self.active_connections)
    
    def get_connection_info(self) -> List[Dict]:
        """Get information about all active connections"""
        current_time = asyncio.get_event_loop().time()
        return [
            {
                "connected_at": metadata["connected_at"],
                "last_activity": metadata["last_activity"],
                "duration": current_time - metadata["connected_at"],
                "idle_time": current_time - metadata["last_activity"]
            }
            for metadata in self.connection_metadata.values()
        ]

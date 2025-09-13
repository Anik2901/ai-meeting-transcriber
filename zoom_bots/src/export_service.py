"""
Export Service
Handles exporting transcripts and summaries in various formats
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import json
import csv
from datetime import datetime
from pathlib import Path
import aiofiles

logger = logging.getLogger(__name__)

class ExportService:
    def __init__(self):
        self.export_dir = Path("exports")
        self.export_dir.mkdir(exist_ok=True)
    
    async def export_transcript_json(self, meeting_id: str, transcript_data: Dict[str, Any]) -> str:
        """Export transcript as JSON file"""
        try:
            filename = f"transcript_{meeting_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.export_dir / filename
            
            async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(transcript_data, indent=2, ensure_ascii=False))
            
            logger.info(f"Transcript exported to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exporting transcript as JSON: {e}")
            raise
    
    async def export_transcript_txt(self, meeting_id: str, transcript_data: Dict[str, Any]) -> str:
        """Export transcript as plain text file"""
        try:
            filename = f"transcript_{meeting_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            filepath = self.export_dir / filename
            
            content = f"Meeting Transcript - {meeting_id}\n"
            content += f"Generated: {datetime.now().isoformat()}\n"
            content += "=" * 50 + "\n\n"
            
            if transcript_data.get("segments"):
                for segment in transcript_data["segments"]:
                    timestamp = segment.get("timestamp", "Unknown time")
                    text = segment.get("text", "")
                    content += f"[{timestamp}] {text}\n"
            else:
                content += transcript_data.get("full_text", "No transcript available")
            
            async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                await f.write(content)
            
            logger.info(f"Transcript exported to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exporting transcript as TXT: {e}")
            raise
    
    async def export_summary_json(self, meeting_id: str, summary_data: Dict[str, Any]) -> str:
        """Export meeting summary as JSON file"""
        try:
            filename = f"summary_{meeting_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.export_dir / filename
            
            async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(summary_data, indent=2, ensure_ascii=False))
            
            logger.info(f"Summary exported to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exporting summary as JSON: {e}")
            raise
    
    async def export_summary_markdown(self, meeting_id: str, summary_data: Dict[str, Any]) -> str:
        """Export meeting summary as Markdown file"""
        try:
            filename = f"summary_{meeting_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            filepath = self.export_dir / filename
            
            content = f"# Meeting Summary - {meeting_id}\n\n"
            content += f"**Generated:** {datetime.now().isoformat()}\n\n"
            
            summary = summary_data.get("summary", {})
            
            # Executive Summary
            if summary.get("executive_summary"):
                content += "## Executive Summary\n\n"
                content += f"{summary['executive_summary']}\n\n"
            
            # Key Decisions
            if summary.get("key_decisions"):
                content += "## Key Decisions\n\n"
                for decision in summary["key_decisions"]:
                    content += f"- {decision}\n"
                content += "\n"
            
            # Action Items
            if summary.get("action_items"):
                content += "## Action Items\n\n"
                for item in summary["action_items"]:
                    content += f"- {item}\n"
                content += "\n"
            
            # Topics Discussed
            if summary.get("topics_discussed"):
                content += "## Topics Discussed\n\n"
                for topic in summary["topics_discussed"]:
                    content += f"- {topic}\n"
                content += "\n"
            
            # Next Steps
            if summary.get("next_steps"):
                content += "## Next Steps\n\n"
                for step in summary["next_steps"]:
                    content += f"- {step}\n"
                content += "\n"
            
            # Meeting Effectiveness
            if summary.get("meeting_effectiveness"):
                effectiveness = summary["meeting_effectiveness"]
                content += "## Meeting Effectiveness\n\n"
                content += f"**Rating:** {effectiveness.get('rating', 'N/A')}\n\n"
                if effectiveness.get("notes"):
                    content += f"**Notes:** {effectiveness['notes']}\n\n"
            
            async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                await f.write(content)
            
            logger.info(f"Summary exported to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exporting summary as Markdown: {e}")
            raise
    
    async def export_action_items_csv(self, meeting_id: str, summary_data: Dict[str, Any]) -> str:
        """Export action items as CSV file"""
        try:
            filename = f"action_items_{meeting_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            filepath = self.export_dir / filename
            
            action_items = summary_data.get("summary", {}).get("action_items", [])
            
            async with aiofiles.open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Action Item", "Owner", "Due Date", "Status", "Notes"])
                
                for item in action_items:
                    # Parse action item for owner and due date if available
                    owner = "TBD"
                    due_date = "TBD"
                    status = "Open"
                    notes = ""
                    
                    # Simple parsing - in production, use more sophisticated parsing
                    if ":" in item:
                        parts = item.split(":", 1)
                        if len(parts) == 2:
                            owner = parts[0].strip()
                            notes = parts[1].strip()
                    
                    writer.writerow([item, owner, due_date, status, notes])
            
            logger.info(f"Action items exported to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error exporting action items as CSV: {e}")
            raise
    
    async def export_all_formats(self, meeting_id: str, transcript_data: Dict[str, Any], summary_data: Dict[str, Any]) -> Dict[str, str]:
        """Export meeting data in all available formats"""
        try:
            exports = {}
            
            # Export transcript
            exports["transcript_json"] = await self.export_transcript_json(meeting_id, transcript_data)
            exports["transcript_txt"] = await self.export_transcript_txt(meeting_id, transcript_data)
            
            # Export summary
            exports["summary_json"] = await self.export_summary_json(meeting_id, summary_data)
            exports["summary_markdown"] = await self.export_summary_markdown(meeting_id, summary_data)
            
            # Export action items
            exports["action_items_csv"] = await self.export_action_items_csv(meeting_id, summary_data)
            
            logger.info(f"All formats exported for meeting {meeting_id}")
            return exports
            
        except Exception as e:
            logger.error(f"Error exporting all formats: {e}")
            raise
    
    def get_export_list(self) -> List[Dict[str, str]]:
        """Get list of all exported files"""
        try:
            exports = []
            for file_path in self.export_dir.glob("*"):
                if file_path.is_file():
                    exports.append({
                        "filename": file_path.name,
                        "filepath": str(file_path),
                        "size": file_path.stat().st_size,
                        "created": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat()
                    })
            
            return sorted(exports, key=lambda x: x["created"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting export list: {e}")
            return []

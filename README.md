# AI Meeting Transcriber

A real-time meeting transcription and analysis tool that captures audio from any meeting platform and provides intelligent insights.

## Features

- **Live Audio Capture**: Records audio from your microphone or system audio
- **Real-time Transcription**: Uses OpenAI Whisper for accurate speech-to-text
- **Intelligent Analysis**: GPT-4 powered extraction of:
  - Important points from conversations
  - Context-aware follow-up questions
  - Action items and tasks
- **Universal Compatibility**: Works with any meeting platform (Zoom, Teams, Meet, etc.)
- **WebSocket Updates**: Real-time UI updates as transcription happens
- **Export Reports**: Save transcripts and insights as HTML

## Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key with access to Whisper and GPT-4
- Working microphone or virtual audio cable

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-meeting-transcriber.git
cd ai-meeting-transcriber
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your OpenAI API key:
Create a `config.env` file in the root directory:
```
OPENAI_API_KEY=your_api_key_here
```

### Running the Application

```bash
python app.py
```

The application will start and be available at:
- **http://localhost:8007**

## How to Use

1. **Open the Web Interface**: Navigate to http://localhost:8007 in your browser
2. **Start Capture**: Click the "Start Capture" button
3. **Select Audio Device**: Choose your microphone or system audio device
4. **Join Your Meeting**: Join any meeting on any platform
5. **Watch Live Transcription**: See real-time transcription and analysis

## Project Structure

```
ai-meeting-transcriber/
├── app.py                    # Main application (FastAPI + WebSocket)
├── config.env               # Environment variables (create this)
├── requirements.txt         # Python dependencies
├── transcripts/            # Saved transcription sessions
├── zoom_bots/             # Zoom SDK integration experiments (archived)
│   ├── main.py
│   ├── simple_zoom_bot.py
│   ├── zoom_meeting_bot.py
│   └── src/
└── demos/                  # Various demo implementations
    ├── transcriber_only/
    │   ├── live_transcriber.py  # Original version
    │   └── streamlit_app.py     # Streamlit variant
    └── real_openai/
```

## Key Features Explained

### Voice Activity Detection (VAD)
The app uses WebRTC VAD to detect when someone is speaking, reducing unnecessary API calls and improving transcription accuracy.

### Smart Buffering
Audio is buffered in 10-second chunks for optimal transcription quality while maintaining real-time responsiveness.

### Context-Aware Analysis
GPT-4 analyzes each transcript segment to provide:
- Bullet-point summaries of important topics
- Relevant follow-up questions based on the conversation
- Actionable tasks mentioned in the meeting

## API Endpoints

- `GET /` - Web interface
- `WS /ws` - WebSocket for real-time updates
- `POST /start` - Start audio capture
- `POST /stop` - Stop capture and generate report
- `GET /devices` - List available audio devices
- `GET /export` - Export session as HTML
- `GET /health` - Health check endpoint

## Troubleshooting

### PyAudio Installation Issues
If you encounter issues installing PyAudio on Windows:
```bash
pip install pipwin
pipwin install pyaudio
```

### No Audio Captured
- Ensure your microphone permissions are enabled
- Try selecting a different audio device from the dropdown
- Check that your microphone is not being used by another application

### API Errors
- Verify your OpenAI API key is valid
- Check you have access to both Whisper and GPT-4 models
- Ensure you have sufficient API credits

## Development

### Running in Development Mode
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8007
```

### Testing
```bash
python test_openai_simple.py  # Test OpenAI API connection
```

## License

MIT License - See LICENSE file for details

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## Support

For issues and questions, please open an issue on GitHub.
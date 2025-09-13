#!/usr/bin/env python3
"""
Live Transcriber - Works in any meeting (Zoom, Meet, Teams) by using your mic
- Start/Stop capture from UI
- Streams 10s chunks to Whisper (HTTP) for transcription
- Sends each transcript to GPT-4 for key points and suggested questions
- Live feed via WebSocket
"""

import os
import sys
import json
import time
import wave
import asyncio
import logging
from io import BytesIO
from datetime import datetime

import pyaudio
import requests
import numpy as np
import webrtcvad
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, Response
import uvicorn

# Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Env loader
def load_env_file():
    env_file = "config.env"
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        logger.info("✅ Environment variables loaded")

load_env_file()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ OPENAI_API_KEY missing in config.env")
    sys.exit(1)

# FastAPI app
app = FastAPI(title="Live Transcriber", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# WebSocket manager
class WS:
    def __init__(self):
        self.conns = []
    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.conns.append(ws)
        logger.info(f"WS connected. Total: {len(self.conns)}")
    def disconnect(self, ws: WebSocket):
        if ws in self.conns:
            self.conns.remove(ws)
        logger.info(f"WS disconnected. Total: {len(self.conns)}")
    async def broadcast(self, message: dict):
        text = json.dumps(message)
        for c in list(self.conns):
            try:
                await c.send_text(text)
            except Exception:
                try:
                    self.disconnect(c)
                except Exception:
                    pass

ws_mgr = WS()

# Capture state
class CaptureState:
    def __init__(self):
        self.running = False
        self.task = None
        self.py = None
        self.stream = None
        self.rate = 16000
        self.channels = 1
        self.chunk = 480  # 30ms frame at 16kHz
        self.buf: list[bytes] = []
        self.device_index: int | None = None
        # New fields
        self.vad = webrtcvad.Vad(2)
        self.pre_frames = 17  # ~510ms
        self.tail_frames = 20  # ~600ms
        self.pre_roll: deque[bytes] = deque(maxlen=self.pre_frames)
        self.in_speech = False
        self.silence_count = 0
        self.segment_frames: list[bytes] = []
        self.baseline_rms: float | None = None
        self.threshold: float | None = None
        self.transcripts: list[dict] = []
        self.points_store: list[str] = []
        self.questions_store: list[str] = []
        self.action_items_store: list[str] = []
        self.session_started_at = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.join(os.getcwd(), 'transcripts')
        os.makedirs(self.save_dir, exist_ok=True)

capture = CaptureState()


def list_input_devices() -> list[dict]:
    devices = []
    py = pyaudio.PyAudio()
    try:
        for i in range(py.get_device_count()):
            info = py.get_device_info_by_index(i)
            if info.get('maxInputChannels', 0) > 0:
                devices.append({
                    'index': i,
                    'name': info.get('name'),
                    'hostApi': info.get('hostApi'),
                    'maxInputChannels': info.get('maxInputChannels'),
                    'defaultSampleRate': info.get('defaultSampleRate')
                })
    finally:
        try:
            py.terminate()
        except Exception:
            pass
    return devices

async def transcribe_bytes(audio_bytes: bytes) -> str:
    try:
        # Build multipart for Whisper v1
        files = {
            'file': ('audio.wav', audio_bytes, 'audio/wav')
        }
        data = {
            'model': 'whisper-1',
            'response_format': 'text'
        }
        headers = { 'Authorization': f'Bearer {OPENAI_API_KEY}' }
        resp = requests.post('https://api.openai.com/v1/audio/transcriptions', headers=headers, data=data, files=files, timeout=60)
        if resp.status_code == 200:
            return resp.text.strip()
        logger.error(f"Whisper error {resp.status_code}: {resp.text}")
        return ""
    except Exception as e:
        logger.error(f"Whisper call failed: {e}")
        return ""

async def analyze_text(text: str) -> dict:
    if not text.strip():
        return {}
    try:
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': 'gpt-4',
            'messages': [
                { 'role': 'system', 'content': (
                    'Extract concrete, context-aware insights, action items, and high-quality follow-up questions from meeting speech. '
                    'Return strict JSON with keys: important_points (list of concise bullets), action_items (list of actionable tasks), '
                    'suggested_questions (list of 3-5 specific, non-generic, actionable questions referencing details from the text).'
                )},
                { 'role': 'user', 'content': text }
            ],
            'temperature': 0.2,
            'max_tokens': 300
        }
        resp = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload, timeout=60)
        if resp.status_code != 200:
            logger.error(f"GPT error {resp.status_code}: {resp.text}")
            return {}
        content = resp.json()['choices'][0]['message']['content']
        try:
            return json.loads(content)
        except Exception:
            return { 'important_points': [content], 'action_items': [], 'suggested_questions': [] }
    except Exception as e:
        logger.error(f"GPT call failed: {e}")
        return {}

async def capture_loop():
    logger.info("🎤 Starting microphone capture (VAD)...")
    capture.py = pyaudio.PyAudio()
    try:
        try:
            capture.stream = capture.py.open(
                format=pyaudio.paInt16,
                channels=capture.channels,
                rate=capture.rate,
                input=True,
                frames_per_buffer=capture.chunk,
                input_device_index=capture.device_index if capture.device_index is not None else None
            )
        except Exception as e:
            logger.warning(f"Primary mic open failed (device_index={capture.device_index}): {e}. Falling back to default input device...")
            capture.stream = capture.py.open(
                format=pyaudio.paInt16,
                channels=capture.channels,
                rate=capture.rate,
                input=True,
                frames_per_buffer=capture.chunk
            )
    except Exception as e:
        logger.error(f"Mic open failed: {e}")
        capture.running = False
        return

    bytes_per_frame = capture.chunk * 2  # int16 mono
    min_sec = 0.6

    # Calibrate ~1s baseline
    try:
        calib_frames = []
        needed = int(capture.rate / (capture.chunk))
        for _ in range(max(1, needed)):
            calib_frames.append(capture.stream.read(capture.chunk, exception_on_overflow=False))
        calib = b''.join(calib_frames)
        np_all = np.frombuffer(calib, dtype=np.int16)
        capture.baseline_rms = float(np.sqrt(np.mean(np.square(np_all)))) if np_all.size else 0.0
        capture.threshold = max(15.0, capture.baseline_rms * 2.0)
        logger.info(f"🔧 Calibrated baseline RMS={capture.baseline_rms:.1f}, threshold={capture.threshold:.1f}")
    except Exception:
        capture.baseline_rms = 0.0
        capture.threshold = 30.0

    try:
        while capture.running:
            frame = capture.stream.read(capture.chunk, exception_on_overflow=False)
            # Meter
            cur_np = np.frombuffer(frame, dtype=np.int16)
            cur_rms = float(np.sqrt(np.mean(np.square(cur_np)))) if cur_np.size else 0.0
            await ws_mgr.broadcast({ 'type': 'meter', 'rms': round(cur_rms, 1) })

            # VAD on 30ms frames
            is_speech = False
            try:
                is_speech = capture.vad.is_speech(frame, capture.rate)
            except Exception:
                is_speech = False

            if is_speech:
                if not capture.in_speech:
                    # start segment; include pre-roll
                    capture.segment_frames = list(capture.pre_roll)
                    capture.silence_count = 0
                    capture.in_speech = True
                capture.segment_frames.append(frame)
            else:
                capture.pre_roll.append(frame)
                if capture.in_speech:
                    capture.segment_frames.append(frame)
                    capture.silence_count += 1
                    if capture.silence_count >= capture.tail_frames:
                        # end of segment -> process
                        segment_bytes = b''.join(capture.segment_frames)
                        duration = len(segment_bytes) / (capture.rate * 2)
                        capture.in_speech = False
                        capture.segment_frames = []
                        capture.silence_count = 0

                        if duration >= min_sec:
                            np_seg = np.frombuffer(segment_bytes, dtype=np.int16)
                            seg_rms = float(np.sqrt(np.mean(np.square(np_seg)))) if np_seg.size else 0.0
                            # normalize
                            target_rms = 1200.0
                            gain = (target_rms / seg_rms) if seg_rms > 0 else 1.0
                            gain = min(gain, 10.0)
                            boosted = np.clip(np_seg.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
                            wav = BytesIO()
                            with wave.open(wav, 'wb') as wf:
                                wf.setnchannels(capture.channels)
                                wf.setsampwidth(2)
                                wf.setframerate(capture.rate)
                                wf.writeframes(boosted.tobytes())
                            wav.seek(0)

                            text = await transcribe_bytes(wav.getvalue())
                            if text:
                                entry = {
                                    'timestamp': datetime.utcnow().isoformat(),
                                    'text': text
                                }
                                capture.transcripts.append(entry)
                                # autosave checkpoint
                                try:
                                    txt_path = os.path.join(capture.save_dir, f'session_{capture.session_started_at}.txt')
                                    with open(txt_path, 'a', encoding='utf-8') as f:
                                        f.write(f"[{entry['timestamp']}] {text}\n")
                                except Exception:
                                    pass

                                await ws_mgr.broadcast({ 'type': 'transcript', 'text': text, 'ts': entry['timestamp'] })
                                analysis = await analyze_text(text)
                                if analysis:
                                    # accumulate
                                    if analysis.get('important_points'):
                                        capture.points_store.extend(analysis['important_points'])
                                        # dedupe keep order
                                        capture.points_store = list(dict.fromkeys(capture.points_store))[-100:]
                                    if analysis.get('suggested_questions'):
                                        capture.questions_store.extend(analysis['suggested_questions'])
                                        capture.questions_store = list(dict.fromkeys(capture.questions_store))[-100:]
                                    if analysis.get('action_items'):
                                        capture.action_items_store.extend(analysis['action_items'])
                                        capture.action_items_store = list(dict.fromkeys(capture.action_items_store))[-100:]
                                    await ws_mgr.broadcast({ 'type': 'analysis', 'analysis': analysis, 'ts': entry['timestamp'] })
                        else:
                            logger.info(f"⏳ Skipped too-short VAD segment (sec={duration:.2f})")
            await asyncio.sleep(0.005)
    finally:
        try:
            if capture.stream:
                capture.stream.stop_stream()
                capture.stream.close()
        except Exception:
            pass
        try:
            if capture.py:
                capture.py.terminate()
        except Exception:
            pass
        logger.info("🛑 Capture ended")

@app.get('/')
async def index():
    html = """
    <!doctype html>
    <html>
    <head>
      <meta charset='utf-8'/>
      <title>Live Transcriber</title>
      <style>
        body { font-family: Inter, Arial, sans-serif; margin: 0; background:#0b0f19; color:#e6e8f0; }
        .wrap { max-width: 1100px; margin: 0 auto; padding: 32px; }
        .card { background:#121829; border:1px solid #1e2a44; border-radius:12px; padding:20px; margin-bottom:16px; }
        .row { display:flex; gap:16px; }
        .col { flex:1; }
        h1 { margin:0 0 12px 0; font-size: 24px; }
        select, button { background:#0b1220; color:#e6e8f0; border:1px solid #1e2a44; border-radius:8px; padding:8px 12px; }
        .btn { cursor:pointer; }
        .status { font-size:12px; opacity:0.8; }
        .meter { height:8px; background:#0b1220; border:1px solid #1e2a44; border-radius:8px; overflow:hidden; }
        .meter-fill { height:100%; width:0%; background:#29c070; transition: width 120ms linear; }
        .item { background:#0b1220; border:1px solid #1e2a44; border-radius:8px; padding:10px; margin:8px 0; }
        .muted { opacity:0.7; }
        a.link { color:#68a0ff; text-decoration:none; }
      </style>
    </head>
    <body>
      <div class='wrap'>
        <div class='card'>
          <h1>🎤 Live Transcriber</h1>
          <div class='status' id='status'>Ready</div>
          <div style='margin-top:10px; display:flex; gap:8px; align-items:center;'>
            <select id='device'></select>
            <button class='btn' onclick='startCapture()'>Start</button>
            <button class='btn' onclick='stopCapture()'>Stop</button>
            <a class='link' href='/download-txt' target='_blank'>Download transcript (.txt)</a>
            <a class='link' href='/download-json' target='_blank'>Download JSON</a>
            <a class='link' href='/download-html' target='_blank'>Download report (HTML)</a>
          </div>
          <div style='margin-top:10px;'>
            <div class='meter'><div id='meterFill' class='meter-fill'></div></div>
          </div>
        </div>

        <div class='row'>
          <div class='card col'>
            <h3>Transcripts</h3>
            <div id='trans'></div>
          </div>
          <div class='card col'>
            <h3>Important Points</h3>
            <div id='points' class='muted'>(waiting)</div>
          </div>
          <div class='card col'>
            <h3>Suggested Questions</h3>
            <div id='qs' class='muted'>(waiting)</div>
          </div>
        </div>
      </div>

      <script>
        let ws = null;
        let pointsStore = [];
        let qsStore = [];
        async function loadDevices(){
          try{
            const r = await fetch('/devices');
            const j = await r.json();
            const sel = document.getElementById('device');
            sel.innerHTML = '';
            j.devices.forEach(d => {
              const opt = document.createElement('option');
              opt.value = d.index; opt.textContent = `[${d.index}] ${d.name}`;
              sel.appendChild(opt);
            });
          }catch(e){ console.log(e); }
        }

        function connectWS(){
          ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
          ws.onopen = () => setStatus('Connected');
          ws.onclose = () => setStatus('Disconnected');
          ws.onmessage = (e) => {
            const d = JSON.parse(e.data);
            if(d.type === 'status') setStatus('Capture ' + d.status);
            if(d.type === 'meter') updateMeter(d.rms);
            if(d.type === 'transcript') addTrans(d.ts, d.text);
            if(d.type === 'analysis') addAnalysis(d.analysis);
          };
        }

        function setStatus(s){ document.getElementById('status').textContent = s; }
        function updateMeter(v){
          const pct = Math.max(0, Math.min(100, Math.round(v / 1500 * 100)));
          document.getElementById('meterFill').style.width = pct + '%';
        }
        function addTrans(ts, text){
          const el = document.getElementById('trans');
          const div = document.createElement('div');
          div.className = 'item';
          div.textContent = `[${new Date(ts).toLocaleTimeString()}] ${text}`;
          el.prepend(div);
        }
        function addAnalysis(a){
          if (a.important_points && a.important_points.length){
            pointsStore = pointsStore.concat(a.important_points).slice(-100);
            const el = document.getElementById('points');
            el.classList.remove('muted');
            el.innerHTML = '<ul>' + pointsStore.map(x => `<li>${x}</li>`).join('') + '</ul>';
          }
          if (a.suggested_questions && a.suggested_questions.length){
            qsStore = qsStore.concat(a.suggested_questions).slice(-100);
            const elq = document.getElementById('qs');
            elq.classList.remove('muted');
            elq.innerHTML = '<ul>' + qsStore.map(x => `<li>${x}</li>`).join('') + '</ul>';
          }
        }

        async function startCapture(){
          try{
            const idx = document.getElementById('device').value;
            await fetch('/start?device_index=' + encodeURIComponent(idx));
            setStatus('Starting...');
          }catch(e){console.log(e)}
        }
        async function stopCapture(){ await fetch('/stop'); }

        loadDevices();
        connectWS();
      </script>
    </body>
    </html>
    """
    return HTMLResponse(html)

@app.get('/devices')
async def devices():
    return { 'devices': list_input_devices() }

@app.get('/start')
async def start_capture(device_index: int | None = Query(default=None)):
    if capture.running:
        return { 'ok': True, 'status': 'already_running' }
    capture.device_index = device_index
    capture.running = True
    capture.task = asyncio.create_task(capture_loop())
    await ws_mgr.broadcast({ 'type': 'status', 'status': 'running', 'device_index': device_index })
    return { 'ok': True, 'device_index': device_index }

@app.get('/stop')
async def stop_capture():
    capture.running = False
    await ws_mgr.broadcast({ 'type': 'status', 'status': 'stopped' })
    return { 'ok': True }

@app.get('/health')
async def health():
    return {
        'status': 'healthy',
        'ts': datetime.utcnow().isoformat(),
        'running': capture.running
    }

@app.get('/download-txt')
async def download_txt():
    txt_path = os.path.join(capture.save_dir, f'session_{capture.session_started_at}.txt')
    if not os.path.exists(txt_path):
        with open(txt_path, 'w', encoding='utf-8') as f:
            pass
    return FileResponse(path=txt_path, filename=os.path.basename(txt_path), media_type='text/plain')

@app.get('/download-json')
async def download_json():
    payload = {
        'session_started_at': capture.session_started_at,
        'transcripts': capture.transcripts,
        'important_points': capture.points_store,
        'suggested_questions': capture.questions_store,
        'action_items': capture.action_items_store
    }
    tmp = BytesIO(json.dumps(payload, ensure_ascii=False, indent=2).encode('utf-8'))
    tmp.seek(0)
    return Response(content=tmp.read(), media_type='application/json', headers={'Content-Disposition': f"attachment; filename=report_{capture.session_started_at}.json"})

@app.get('/download-html')
async def download_html():
    # build beautiful HTML report
    title = 'Meeting Report'
    started = capture.session_started_at
    transcripts = capture.transcripts
    points = capture.points_store
    questions = capture.questions_store
    actions = capture.action_items_store

    body = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Meeting Report</title>",
        "<style>",
        "body{font-family:Inter,Arial,sans-serif;background:#0b0f19;color:#e6e8f0;margin:0;padding:24px;}",
        ".wrap{max-width:900px;margin:0 auto;}",
        ".card{background:#121829;border:1px solid #1e2a44;border-radius:12px;padding:20px;margin:12px 0;}",
        "h1{margin:0 0 8px 0;font-size:24px;}",
        "h2{margin:0 0 10px 0;font-size:18px;color:#9bb0ff;}",
        "ul{margin:8px 0 0 18px;}",
        "li{margin:6px 0;}",
        ".ts{opacity:.8;font-size:12px}",
        "</style></head><body><div class='wrap'>",
        f"<div class='card'><h1>{title}</h1><div class='ts'>Session: {started}</div></div>",
    ]

    if points:
        body.append("<div class='card'><h2>Important Points</h2><ul>")
        for p in points:
            body.append(f"<li>{p}</li>")
        body.append("</ul></div>")

    if actions:
        body.append("<div class='card'><h2>Action Items</h2><ul>")
        for a in actions:
            body.append(f"<li>{a}</li>")
        body.append("</ul></div>")

    if questions:
        body.append("<div class='card'><h2>Suggested Questions</h2><ul>")
        for q in questions:
            body.append(f"<li>{q}</li>")
        body.append("</ul></div>")

    if transcripts:
        body.append("<div class='card'><h2>Transcript</h2>")
        for t in transcripts:
            body.append(f"<div><span class='ts'>[{t['timestamp']}]</span> {t['text']}</div>")
        body.append("</div>")

    body.append("</div></body></html>")
    html = "".join(body)
    return Response(content=html, media_type='text/html', headers={'Content-Disposition': f"attachment; filename=report_{started}.html"})

@app.websocket('/ws')
async def ws(ws: WebSocket):
    await ws_mgr.connect(ws)
    # Auto-start capture when first client connects
    try:
        if not capture.running:
            capture.device_index = capture.device_index  # keep selection if set via /start
            capture.running = True
            capture.task = asyncio.create_task(capture_loop())
            await ws_mgr.broadcast({ 'type': 'status', 'status': 'running', 'device_index': capture.device_index })
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_mgr.disconnect(ws)
        # Auto-stop when no clients remain
        if not ws_mgr.conns and capture.running:
            capture.running = False
            await ws_mgr.broadcast({ 'type': 'status', 'status': 'stopped' })

if __name__ == '__main__':
    print("\nLive Transcriber running:")
    print("UI: http://localhost:8007")
    print("Health: http://localhost:8007/health")
    uvicorn.run(app, host='0.0.0.0', port=8007, reload=False, log_level='info')

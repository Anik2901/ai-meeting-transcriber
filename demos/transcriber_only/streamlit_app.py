#!/usr/bin/env python3
import os
import json
import time
import wave
from io import BytesIO
from collections import deque

import numpy as np
import streamlit as st
import requests
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av

# Load env from config.env if present
CONFIG = "config.env"
if os.path.exists(CONFIG):
    with open(CONFIG, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ[k] = v

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY missing. Add it to config.env or environment variables.")
    st.stop()

st.set_page_config(page_title="Live Transcriber (Single App)", layout="wide")
st.title("🎤 Live Transcriber - Single App")
st.caption("One command. In-browser mic. Whisper + GPT-4. No separate backend.")

# UI placeholders
left, mid, right = st.columns(3)
with left:
    st.subheader("Transcripts")
    trans_box = st.empty()
with mid:
    st.subheader("Important Points")
    points_box = st.empty()
with right:
    st.subheader("Suggested Questions")
    q_box = st.empty()

status = st.empty()

# Shared state
if "messages" not in st.session_state:
    st.session_state["messages"] = deque(maxlen=200)
if "points" not in st.session_state:
    st.session_state["points"] = deque(maxlen=100)
if "questions" not in st.session_state:
    st.session_state["questions"] = deque(maxlen=100)

# Whisper and GPT helpers
def transcribe_bytes(audio_bytes: bytes) -> str:
    try:
        files = { 'file': ('audio.wav', audio_bytes, 'audio/wav') }
        data = { 'model': 'whisper-1', 'response_format': 'text' }
        headers = { 'Authorization': f'Bearer {OPENAI_API_KEY}' }
        r = requests.post('https://api.openai.com/v1/audio/transcriptions', headers=headers, data=data, files=files, timeout=60)
        if r.status_code == 200:
            return r.text.strip()
        return ""
    except Exception:
        return ""

def analyze_text(text: str) -> dict:
    if not text.strip():
        return {}
    headers = { 'Authorization': f'Bearer {OPENAI_API_KEY}', 'Content-Type': 'application/json' }
    payload = {
        'model': 'gpt-4',
        'messages': [
            { 'role': 'system', 'content': (
                'Extract concrete, context-aware insights and high-quality follow-up questions from meeting speech. '
                'Return strict JSON with keys: important_points (list of concise bullets), '
                'suggested_questions (list of 3-5 specific, non-generic, actionable questions that reference details from the text).'
            )},
            { 'role': 'user', 'content': text }
        ],
        'temperature': 0.2,
        'max_tokens': 300
    }
    try:
        r = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            return {}
        content = r.json()['choices'][0]['message']['content']
        try:
            return json.loads(content)
        except Exception:
            return { 'important_points': [content], 'suggested_questions': [] }
    except Exception:
        return {}

# Audio processor
class MicProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        self.sample_rate_browser = 48000  # browser default
        self.target_rate = 16000
        self.channels = 1
        self.queue = deque()
        self.last_ts = time.time()
        self.window_sec = 5.0
        self.overlap_sec = 1.0
        self.min_sec = 0.6
        self.baseline_rms = None
        self.threshold = None
        self._accum_samples = 0

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        # Convert to mono int16 numpy at 48000
        pcm = frame.to_ndarray()  # shape: (channels, samples)
        if pcm.ndim == 2:
            pcm = pcm.mean(axis=0)
        pcm = pcm.astype(np.float32)
        # Normalize float32 -1..1 if needed
        if pcm.max() <= 1.0 and pcm.min() >= -1.0:
            pcm = (pcm * 32767.0)
        pcm = np.clip(pcm, -32768, 32767).astype(np.int16)

        # Downsample to 16k (simple decimation for demo)
        decimate = int(self.sample_rate_browser / self.target_rate)
        if decimate > 1:
            pcm = pcm[::decimate]
        self.queue.append(pcm.tobytes())
        self._accum_samples += pcm.size

        # Calibrate on first ~1s
        if self.baseline_rms is None and self._accum_samples >= self.target_rate:
            all_bytes = b''.join(list(self.queue))
            np_all = np.frombuffer(all_bytes, dtype=np.int16)
            self.baseline_rms = float(np.sqrt(np.mean(np.square(np_all)))) if np_all.size else 0.0
            self.threshold = max(15.0, self.baseline_rms * 2.0)

        # Process window
        win_samples = int(self.window_sec * self.target_rate)
        if self._accum_samples >= win_samples:
            all_bytes = b''.join(list(self.queue))
            duration = len(all_bytes) / (self.target_rate * 2)
            if duration >= self.min_sec:
                np_all = np.frombuffer(all_bytes, dtype=np.int16)
                cur_rms = float(np.sqrt(np.mean(np.square(np_all)))) if np_all.size else 0.0
                base = self.baseline_rms or 0.0
                thresh = max(self.threshold or 0.0, base * 2.5)
                if cur_rms >= thresh:
                    # Gain normalize
                    target_rms = 1000.0
                    gain = (target_rms / cur_rms) if cur_rms > 0 else 1.0
                    gain = min(gain, 10.0)
                    boosted = np.clip(np_all.astype(np.float32) * gain, -32768, 32767).astype(np.int16)

                    wav = BytesIO()
                    with wave.open(wav, 'wb') as wf:
                        wf.setnchannels(self.channels)
                        wf.setsampwidth(2)
                        wf.setframerate(self.target_rate)
                        wf.writeframes(boosted.tobytes())
                    wav.seek(0)

                    text = transcribe_bytes(wav.getvalue())
                    if text:
                        st.session_state["messages"].append(text)
                        analysis = analyze_text(text)
                        if analysis:
                            for p in analysis.get('important_points', []):
                                st.session_state["points"].append(p)
                            for q in analysis.get('suggested_questions', []):
                                st.session_state["questions"].append(q)
                # Keep overlap
                overlap_bytes = int(self.overlap_sec * self.target_rate * 2)
                tail = all_bytes[-overlap_bytes:] if len(all_bytes) > overlap_bytes else all_bytes
                self.queue.clear()
                if tail:
                    self.queue.append(tail)
                self._accum_samples = len(tail) // 2

        return frame

# WebRTC widget (single-process)
st.subheader("Microphone")
webrtc_ctx = webrtc_streamer(key="live-transcriber", mode=WebRtcMode.SENDONLY, audio_processor_factory=MicProcessor, media_stream_constraints={"audio": True, "video": False})

# Render live outputs
status.info("Connected" if webrtc_ctx.state.playing else "Waiting for mic permission...")

transcript_text = "\n".join(list(st.session_state["messages"])[-20:])
points_text = "\n".join([f"• {p}" for p in list(st.session_state["points"])[-10:]])
questions_text = "\n".join([f"• {q}" for q in list(st.session_state["questions"])[-10:]])

trans_box.write(transcript_text or "(waiting for speech)")
points_box.write(points_text or "(no points yet)")
q_box.write(questions_text or "(no questions yet)")

# Meeting Copilot – Investor One-Pager

## Vision
Real-time meeting copilot that turns live conversations into decisions, owners, and deadlines—automatically. No workflow change. Works with any meeting.

## Problem
- Meetings create decisions but lose accountability. Notes are inconsistent, late, or missing.
- Post-meeting summaries are slow and generic; teams miss follow‑ups and repeat discussions.
- IT needs privacy and simple deployment—no calendar sprawl or bots with broad permissions.

## Solution
- Live transcription and in‑meeting intelligence:
  - Real‑time transcript and captions
  - Instant Key Points, Action Items, and Suggested Questions
  - One‑click export (TXT/HTML/JSON) at meeting end
- Works with Zoom/Meet/Teams via mic capture; optional marketplace integrations later.
- Privacy‑first: local audio processing by default; configurable cloud analysis.

## Why Now
- Remote/hybrid is the default; meetings per employee ↑
- On‑device/edge AI and small models enable low-latency, high‑accuracy capture
- Procurement cares about privacy and low-touch deployment

## Product (Demo Flow)
1. Pick microphone → Start
2. Speak for 30–45s (project standup script)
3. Watch Key Points/Actions populate in real-time
4. Download branded HTML report and share

## Technical Architecture (v2)
- UI/API: FastAPI + WebSocket (single binary)
- Audio: PyAudio (16 kHz mono), VAD (webrtcvad)
- STT options:
  - Whisper API (cloud, higher accuracy)
  - Vosk (offline, low-latency, private)
- Analysis: OpenAI gpt‑4o‑mini (fast) or gpt‑4 (deeper) via modular adapter
- Storage: local session JSON + optional S3/GCS connector (enterprise)
- Exports: TXT / JSON / HTML (branded)

### Latency Targets
- First caption: < 600 ms (partial)
- Finalized segment: 1.5–2.5 s
- Analysis refresh: every 6–8 s or +50 words

### Privacy & Security
- Default: audio never leaves device; transcripts saved locally only
- Toggles: redaction, retention policy, per‑workspace keys
- Enterprise: SSO/SCIM, audit trail, DLP hooks (roadmap), SOC2

## Differentiation
- Real-time (in‑meeting) insights vs. post‑meeting dumps
- Works anywhere (mic capture) → zero vendor lock-in
- High-signal outputs (owner/due-date prompts, specific questions)
- Modular AI (swap STT/LLM providers) to balance cost/compliance/quality

## Market & ICP
- Primary: Product/Engineering, CX/Support, Sales/CS (recurring meetings)
- Bottom‑up adoption; expand to enterprise via admin controls

## Business Model
- SaaS per-seat/month
  - Starter: live transcription + exports
  - Pro: integrations (Slack/Jira/Linear), workspace memory, advanced prompts
  - Enterprise: SSO/SOC2, on‑prem/private cloud, admin governance

## KPIs
- Meetings processed / week
- Action items per meeting & 7‑day completion rate
- Report open rate / share rate
- Latency (p50/p90), WER (WER↓ over time via fine‑tuning)

## Traction Plan (12 Months)
- Q1: Hybrid STT (offline/online), multilingual, speaker labels (basic)
- Q2: Calendar/Slack/Jira integrations, templates per team
- Q3: Cross-meeting memory and analytics dashboard
- Q4: Enterprise compliance (DLP, SOC2), on‑prem option

## The Ask
- $X seed to hire 3 engineers + 1 GTM; 12 months runway
- Goals: 100 logos, 30K meetings/month, >50% weekly retention, SOC2 Type I

---

### Live Demo Script (45s)
“We agreed to ship auth by next Friday. Sarah owns frontend, Arjun owns API. We still need a decision on pricing tiers before launch. Risks: data migration and mobile parity. Next: confirm copy by Wednesday and schedule QA.”

Expected: 3–5 key points, 2 action items with implied owners, 2–3 follow‑up questions.

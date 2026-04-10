# Groq Interview Voice Agent (Windows)

Desktop app that listens to interview questions, transcribes them with Groq, generates concise interview answers, and speaks responses in real time.

## Features
- Real-time interview Q&A assistant with voice input and voice output.
- STT: `whisper-large-v3-turbo`
- LLM: `llama-3.1-8b-instant`
- TTS: `canopylabs/orpheus-v1-english`
- Two output modes:
  - `virtual_cable` (recommended for Google Meet/Zoom routing)
  - `speakers` (local testing/fallback)
- Two input modes:
  - `system_audio` (default)
  - `microphone`

## Project Files
- `main.py` - main application (Tkinter UI + audio pipeline + Groq calls)
- `install.bat` - one-click installer (sets up Python/dependencies and `config.json`)
- `start.bat` - starts the app using `.venv` or embedded `.python`
- `config.json.example` - configuration template
- `requirements.txt` - Python dependencies

## Requirements
- Windows
- Internet connection
- Groq API key
- Optional but recommended: VB-Audio Virtual Cable (`virtual_cable` mode)

## Quick Start
1. Run `install.bat`.
2. Run `start.bat`.
3. In the setup window, fill:
   - `groq_api_key`
   - `job_title`
   - `company_name`
   - `resume_summary`
   - `output_mode` and audio device settings
4. Click **Start Interview**.

The app saves settings to `config.json`.

## Configuration
Default template (`config.json.example`):

```json
{
  "groq_api_key": "paste_your_groq_api_key_here",
  "job_title": "Senior Python Developer",
  "company_name": "Example Company",
  "resume_summary": "7+ years building Python backend systems, APIs, cloud deployments, and mentoring teams.",
  "output_mode": "virtual_cable",
  "input_mode": "system_audio",
  "loopback_device_contains": "",
  "tts_voice": "troy",
  "tts_speed": 1.0,
  "silence_timeout_sec": 1.8,
  "vad_threshold": 0.012
}
```

Notes:
- Required fields: `groq_api_key`, `job_title`, `company_name`, `resume_summary`
- `silence_timeout_sec` controls how long silence is required before answering
- `vad_threshold` controls speech sensitivity (higher = less sensitive)

## Google Meet Setup (Recommended)
1. Use headphones.
2. In Google Meet:
   - Microphone: `CABLE Output (VB-Audio Virtual Cable)`
   - Speaker: your normal headphones/speakers
3. In app config/UI:
   - `output_mode`: `virtual_cable`
   - `output_device_name`: the exact `CABLE Input ...` device if auto-detect picks the wrong endpoint
   - `input_mode`: `system_audio` (default)
4. Do not set Meet/Zoom speaker output to VB-Cable. The app listens to your normal playback path and separately speaks into the virtual cable for the browser mic.

## Troubleshooting
- Dependency issues: run `install.bat` again.
- No output in meeting app:
  - Verify meeting microphone points to VB-Cable output.
  - Verify app log says `TTS playback device:` with a VB-Cable endpoint, not `default speakers`.
  - Check the startup `Audio device diagnostics:` block and confirm the device tagged `browser-mic` is what Meet/Zoom is using.
  - Temporarily switch app `output_mode` to `speakers` to verify TTS works.
- App not capturing interviewer audio:
  - Ensure interviewer sound is on your default Windows playback device.
  - Keep interview app and meeting app on the same playback path.
- Too sensitive / not sensitive enough:
  - Increase `vad_threshold` for less sensitivity.
  - Decrease `vad_threshold` for more sensitivity.
- Responds too early/late:
  - Tune `silence_timeout_sec` (usually `1.5` to `2.2`).

## Run Without Batch Scripts
If you prefer manual Python execution:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py
```

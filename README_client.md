# Groq Interview Voice Agent (Windows)

This desktop app listens to interview questions, transcribes them with Groq Whisper, generates strong interview answers with Groq LLM, and speaks the answer with Groq Orpheus voice.

## What this package does
- Listens continuously through **system audio** (your default playback device via loopback capture).
- Waits for ~1.8 seconds of silence before answering.
- Uses:
  - STT: `whisper-large-v3-turbo`
  - LLM: `llama-3.1-8b-instant`
  - TTS: `canopylabs/orpheus-v1-english`
- Sends voice output to:
  - `virtual_cable` (for Google Meet), or
  - `speakers` (fallback/testing mode).

## Interview behavior guarantees
- No barge-in: once the app starts generating or speaking an answer, it finishes that turn before listening for the next question.
- Loopback-safe playback: in system-audio mode, input is muted during TTS when needed to avoid self-transcription loops.
- Consistency: response memory is preserved only for completed turns.
- Clarification-first on unclear prompts: model instructions explicitly ask for clarification when a question is cut off or unclear.

## 1) Install prerequisites (one-time)
1. Install Python 3.10+ (check "Add Python to PATH" during install):
   - https://www.python.org/downloads/windows/
2. Create a Groq account + API key:
   - https://console.groq.com/
3. Install VB-Audio Virtual Cable (recommended for Google Meet):
   - https://vb-audio.com/Cable/

## 2) Install the app
1. Extract this folder anywhere (for example Desktop).
2. Double-click `install.bat`.
3. Wait until it says **Install complete**.

## 3) Configure in app (no manual config editing)
1. Double-click `start.bat`.
2. A setup window opens automatically.
3. Fill:
   - Groq API key
   - Job title
   - Company
   - Resume summary
   - TTS speed
   - Output mode (`virtual_cable` or `speakers`)
4. Click **Start Interview**.
5. The app saves your settings to `config.json` automatically.

## 4) Run interview mode
1. In the setup window, click **Start Interview**.
2. You will see:
   - Status (`Listening...`, `Transcribing...`, etc.)
   - Live logs
   - Stop button (you can reconfigure and start again)
3. Console and window will show:
   - `Question heard: ...`
   - `Answering: ...`

## 5) Google Meet audio setup (important)
1. Use **headphones** (reduces echo and feedback).
2. In Google Meet:
   - Microphone input: **CABLE Output (VB-Audio Virtual Cable)**
   - Speaker output: your headphones
3. Keep your real physical mic muted in Meet.
4. In the setup window, set Output mode to `"virtual_cable"`.

### Standard interview mode (headphone-safe)
- Interviewer audio is captured directly from system playback (headphone output) automatically.
- Keep `output_mode: "virtual_cable"` so your generated answer is sent to Meet without being re-captured from speakers.

## Troubleshooting
- `Rate limit reached - wait a bit and retry`:
  - Pause briefly, then continue.
- No sound in Meet:
  - Recheck Meet microphone input and Windows sound devices.
  - Test once with `output_mode: "speakers"` to confirm TTS works.
- System audio not capturing expected audio:
  - Ensure interviewer audio is routed to your default Windows playback device.
  - Keep the interview app and Meet/Zoom on the same playback device.
  - Re-run `install.bat` to ensure audio dependencies are installed.
- Error about `fromstring`/`frombuffer` in system audio capture:
  - This is a NumPy/SoundCard compatibility mismatch.
  - Run `install.bat` again (the project now pins NumPy below 2.3).
- Too sensitive / not sensitive enough:
  - Adjust `vad_threshold` in `config.json`:
    - Higher value = less sensitive
    - Lower value = more sensitive
- Responds too early or too late:
  - Adjust `silence_timeout_sec` (recommended 1.5 to 2.2).

## Notes
- Keep internet stable throughout the interview.
- Default voice speed is `1.0` (most reliable for current Orpheus endpoint).

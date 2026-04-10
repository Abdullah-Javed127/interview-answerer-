import io
import json
import logging
import queue
import random
import threading
import time
import ctypes
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
import tkinter as tk
from groq import Groq
from tkinter import messagebox, scrolledtext, ttk

try:
    import soundcard as sc
except Exception:  # pragma: no cover - optional import at runtime
    sc = None


LOG_FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("interview-agent")


CONFIG_PATH = Path("config.json")
FRAME_DURATION_SEC = 0.2
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_TIMEOUT_DEFAULT = 1.8
VAD_THRESHOLD_DEFAULT = 0.005
MIN_QUESTION_SECONDS = 0.8
MAX_QUESTION_SECONDS = 25.0
MAX_HISTORY_MESSAGES = 8
MAX_PROMPT_RECENT_MESSAGES = 4
BARGE_IN_ENABLED = False
INTERRUPT_CONSECUTIVE_FRAMES = 2
PREEMPT_CONSECUTIVE_FRAMES = 3
OUTPUT_CHUNK_SAMPLES = 1024
ANSWER_MAX_TOKENS = 190
ANSWER_MAX_WORDS = 170
ANSWER_MAX_CHARS = 980
TTS_RETRY_MAX_WORDS = 90
TTS_RETRY_MAX_CHARS = 560
INTERRUPT_PREROLL_FRAMES = 5
INTERRUPT_MIN_QUESTION_SECONDS = 0.25
MEMORY_SUMMARY_MAX_CHARS = 700
MODEL_HISTORY_TEXT_MAX_CHARS = 320

# ── Interrupt mic VAD ──
# Simpler, more sensitive threshold for mic-based interrupt detection.
# This uses RMS-only (no ZCR/spectral) since we just need to know
# "is someone speaking?" not "is this clean speech?"
INTERRUPT_MIC_RMS_THRESHOLD = 0.01

# ── Robust VAD constants ──
# For 200ms frames at 16kHz = 3200 samples:
#   Speech ZCR: ~100-800 crossings (depends on pitch and harmonics)
#   Fan/hiss noise: ~1500+ crossings (broadband)
VAD_ZCR_MIN = 15            # Reject near-DC rumble (very few crossings)
VAD_ZCR_MAX = 1200          # Reject broadband hiss/noise (very many crossings)
VAD_SPEECH_BAND_HZ = (250, 3800)  # Human speech frequency band
VAD_SPEECH_ENERGY_RATIO_MIN = 0.20  # Min fraction of energy in speech band
NOISE_FLOOR_ALPHA = 0.05  # EMA smoothing for adaptive noise floor
NOISE_FLOOR_MULTIPLIER = 3.0  # Effective threshold = noise_floor * this
NOISE_FLOOR_INITIAL = 0.005  # Initial noise floor estimate
NOISE_FLOOR_MAX = 0.08  # Cap the noise floor so it doesn't adapt to speech

# ── Whisper hallucination guard ──
WHISPER_HALLUCINATION_BLOCKLIST = {
    "thank you.", "thanks for watching.", "thanks for watching!",
    "thank you for watching.", "thank you for watching!",
    "please subscribe.", "subscribe.", "silence.", "music.",
    "bye.", "bye!", "you.", "okay.", "so.", "yeah.",
    "the end.", "...", "hmm.", "uh.", "um.",
    "thanks for listening.", "thanks for listening!",
    "see you next time.", "see you next time!",
    "like and subscribe.", "please like and subscribe.",
    "subtitles by the amara.org community",
}
WHISPER_MIN_WORDS = 3
WHISPER_NO_SPEECH_PROB_MAX = 0.6
WHISPER_AVG_LOGPROB_MIN = -1.0

# ── Filler clips ──
FILLER_PHRASES = [
    "Oh, uh,",
    "Mm, right,",
    "Ah, sure,",
    "Um,",
    "Oh, okay,",
]


def compute_zcr(chunk: np.ndarray) -> int:
    """Count zero-crossings in an audio chunk — voice has moderate ZCR, noise is very high."""
    signs = np.signbit(chunk)
    crossings = int(np.sum(np.abs(np.diff(signs.astype(np.int8)))))
    return crossings


def compute_speech_band_energy_ratio(chunk: np.ndarray, sample_rate: int) -> float:
    """Fraction of spectral energy in the human speech band (250-3800 Hz) vs full spectrum."""
    n = len(chunk)
    if n < 16:
        return 0.0
    spectrum = np.abs(np.fft.rfft(chunk))
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    total_energy = float(np.sum(spectrum ** 2)) + 1e-12
    lo, hi = VAD_SPEECH_BAND_HZ
    mask = (freqs >= lo) & (freqs <= hi)
    speech_energy = float(np.sum(spectrum[mask] ** 2))
    return speech_energy / total_energy


def is_voice_frame(
    chunk: np.ndarray,
    rms: float,
    effective_threshold: float,
    sample_rate: int = SAMPLE_RATE,
) -> bool:
    """Multi-signal voice activity detection: RMS + ZCR + spectral energy ratio.

    Returns True only if all three signals suggest human speech, not noise.
    """
    if rms < effective_threshold:
        return False

    zcr = compute_zcr(chunk)
    if zcr < VAD_ZCR_MIN or zcr > VAD_ZCR_MAX:
        logger.debug("VAD rejected: ZCR=%d (range %d-%d), rms=%.4f", zcr, VAD_ZCR_MIN, VAD_ZCR_MAX, rms)
        return False

    ratio = compute_speech_band_energy_ratio(chunk, sample_rate)
    if ratio < VAD_SPEECH_ENERGY_RATIO_MIN:
        logger.debug("VAD rejected: spectral_ratio=%.3f (min %.3f), rms=%.4f", ratio, VAD_SPEECH_ENERGY_RATIO_MIN, rms)
        return False

    logger.debug("VAD accepted: rms=%.4f zcr=%d ratio=%.3f", rms, zcr, ratio)
    return True


def is_voice_frame_simple(chunk: np.ndarray, rms_threshold: float = INTERRUPT_MIC_RMS_THRESHOLD) -> bool:
    """Simplified voice detection using only RMS energy.

    Used for interrupt detection via microphone where the audio may be
    compressed/processed (video call audio through speakers) and may not
    pass the stricter ZCR/spectral checks.
    """
    rms = float(np.sqrt(np.mean(np.square(chunk))) + 1e-12)
    return rms > rms_threshold


def is_whisper_hallucination(text: str, transcription_obj: Any = None) -> bool:
    """Check if a Whisper transcription is likely a hallucination (noise/silence artifact)."""
    cleaned = text.strip().lower()
    if not cleaned:
        return True

    # Blocklist check
    if cleaned in WHISPER_HALLUCINATION_BLOCKLIST:
        logger.info("Whisper hallucination blocked (blocklist): %r", text)
        return True

    # Too few words
    word_count = len(cleaned.split())
    if word_count < WHISPER_MIN_WORDS:
        logger.info("Whisper hallucination blocked (only %d words): %r", word_count, text)
        return True

    # Confidence check from verbose_json segments
    if transcription_obj is not None:
        segments = getattr(transcription_obj, "segments", None)
        if segments and len(segments) > 0:
            avg_no_speech = sum(
                s.get("no_speech_prob", 0) if isinstance(s, dict) else getattr(s, "no_speech_prob", 0)
                for s in segments
            ) / len(segments)
            avg_logprob = sum(
                s.get("avg_logprob", 0) if isinstance(s, dict) else getattr(s, "avg_logprob", 0)
                for s in segments
            ) / len(segments)
            if avg_no_speech > WHISPER_NO_SPEECH_PROB_MAX:
                logger.info(
                    "Whisper hallucination blocked (no_speech_prob=%.2f): %r",
                    avg_no_speech, text
                )
                return True
            if avg_logprob < WHISPER_AVG_LOGPROB_MIN:
                logger.info(
                    "Whisper hallucination blocked (avg_logprob=%.2f): %r",
                    avg_logprob, text
                )
                return True

    return False


@dataclass
class AppConfig:
    groq_api_key: str
    job_title: str
    company_name: str
    resume_summary: str
    output_mode: str
    input_mode: str = "system_audio"
    loopback_device_contains: str = ""
    tts_voice: str = "troy"
    tts_speed: float = 1.0
    silence_timeout_sec: float = SILENCE_TIMEOUT_DEFAULT
    vad_threshold: float = VAD_THRESHOLD_DEFAULT
    input_device_name: str = ""   # Empty = auto-detect / default
    output_device_name: str = ""  # Empty = auto-detect / default


def enumerate_audio_devices() -> Dict[str, List[Tuple[int, str]]]:
    """Return dict with 'input' and 'output' lists of (device_index, device_name)."""
    devices = sd.query_devices()
    inputs: List[Tuple[int, str]] = []
    outputs: List[Tuple[int, str]] = []
    for idx, dev in enumerate(devices):
        name = str(dev.get("name", f"Device {idx}"))
        if dev.get("max_input_channels", 0) >= 1:
            inputs.append((idx, name))
        if dev.get("max_output_channels", 0) >= 1:
            outputs.append((idx, name))
    return {"input": inputs, "output": outputs}


def looks_like_virtual_cable_device(name: str) -> bool:
    normalized = name.strip().lower()
    if not normalized:
        return False
    return any(token in normalized for token in ("cable input", "cable output", "vb-audio", "virtual cable"))


def get_hostapi_name(dev: Dict[str, Any], hostapis: Optional[List[Dict[str, Any]]] = None) -> str:
    hostapi_index = dev.get("hostapi")
    if hostapi_index is None:
        return ""
    try:
        hostapis = hostapis if hostapis is not None else sd.query_hostapis()
        if 0 <= int(hostapi_index) < len(hostapis):
            return str(hostapis[int(hostapi_index)].get("name", "")).strip()
    except Exception:
        pass
    return ""


def describe_audio_output_device(
    idx: Optional[int], dev: Optional[Dict[str, Any]], hostapis: Optional[List[Dict[str, Any]]] = None
) -> str:
    if idx is None or dev is None:
        return "default speakers"
    name = str(dev.get("name", f"Device {idx}"))
    hostapi_name = get_hostapi_name(dev, hostapis)
    if hostapi_name:
        return f'"{name}" via {hostapi_name} (index {idx})'
    return f'"{name}" (index {idx})'


def describe_audio_device(idx: int, dev: Dict[str, Any], hostapis: Optional[List[Dict[str, Any]]] = None) -> str:
    name = str(dev.get("name", f"Device {idx}"))
    hostapi_name = get_hostapi_name(dev, hostapis)
    input_channels = int(dev.get("max_input_channels", 0) or 0)
    output_channels = int(dev.get("max_output_channels", 0) or 0)
    directions: List[str] = []
    if input_channels >= 1:
        directions.append(f"in:{input_channels}")
    if output_channels >= 1:
        directions.append(f"out:{output_channels}")
    directions_text = ", ".join(directions) if directions else "no-io"
    if hostapi_name:
        return f'"{name}" via {hostapi_name} (index {idx}, {directions_text})'
    return f'"{name}" (index {idx}, {directions_text})'


def choose_best_virtual_cable_output() -> Tuple[Optional[int], str, str]:
    devices = sd.query_devices()
    try:
        hostapis = sd.query_hostapis()
    except Exception:
        hostapis = []

    candidates: List[Tuple[int, int, str, str]] = []
    for idx, dev in enumerate(devices):
        if dev.get("max_output_channels", 0) < 1:
            continue
        raw_name = str(dev.get("name", ""))
        lowered_name = raw_name.lower()
        if not looks_like_virtual_cable_device(raw_name):
            continue

        hostapi_name = get_hostapi_name(dev, hostapis).lower()
        score = 0
        if "cable input" in lowered_name:
            score += 100
        if "virtual cable" in lowered_name:
            score += 30
        if "vb-audio" in lowered_name:
            score += 20
        if "wasapi" in hostapi_name:
            score += 10
        if "mme" in hostapi_name:
            score -= 5

        candidates.append((score, idx, raw_name, describe_audio_output_device(idx, dev, hostapis)))

    if not candidates:
        return None, "", "default speakers"

    _, idx, raw_name, label = max(candidates, key=lambda item: (item[0], -item[1]))
    return idx, raw_name, label


def resample_audio_mono(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate <= 0 or dst_rate <= 0 or src_rate == dst_rate or audio.size == 0:
        return audio
    src_positions = np.arange(audio.shape[0], dtype=np.float64)
    duration = audio.shape[0] / float(src_rate)
    target_length = max(1, int(round(duration * dst_rate)))
    dst_positions = np.linspace(0, max(audio.shape[0] - 1, 0), num=target_length, dtype=np.float64)
    resampled = np.interp(dst_positions, src_positions, audio.astype(np.float64))
    return resampled.astype(np.float32, copy=False)


def default_config_raw() -> Dict[str, Any]:
    return {
        "groq_api_key": "",
        "job_title": "Senior Python Developer",
        "company_name": "Example Company",
        "resume_summary": "7+ years building Python backend systems, APIs, cloud deployments, and mentoring teams.",
        "output_mode": "virtual_cable",
        "input_mode": "system_audio",
        "loopback_device_contains": "",
        "tts_voice": "troy",
        "tts_speed": 1.0,
        "silence_timeout_sec": SILENCE_TIMEOUT_DEFAULT,
        "vad_threshold": VAD_THRESHOLD_DEFAULT,
        "input_device_name": "",
        "output_device_name": "",
    }


def load_config_raw(path: Path) -> Dict[str, Any]:
    raw = default_config_raw()
    if not path.exists():
        return raw
    try:
        with path.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception:
        return raw
    if isinstance(loaded, dict):
        raw.update(loaded)
    return raw


def build_config(raw: Dict[str, Any], require_required: bool = True) -> AppConfig:
    required_fields = ["groq_api_key", "job_title", "company_name", "resume_summary"]
    if require_required:
        missing = [key for key in required_fields if not str(raw.get(key, "")).strip()]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")

    output_mode = str(raw.get("output_mode", "virtual_cable")).strip().lower()
    if output_mode not in {"virtual_cable", "speakers"}:
        raise ValueError('output_mode must be either "virtual_cable" or "speakers".')

    input_mode = str(raw.get("input_mode", "system_audio")).strip().lower()
    if input_mode not in {"microphone", "system_audio"}:
        raise ValueError('input_mode must be either "microphone" or "system_audio".')

    try:
        speed = float(raw.get("tts_speed", 1.0))
    except Exception as exc:
        raise ValueError("tts_speed must be a number.") from exc
    if speed <= 0:
        raise ValueError("tts_speed must be > 0.")

    try:
        silence_timeout = float(raw.get("silence_timeout_sec", SILENCE_TIMEOUT_DEFAULT))
    except Exception as exc:
        raise ValueError("silence_timeout_sec must be a number.") from exc
    if silence_timeout < 1.2 or silence_timeout > 4.0:
        raise ValueError("silence_timeout_sec should be between 1.2 and 4.0 seconds.")

    try:
        vad_threshold = float(raw.get("vad_threshold", VAD_THRESHOLD_DEFAULT))
    except Exception as exc:
        raise ValueError("vad_threshold must be a number.") from exc
    if vad_threshold <= 0:
        raise ValueError("vad_threshold must be > 0.")

    return AppConfig(
        groq_api_key=str(raw.get("groq_api_key", "")).strip(),
        job_title=str(raw.get("job_title", "")).strip(),
        company_name=str(raw.get("company_name", "")).strip(),
        resume_summary=str(raw.get("resume_summary", "")).strip(),
        output_mode=output_mode,
        input_mode=input_mode,
        loopback_device_contains=str(raw.get("loopback_device_contains", "")).strip(),
        tts_voice=str(raw.get("tts_voice", "troy")).strip() or "troy",
        tts_speed=speed,
        silence_timeout_sec=silence_timeout,
        vad_threshold=vad_threshold,
        input_device_name=str(raw.get("input_device_name", "")).strip(),
        output_device_name=str(raw.get("output_device_name", "")).strip(),
    )


def save_config(cfg: AppConfig, path: Path) -> None:
    raw = {
        "groq_api_key": cfg.groq_api_key,
        "job_title": cfg.job_title,
        "company_name": cfg.company_name,
        "resume_summary": cfg.resume_summary,
        "output_mode": cfg.output_mode,
        "input_mode": cfg.input_mode,
        "loopback_device_contains": cfg.loopback_device_contains,
        "tts_voice": cfg.tts_voice,
        "tts_speed": cfg.tts_speed,
        "silence_timeout_sec": cfg.silence_timeout_sec,
        "vad_threshold": cfg.vad_threshold,
        "input_device_name": cfg.input_device_name,
        "output_device_name": cfg.output_device_name,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)


def build_system_prompt(cfg: AppConfig) -> str:
    template = (
        "You are interviewing for [JOB_TITLE] at [COMPANY_NAME]. "
        "Background: [RESUME_SUMMARY].\n"
        "Answer like a real candidate on a live call.\n"
        "- Be concise: usually 1 to 4 sentences.\n"
        "- Be natural and direct, not formal or robotic.\n"
        "- Continue the existing conversation consistently.\n"
        "- After the opening hello, skip pleasantries and answer immediately.\n"
        "- Never say you are an AI."
    )
    return (
        template.replace("[JOB_TITLE]", cfg.job_title)
        .replace("[COMPANY_NAME]", cfg.company_name)
        .replace("[RESUME_SUMMARY]", cfg.resume_summary)
    )


def is_rate_limit_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    text = str(exc).lower()
    return status_code == 429 or "rate limit" in text or "too many requests" in text


def is_tts_payload_too_large_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "413" in text
        or "payload too large" in text
        or "request too large" in text
        or "tokens per minute" in text
    )


def compact_model_text(text: str, max_chars: int = MODEL_HISTORY_TEXT_MAX_CHARS) -> str:
    cleaned = " ".join(str(text).replace("\n", " ").replace("\r", " ").split())
    if len(cleaned) <= max_chars:
        return cleaned
    shortened = cleaned[:max_chars].rstrip(" ,.;:")
    last_sentence_break = max(shortened.rfind(". "), shortened.rfind("? "), shortened.rfind("! "))
    if last_sentence_break >= max_chars // 2:
        shortened = shortened[: last_sentence_break + 1]
    return shortened


def format_history_for_memory(history_items: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for item in history_items:
        role = item.get("role", "")
        content = compact_model_text(item.get("content", ""))
        if not content:
            continue
        speaker = "Interviewer" if role == "user" else "Candidate"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


class InterviewUI:
    def __init__(self, on_start, on_stop, on_exit, initial_raw: Dict[str, Any]):
        self.root = tk.Tk()
        self.root.title("Groq Interview Voice Agent")
        self.root.geometry("900x700")
        self.root.minsize(820, 640)
        self.root.protocol("WM_DELETE_WINDOW", on_exit)
        self.status_var = tk.StringVar(value="Ready")
        self.output_mode_var = tk.StringVar(
            value=str(initial_raw.get("output_mode", "virtual_cable")).strip().lower()
            or "virtual_cable"
        )
        self.input_mode_var = tk.StringVar(
            value=str(initial_raw.get("input_mode", "system_audio")).strip().lower()
            or "system_audio"
        )
        self.input_device_var = tk.StringVar(
            value=str(initial_raw.get("input_device_name", "")).strip() or "Auto-detect"
        )
        self.output_device_var = tk.StringVar(
            value=str(initial_raw.get("output_device_name", "")).strip() or "Auto-detect"
        )
        self.form_widgets: List[Any] = []

        # Enumerate available devices for dropdowns
        try:
            dev_map = enumerate_audio_devices()
            self._input_device_names = ["Auto-detect"] + [name for _, name in dev_map["input"]]
            self._output_device_names = ["Auto-detect"] + [name for _, name in dev_map["output"]]
        except Exception:
            self._input_device_names = ["Auto-detect"]
            self._output_device_names = ["Auto-detect"]

        shell = ttk.Frame(self.root, padding=10)
        shell.pack(fill="both", expand=True)

        ttk.Label(
            shell,
            text="Groq Voice Interview Agent",
            font=("Segoe UI", 14, "bold"),
        ).pack(anchor="w")

        top_row = ttk.Frame(shell)
        top_row.pack(fill="x", pady=(8, 8))
        ttk.Label(top_row, text="Status:", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Label(top_row, textvariable=self.status_var).pack(side="left", padx=6)
        self.start_button = ttk.Button(top_row, text="Start Interview", command=on_start)
        self.start_button.pack(side="right", padx=(8, 0))
        self.stop_button = ttk.Button(top_row, text="Stop", command=on_stop, state="disabled")
        self.stop_button.pack(side="right")

        setup = ttk.LabelFrame(shell, text="Interview Setup", padding=10)
        setup.pack(fill="x")
        setup.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(setup, text="Groq API Key").grid(row=row, column=0, sticky="w", pady=4)
        self.api_key_entry = ttk.Entry(setup, show="*")
        self.api_key_entry.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=4)
        self.api_key_entry.insert(0, str(initial_raw.get("groq_api_key", "")))
        self.form_widgets.append(self.api_key_entry)

        row += 1
        ttk.Label(setup, text="Job Title").grid(row=row, column=0, sticky="w", pady=4)
        self.job_title_entry = ttk.Entry(setup)
        self.job_title_entry.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=4)
        self.job_title_entry.insert(0, str(initial_raw.get("job_title", "")))
        self.form_widgets.append(self.job_title_entry)

        row += 1
        ttk.Label(setup, text="Company").grid(row=row, column=0, sticky="w", pady=4)
        self.company_entry = ttk.Entry(setup)
        self.company_entry.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=4)
        self.company_entry.insert(0, str(initial_raw.get("company_name", "")))
        self.form_widgets.append(self.company_entry)

        row += 1
        ttk.Label(setup, text="TTS Speed").grid(row=row, column=0, sticky="w", pady=4)
        self.tts_speed_entry = ttk.Entry(setup, width=12)
        self.tts_speed_entry.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=4)
        self.tts_speed_entry.insert(0, str(initial_raw.get("tts_speed", 1.0)))
        self.form_widgets.append(self.tts_speed_entry)

        row += 1
        ttk.Label(setup, text="Output").grid(row=row, column=0, sticky="w", pady=4)
        self.output_mode_combo = ttk.Combobox(
            setup,
            values=["virtual_cable", "speakers"],
            textvariable=self.output_mode_var,
            state="readonly",
            width=20,
        )
        self.output_mode_combo.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=4)
        self.form_widgets.append(self.output_mode_combo)

        row += 1
        ttk.Label(setup, text="Input Mode").grid(row=row, column=0, sticky="w", pady=4)
        self.input_mode_combo = ttk.Combobox(
            setup,
            values=["microphone", "system_audio"],
            textvariable=self.input_mode_var,
            state="readonly",
            width=20,
        )
        self.input_mode_combo.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=4)
        self.form_widgets.append(self.input_mode_combo)

        row += 1
        ttk.Label(setup, text="Input Device").grid(row=row, column=0, sticky="w", pady=4)
        self.input_device_combo = ttk.Combobox(
            setup,
            values=self._input_device_names,
            textvariable=self.input_device_var,
            state="readonly",
            width=40,
        )
        self.input_device_combo.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=4)
        self.form_widgets.append(self.input_device_combo)

        row += 1
        ttk.Label(setup, text="Output Device").grid(row=row, column=0, sticky="w", pady=4)
        self.output_device_combo = ttk.Combobox(
            setup,
            values=self._output_device_names,
            textvariable=self.output_device_var,
            state="readonly",
            width=40,
        )
        self.output_device_combo.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=4)
        self.form_widgets.append(self.output_device_combo)

        row += 1
        ttk.Label(setup, text="Resume Summary").grid(
            row=row, column=0, sticky="nw", pady=(8, 4)
        )
        self.resume_text = scrolledtext.ScrolledText(setup, height=6, wrap=tk.WORD)
        self.resume_text.grid(row=row, column=1, sticky="ew", padx=(8, 0), pady=(8, 4))
        self.resume_text.insert(tk.END, str(initial_raw.get("resume_summary", "")))
        self.form_widgets.append(self.resume_text)

        tips = ttk.Frame(shell)
        tips.pack(fill="x", pady=(8, 4))
        ttk.Label(
            tips,
            text="Select your audio devices above, or leave as Auto-detect for defaults.",
        ).pack(anchor="w")
        ttk.Label(
            tips,
            text="If VB-Cable is not installed, output will auto-fallback to speakers.",
        ).pack(anchor="w")

        body = ttk.Frame(shell, padding=(0, 8, 0, 0))
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="Live Activity", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.log_box = scrolledtext.ScrolledText(
            body, height=12, wrap=tk.WORD, font=("Consolas", 10)
        )
        self.log_box.pack(fill="both", expand=True, pady=(4, 0))
        self.log_box.configure(state="disabled")

    def get_config_raw(self) -> Dict[str, Any]:
        input_dev = self.input_device_var.get().strip()
        output_dev = self.output_device_var.get().strip()
        return {
            "groq_api_key": self.api_key_entry.get().strip(),
            "job_title": self.job_title_entry.get().strip(),
            "company_name": self.company_entry.get().strip(),
            "resume_summary": self.resume_text.get("1.0", tk.END).strip(),
            "output_mode": self.output_mode_var.get().strip().lower(),
            "input_mode": self.input_mode_var.get().strip().lower() or "system_audio",
            "loopback_device_contains": "",
            "tts_voice": "troy",
            "tts_speed": self.tts_speed_entry.get().strip() or "1.0",
            "silence_timeout_sec": SILENCE_TIMEOUT_DEFAULT,
            "vad_threshold": VAD_THRESHOLD_DEFAULT,
            "input_device_name": "" if input_dev == "Auto-detect" else input_dev,
            "output_device_name": "" if output_dev == "Auto-detect" else output_dev,
        }

    def set_running(self, is_running: bool) -> None:
        self.start_button.configure(state="disabled" if is_running else "normal")
        self.stop_button.configure(state="normal" if is_running else "disabled")
        for widget in self.form_widgets:
            target_state = "disabled" if is_running else "normal"
            if isinstance(widget, ttk.Combobox):
                target_state = "disabled" if is_running else "readonly"
            widget.configure(state=target_state)

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def append_log(self, line: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert(tk.END, f"{line}\n")
        self.log_box.see(tk.END)
        self.log_box.configure(state="disabled")

    def show_error(self, title: str, text: str) -> None:
        messagebox.showerror(title, text)


class InterviewAgent:
    def __init__(self, cfg: AppConfig, ui: InterviewUI):
        self.cfg = cfg
        self.ui = ui
        self.client = Groq(api_key=cfg.groq_api_key, timeout=20.0, max_retries=1)
        self.system_prompt = build_system_prompt(cfg)
        self.stop_event = threading.Event()
        self.audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=120)
        self.ui_queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self.stream: Optional[sd.InputStream] = None
        self.capture_thread: Optional[threading.Thread] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.response_thread: Optional[threading.Thread] = None
        self.history: List[Dict[str, str]] = []
        self.memory_summary: str = ""
        self.turn_count = 0
        self.processing_lock = threading.Lock()
        self.history_lock = threading.Lock()
        self.turn_lock = threading.Lock()
        self.active_response_workers = 0
        self.frame_samples = int(SAMPLE_RATE * FRAME_DURATION_SEC)
        self.output_device_name_resolved = ""
        self.output_device_label = "default speakers"
        self.output_device = self._select_output_device()
        self.input_device, self.loopback_speaker_name, self.input_source_label = (
            self._select_input_source()
        )
        self.capture_own_output_risk = self._detect_output_loopback_overlap()
        self.is_speaking = threading.Event()
        self.turn_in_progress = threading.Event()
        self.interrupt_requested = threading.Event()
        self.interrupt_voice_frames = 0
        self.preempt_requested = threading.Event()
        self.preempt_voice_frames = 0
        self.last_response_interrupted = False
        self.current_turn_id = 0
        self.interrupt_preroll: deque[np.ndarray] = deque(maxlen=INTERRUPT_PREROLL_FRAMES)
        self.interrupt_buffer_chunks: List[np.ndarray] = []
        self.pending_priority_chunks: List[np.ndarray] = []
        self._pending_priority_lock = threading.Lock()

        # ── Adaptive noise floor ──
        self.noise_floor = NOISE_FLOOR_INITIAL
        self._noise_floor_lock = threading.Lock()

        # ── Pre-generated filler clips (populated on start) ──
        self.filler_clips: List[bytes] = []
        self._filler_generation_done = threading.Event()

        # ── Secondary microphone stream for interrupt detection ──
        # When input_mode is system_audio with capture_own_output_risk,
        # the system audio loopback captures the bot's own TTS output,
        # so we can't use it for interrupt detection. Instead, we open
        # a separate microphone stream that only listens during TTS
        # playback to detect when the interviewer is speaking.
        self._mic_interrupt_stream: Optional[sd.InputStream] = None
        self._mic_interrupt_available = False
        self._mic_interrupt_voice_frames = 0

    def _can_detect_interruptions(self) -> bool:
        """Return whether live playback interruption detection is enabled."""
        if not BARGE_IN_ENABLED:
            return False
        if self.cfg.input_mode == "microphone":
            return True
        # When we have a mic interrupt stream, we can always detect.
        if self._mic_interrupt_available:
            return True
        # Fallback: only safe when output is not routed to default speakers.
        return self.cfg.input_mode == "system_audio" and not self.capture_own_output_risk

    def _select_input_source(self) -> Tuple[Optional[int], Any, str]:
        if self.cfg.input_mode == "microphone":
            # If user selected a specific input device by name, resolve its index
            if self.cfg.input_device_name:
                devices = sd.query_devices()
                for idx, dev in enumerate(devices):
                    if dev.get("max_input_channels", 0) >= 1 and dev.get("name") == self.cfg.input_device_name:
                        label = f'microphone: "{dev["name"]}"'
                        self._emit_ui("log", f"Input device selected: {label}")
                        return idx, None, label
                self._emit_ui("log", f'Configured input device "{self.cfg.input_device_name}" not found. Using default.')
            return None, None, "microphone (fallback mode, default input device)"

        if sc is None:
            raise RuntimeError(
                "System audio capture backend is missing. Run install.bat to install requirements."
            )

        # If user selected a specific loopback source by name, use it
        if self.cfg.input_device_name:
            try:
                mics = sc.all_microphones(include_loopback=True)
                for mic in mics:
                    if self.cfg.input_device_name in str(mic.name):
                        label = f'system audio loopback from "{mic.name}" (user-selected)'
                        self._emit_ui("log", f"Input source: {label}")
                        return None, str(mic.name), label
            except Exception:
                pass
            self._emit_ui("log", f'Configured input device "{self.cfg.input_device_name}" not found for loopback. Using default.')

        try:
            speaker = sc.default_speaker()
        except Exception as exc:
            raise RuntimeError(f"Could not get default speaker for loopback capture: {exc}") from exc
        if speaker is None:
            raise RuntimeError("No default speaker found for system audio capture.")

        label = f'system audio loopback from "{speaker.name}"'
        return None, str(speaker.name), label

    def _update_noise_floor(self, rms: float) -> None:
        """Update the adaptive noise floor using an exponential moving average.
        Only updates when we are NOT speaking and NOT processing a turn."""
        if self.is_speaking.is_set() or self.turn_in_progress.is_set():
            return
        with self._noise_floor_lock:
            self.noise_floor = (
                NOISE_FLOOR_ALPHA * rms + (1 - NOISE_FLOOR_ALPHA) * self.noise_floor
            )
            self.noise_floor = min(self.noise_floor, NOISE_FLOOR_MAX)

    def _get_effective_threshold(self) -> float:
        """Dynamic VAD threshold: max of configured threshold and adaptive noise floor * multiplier."""
        with self._noise_floor_lock:
            adaptive = self.noise_floor * NOISE_FLOOR_MULTIPLIER
        return max(self.cfg.vad_threshold, adaptive)

    def _reset_interrupt_capture(self) -> None:
        self.interrupt_requested.clear()
        self.interrupt_voice_frames = 0
        self.interrupt_preroll.clear()
        self.interrupt_buffer_chunks = []

    def _capture_interrupt_chunk(self, chunk: np.ndarray, voiced: bool) -> None:
        if not voiced:
            self.interrupt_preroll.append(chunk.copy())
            if self.interrupt_buffer_chunks:
                self.interrupt_buffer_chunks.append(chunk.copy())
            return

        if not self.interrupt_buffer_chunks:
            self.interrupt_buffer_chunks = [buf.copy() for buf in self.interrupt_preroll]
        self.interrupt_buffer_chunks.append(chunk.copy())
        self.interrupt_preroll.append(chunk.copy())

    def _promote_interrupt_buffer_to_pending(self) -> None:
        if not self.interrupt_buffer_chunks:
            return
        with self._pending_priority_lock:
            self.pending_priority_chunks.extend(self.interrupt_buffer_chunks)
        self.interrupt_buffer_chunks = []

    def _take_pending_priority_chunks(self) -> List[np.ndarray]:
        with self._pending_priority_lock:
            if not self.pending_priority_chunks:
                return []
            chunks = self.pending_priority_chunks
            self.pending_priority_chunks = []
            return chunks

    def _advance_turn_id(self) -> int:
        with self.turn_lock:
            self.current_turn_id += 1
            return self.current_turn_id

    def _get_current_turn_id(self) -> int:
        with self.turn_lock:
            return self.current_turn_id

    def _is_turn_current(self, turn_id: int) -> bool:
        return not self.stop_event.is_set() and turn_id == self._get_current_turn_id()

    def _interrupt_active_turn(self, reason: str) -> None:
        if not BARGE_IN_ENABLED:
            return
        if self.interrupt_requested.is_set():
            return
        self.last_response_interrupted = True
        self._advance_turn_id()
        self.interrupt_requested.set()
        self.preempt_requested.set()
        self._emit_ui("log", reason)

    def _mic_interrupt_callback(self, indata, frames, _time_info, status) -> None:
        """Callback for the secondary microphone stream used for interrupt detection.

        This only processes audio when the bot is speaking. It uses a simple
        RMS-only VAD to detect if the interviewer is talking over the bot.
        """
        if status:
            logger.debug("Mic interrupt stream status: %s", status)
        if self.stop_event.is_set():
            return

        # Only check during TTS playback
        if not self.is_speaking.is_set():
            self._mic_interrupt_voice_frames = 0
            return

        # Already interrupted — nothing more to do
        if self.interrupt_requested.is_set():
            return

        chunk = indata[:, 0].copy().astype(np.float32)
        voiced = is_voice_frame_simple(chunk)

        if voiced:
            self._mic_interrupt_voice_frames += 1
            if self._mic_interrupt_voice_frames >= INTERRUPT_CONSECUTIVE_FRAMES:
                self._interrupt_active_turn(
                    "[MIC] Interviewer speech detected during playback. Stopping current answer."
                )
        else:
            self._mic_interrupt_voice_frames = max(0, self._mic_interrupt_voice_frames - 1)

    def _start_mic_interrupt_stream(self) -> None:
        """Open a secondary microphone stream for interrupt detection.

        This is used when input_mode is system_audio and there's a risk of
        capturing our own TTS output. The mic stream runs independently and
        only triggers interrupt detection during TTS playback.
        """
        if self._mic_interrupt_stream is not None:
            return
        try:
            self._mic_interrupt_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                blocksize=self.frame_samples,
                dtype="float32",
                device=None,  # default microphone
                callback=self._mic_interrupt_callback,
            )
            self._mic_interrupt_stream.start()
            self._mic_interrupt_available = True
            self._emit_ui(
                "log",
                "Secondary microphone opened for interrupt/barge-in detection.",
            )
        except Exception as exc:
            self._mic_interrupt_available = False
            self._emit_ui(
                "log",
                f"Could not open microphone for interrupt detection: {exc}. "
                "Barge-in may not work during playback.",
            )
            logger.debug("Mic interrupt stream open failed", exc_info=True)

    def _stop_mic_interrupt_stream(self) -> None:
        """Close the secondary microphone stream."""
        if self._mic_interrupt_stream is not None:
            try:
                self._mic_interrupt_stream.stop()
                self._mic_interrupt_stream.close()
            except Exception:
                logger.debug("Mic interrupt stream close warning", exc_info=True)
            finally:
                self._mic_interrupt_stream = None
                self._mic_interrupt_available = False

    def _ingest_input_chunk(self, chunk: np.ndarray) -> None:
        if self.stop_event.is_set():
            return
        if chunk.size == 0:
            return
        chunk = np.asarray(chunk, dtype=np.float32)
        rms = float(np.sqrt(np.mean(np.square(chunk))) + 1e-12)
        effective_threshold = self._get_effective_threshold()
        can_detect_interruptions = self._can_detect_interruptions()

        # Update noise floor during quiet periods
        voiced = is_voice_frame(chunk, rms, effective_threshold)
        if not voiced:
            self._update_noise_floor(rms)

        # Barge-in is disabled. Ignore incoming audio while we are generating or
        # speaking so the current turn always completes before a new one starts.
        if not BARGE_IN_ENABLED and self.turn_in_progress.is_set():
            return

        if self.turn_in_progress.is_set() and not self.is_speaking.is_set() and can_detect_interruptions:
            if voiced:
                self.preempt_voice_frames += 1
                if self.preempt_voice_frames >= PREEMPT_CONSECUTIVE_FRAMES:
                    self._interrupt_active_turn(
                        "New interviewer speech detected while a response was being prepared. Cancelling stale turn."
                    )
            else:
                self.preempt_voice_frames = max(0, self.preempt_voice_frames - 1)
        elif not self.turn_in_progress.is_set():
            self.preempt_voice_frames = 0
            self.preempt_requested.clear()

        # In system-audio mode, when output goes to default speakers, loopback captures our
        # own TTS. Drop input while speaking to prevent self-transcription loops.
        # NOTE: Interrupt detection is handled by the secondary mic stream (_mic_interrupt_callback),
        # not by this path. We still drop the loopback audio to prevent self-transcription.
        if self.is_speaking.is_set() and self.capture_own_output_risk:
            # If an interrupt was already triggered (by the mic stream), let audio through
            # so the post-interrupt speech enters the main queue for transcription.
            if self.interrupt_requested.is_set():
                # Interrupt was detected — stop dropping, let audio flow to queue
                pass
            else:
                return

        if self.is_speaking.is_set() and can_detect_interruptions and not self.capture_own_output_risk:
            # Non-risk path: system audio doesn't capture own TTS (e.g. VB-Cable),
            # or input_mode is microphone. Use the main stream for interrupt detection.
            self._capture_interrupt_chunk(chunk, voiced)
            if voiced:
                self.interrupt_voice_frames += 1
                if self.interrupt_voice_frames >= INTERRUPT_CONSECUTIVE_FRAMES:
                    self._interrupt_active_turn(
                        "Interviewer speech detected during playback. Stopping current answer."
                    )
            else:
                self.interrupt_voice_frames = max(0, self.interrupt_voice_frames - 1)
            return
        try:
            self.audio_queue.put_nowait(chunk)
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.audio_queue.put_nowait(chunk)
            except queue.Full:
                logger.debug("Audio queue still full; dropping newest chunk.")

    def _system_audio_capture_worker(self) -> None:
        if not self.loopback_speaker_name:
            return
        com_initialized = False
        try:
            # soundcard uses COM internally on Windows; initialize COM in this worker thread.
            hr = ctypes.windll.ole32.CoInitializeEx(None, 0x2)
            # RPC_E_CHANGED_MODE means COM is already initialized with another model.
            if hr not in (0, 1, -2147417850):
                raise RuntimeError(f"COM init failed (HRESULT=0x{hr & 0xFFFFFFFF:08x}).")
            com_initialized = hr in (0, 1)

            loopback_recorder = sc.get_microphone(
                id=self.loopback_speaker_name, include_loopback=True
            )
            with loopback_recorder.recorder(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                blocksize=self.frame_samples,
            ) as recorder:
                while not self.stop_event.is_set():
                    data = recorder.record(numframes=self.frame_samples)
                    if data is None:
                        continue
                    if getattr(data, "ndim", 1) > 1:
                        chunk = data[:, 0]
                    else:
                        chunk = data
                    self._ingest_input_chunk(chunk)
        except Exception as exc:
            msg = str(exc).lower()
            if "binary mode of fromstring is removed" in msg or "use frombuffer instead" in msg:
                self._emit_ui(
                    "log",
                    (
                        "System audio capture dependency mismatch detected (SoundCard vs NumPy). "
                        "Please run install.bat to reinstall pinned dependencies."
                    ),
                )
            self._emit_ui("log", f"System audio capture stopped: {exc}")
            self._emit_ui("status", "Stopped")
            self.stop_event.set()
        finally:
            if com_initialized:
                try:
                    ctypes.windll.ole32.CoUninitialize()
                except Exception:
                    logger.debug("COM uninitialize warning", exc_info=True)

    def _select_output_device(self) -> Optional[int]:
        devices = sd.query_devices()
        try:
            hostapis = sd.query_hostapis()
        except Exception:
            hostapis = []

        # If user explicitly selected an output device by name, use it
        if self.cfg.output_device_name:
            for idx, dev in enumerate(devices):
                dev_name = str(dev.get("name", "")).strip()
                if dev.get("max_output_channels", 0) >= 1 and dev_name.lower() == self.cfg.output_device_name.lower():
                    self.output_device_name_resolved = dev_name
                    self.output_device_label = describe_audio_output_device(idx, dev, hostapis)
                    self._emit_ui("log", f"Output device selected: {self.output_device_label}.")
                    return idx
            self._emit_ui("log", f'Configured output device "{self.cfg.output_device_name}" not found. Falling back to auto-detect.')

        if self.cfg.output_mode == "speakers":
            self.output_device_name_resolved = ""
            self.output_device_label = "default speakers"
            self._emit_ui("log", "Output mode: speakers (default output device).")
            return None

        idx, raw_name, label = choose_best_virtual_cable_output()
        if idx is not None:
            self.output_device_name_resolved = raw_name
            self.output_device_label = label
            self._emit_ui("log", f"Output mode: virtual_cable (using {label}).")
            return idx

        self.output_device_name_resolved = ""
        self.output_device_label = "default speakers"
        self._emit_ui(
            "log",
            "Virtual cable requested but no VB-Cable output device found. Falling back to speakers.",
        )
        return None

    def _detect_output_loopback_overlap(self) -> bool:
        if self.cfg.input_mode != "system_audio":
            return False
        if self.output_device is None:
            return True
        if not self.output_device_name_resolved or not self.loopback_speaker_name:
            return False

        output_name = self.output_device_name_resolved.strip().lower()
        loopback_name = str(self.loopback_speaker_name).strip().lower()
        if not output_name or not loopback_name:
            return False

        if output_name in loopback_name or loopback_name in output_name:
            return True
        if looks_like_virtual_cable_device(output_name) and looks_like_virtual_cable_device(loopback_name):
            return True
        return False

    def _emit_routing_guidance(self) -> None:
        if self.cfg.output_mode != "virtual_cable":
            return

        if self.output_device is None:
            self._emit_ui(
                "log",
                "WARNING: virtual_cable mode is not active. The app fell back to speakers, so Meet/Zoom will not receive bot audio until you choose a VB-Cable output device.",
            )
            return

        self._emit_ui(
            "log",
            "Meet/Zoom must use 'CABLE Output (VB-Audio Virtual Cable)' as the microphone. This app only chooses where the bot speaks; it cannot switch the browser mic automatically.",
        )

        if self.cfg.input_mode == "system_audio" and looks_like_virtual_cable_device(self.loopback_speaker_name or ""):
            self._emit_ui(
                "log",
                "WARNING: system-audio capture is currently loopbacking a virtual cable endpoint. Keep Meet/Zoom speaker output on your real headphones or speakers, not on VB-Cable, or the app will listen to the wrong path.",
            )

    def _emit_audio_device_diagnostics(self) -> None:
        try:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            default_input, default_output = sd.default.device
        except Exception as exc:
            self._emit_ui("log", f"Audio diagnostics unavailable: {exc}")
            return

        self._emit_ui("log", "Audio device diagnostics:")
        if self.cfg.output_mode == "virtual_cable":
            self._emit_ui("log", "Browser microphone should be: CABLE Output (VB-Audio Virtual Cable)")
        else:
            self._emit_ui("log", "Browser microphone should stay on your normal microphone because app output mode is speakers.")

        if self.cfg.output_device_name:
            self._emit_ui("log", f'Configured output device name: "{self.cfg.output_device_name}"')
        if self.cfg.input_device_name:
            self._emit_ui("log", f'Configured input device name: "{self.cfg.input_device_name}"')

        loopback_name = str(self.loopback_speaker_name or "").strip().lower()
        selected_output_name = self.output_device_name_resolved.strip().lower()

        for idx, dev in enumerate(devices):
            input_channels = int(dev.get("max_input_channels", 0) or 0)
            output_channels = int(dev.get("max_output_channels", 0) or 0)
            if input_channels < 1 and output_channels < 1:
                continue

            name = str(dev.get("name", "")).strip()
            lowered_name = name.lower()
            markers: List[str] = []

            if idx == default_input:
                markers.append("default-in")
            if idx == default_output:
                markers.append("default-out")
            if self.output_device is not None and idx == self.output_device:
                markers.append("tts-out")
            if self.cfg.output_device_name and lowered_name == self.cfg.output_device_name.lower():
                markers.append("configured-out")
            if self.cfg.input_mode == "microphone" and self.input_device is not None and idx == self.input_device:
                markers.append("capture-in")
            if self.cfg.input_device_name and lowered_name == self.cfg.input_device_name.lower():
                markers.append("configured-in")
            if looks_like_virtual_cable_device(name):
                markers.append("vb-cable")
            if loopback_name and (lowered_name in loopback_name or loopback_name in lowered_name):
                markers.append("loopback-target")
            if selected_output_name and lowered_name == selected_output_name:
                markers.append("resolved-out")
            if "cable output" in lowered_name:
                markers.append("browser-mic")

            marker_text = f" [{' | '.join(markers)}]" if markers else ""
            self._emit_ui("log", f"  - {describe_audio_device(idx, dev, hostapis)}{marker_text}")

    def _emit_ui(self, kind: str, text: str) -> None:
        self.ui_queue.put((kind, text))
        if kind == "log":
            logger.info(text)

    def _audio_callback(self, indata, frames, _time_info, status) -> None:
        if status:
            logger.warning("Audio callback status: %s", status)
        chunk = indata[:, 0].copy()
        self._ingest_input_chunk(chunk)

    def _generate_filler_clips_background(self) -> None:
        """Pre-generate short filler audio clips via Groq TTS in a background thread."""
        clips = []
        for phrase in FILLER_PHRASES:
            if self.stop_event.is_set():
                break
            try:
                response = self.client.audio.speech.create(
                    model="canopylabs/orpheus-v1-english",
                    voice=self.cfg.tts_voice,
                    input=phrase,
                    response_format="wav",
                    speed=1.0,
                )
                if hasattr(response, "read"):
                    audio_bytes = response.read()
                elif hasattr(response, "content"):
                    audio_bytes = response.content
                else:
                    continue
                if audio_bytes:
                    clips.append(audio_bytes)
            except Exception as exc:
                logger.debug("Filler clip generation failed for %r: %s", phrase, exc)
                # If rate-limited, stop trying more fillers
                if is_rate_limit_error(exc):
                    break
                continue
        self.filler_clips = clips
        self._filler_generation_done.set()
        if clips:
            self._emit_ui("log", f"Pre-generated {len(clips)} interrupt filler clips.")
        else:
            self._emit_ui("log", "Could not generate filler clips (will skip on interrupt).")

    def _play_interrupt_filler(self) -> None:
        """Play a random short filler clip on interrupt to sound natural."""
        if not self.filler_clips:
            return
        clip = random.choice(self.filler_clips)
        try:
            self._play_raw_audio(clip)
        except Exception as exc:
            logger.debug("Filler playback failed: %s", exc)

    def _play_raw_audio(self, audio_bytes: bytes) -> None:
        """Play audio bytes without interrupt detection (used for short fillers)."""
        data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        playback_data = data if data.ndim == 1 else data[:, 0]
        if playback_data.size == 0:
            return
        stream, playback_data, sample_rate = self._open_output_stream(playback_data, int(sample_rate))
        try:
            stream.start()
            cursor = 0
            total = playback_data.shape[0]
            while cursor < total and not self.stop_event.is_set():
                end = min(cursor + OUTPUT_CHUNK_SAMPLES, total)
                chunk = playback_data[cursor:end]
                if chunk.shape[0] < OUTPUT_CHUNK_SAMPLES:
                    chunk = np.pad(chunk, (0, OUTPUT_CHUNK_SAMPLES - chunk.shape[0]))
                stream.write(chunk.reshape(-1, 1))
                cursor = end
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

    def _get_output_device_default_samplerate(self) -> Optional[int]:
        try:
            device_info = sd.query_devices(self.output_device, "output")
        except Exception:
            return None
        try:
            default_rate = int(round(float(device_info.get("default_samplerate", 0) or 0)))
        except Exception:
            return None
        return default_rate if default_rate > 0 else None

    def _open_output_stream(
        self, playback_data: np.ndarray, sample_rate: int
    ) -> Tuple[sd.OutputStream, np.ndarray, int]:
        try:
            stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=self.output_device,
                blocksize=OUTPUT_CHUNK_SAMPLES,
            )
            return stream, playback_data, sample_rate
        except sd.PortAudioError as exc:
            message = str(exc).lower()
            if "invalid sample rate" not in message:
                raise

            fallback_rate = self._get_output_device_default_samplerate()
            if not fallback_rate or fallback_rate == sample_rate:
                raise

            self._emit_ui(
                "log",
                f"Playback sample rate {sample_rate} Hz was rejected by {self.output_device_label}; retrying at {fallback_rate} Hz.",
            )
            resampled = resample_audio_mono(playback_data, sample_rate, fallback_rate)
            stream = sd.OutputStream(
                samplerate=fallback_rate,
                channels=1,
                dtype="float32",
                device=self.output_device,
                blocksize=OUTPUT_CHUNK_SAMPLES,
            )
            return stream, resampled, fallback_rate

    def start(self) -> None:
        self._emit_ui("status", "Listening...")
        self._emit_ui("log", f"Input mode: {self.cfg.input_mode}, output mode: {self.cfg.output_mode}")
        self._emit_ui("log", f"Loopback capture source: {self.input_source_label}")
        self._emit_ui("log", f"Default loopback speaker: {self.loopback_speaker_name or 'n/a'}")
        self._emit_ui(
            "log",
            f"TTS playback device: {self.output_device_label}",
        )
        self._emit_audio_device_diagnostics()
        self._emit_routing_guidance()
        if self.capture_own_output_risk:
            self._emit_ui(
                "log",
                "Speaker loopback risk detected; system audio will be muted during TTS to prevent self-loop.",
            )
        self.worker_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.worker_thread.start()

        if BARGE_IN_ENABLED and self.capture_own_output_risk:
            self._start_mic_interrupt_stream()
        if BARGE_IN_ENABLED:
            filler_thread = threading.Thread(
                target=self._generate_filler_clips_background, daemon=True
            )
            filler_thread.start()

        if self.cfg.input_mode == "system_audio":
            self.capture_thread = threading.Thread(
                target=self._system_audio_capture_worker,
                daemon=True,
            )
            self.capture_thread.start()
        else:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                blocksize=self.frame_samples,
                dtype="float32",
                device=self.input_device,
                callback=self._audio_callback,
            )
            self.stream.start()
        self._emit_ui(
            "log",
            f"Input stream started ({self.input_source_label}) at {SAMPLE_RATE} Hz. Waiting for interviewer question...",
        )

    def stop(self) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        self._emit_ui("status", "Stopping...")
        self._stop_mic_interrupt_stream()
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            logger.exception("Error while closing audio stream.")
        finally:
            self.stream = None

        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        self.capture_thread = None

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2.0)
        if self.response_thread and self.response_thread.is_alive():
            self.response_thread.join(timeout=2.0)

    def _process_audio_chunk(
        self,
        chunk: np.ndarray,
        speaking: bool,
        silence_sec: float,
        speech_chunks: List[np.ndarray],
        speech_sec: float,
    ) -> Tuple[bool, float, List[np.ndarray], float]:
        rms = float(np.sqrt(np.mean(np.square(chunk))) + 1e-12)
        effective_threshold = self._get_effective_threshold()
        voiced = is_voice_frame(chunk, rms, effective_threshold)

        if voiced:
            if not speaking:
                speaking = True
                speech_chunks = []
                speech_sec = 0.0
                silence_sec = 0.0
                self._emit_ui("status", "Listening... (speech detected)")
            speech_chunks.append(chunk)
            speech_sec += FRAME_DURATION_SEC
            silence_sec = 0.0
            if speech_sec >= MAX_QUESTION_SECONDS:
                self._dispatch_detected_question(speech_chunks)
                speaking = False
                speech_chunks = []
                speech_sec = 0.0
                silence_sec = 0.0
            return speaking, silence_sec, speech_chunks, speech_sec

        if speaking:
            speech_chunks.append(chunk)
            speech_sec += FRAME_DURATION_SEC
            silence_sec += FRAME_DURATION_SEC
            if silence_sec >= self.cfg.silence_timeout_sec:
                if speech_sec >= MIN_QUESTION_SECONDS:
                    self._dispatch_detected_question(speech_chunks)
                speaking = False
                speech_chunks = []
                speech_sec = 0.0
                silence_sec = 0.0
                self._emit_ui("status", "Listening...")

        return speaking, silence_sec, speech_chunks, speech_sec

    def _audio_worker(self) -> None:
        speaking = False
        silence_sec = 0.0
        speech_chunks: List[np.ndarray] = []
        speech_sec = 0.0

        while not self.stop_event.is_set():
            pending_chunks = self._take_pending_priority_chunks()
            if pending_chunks:
                if not speaking:
                    self._emit_ui("status", "Listening... (interruption captured)")
                for chunk in pending_chunks:
                    (
                        speaking,
                        silence_sec,
                        speech_chunks,
                        speech_sec,
                    ) = self._process_audio_chunk(
                        chunk, speaking, silence_sec, speech_chunks, speech_sec
                    )
                continue

            try:
                chunk = self.audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            speaking, silence_sec, speech_chunks, speech_sec = self._process_audio_chunk(
                chunk, speaking, silence_sec, speech_chunks, speech_sec
            )

    def _dispatch_detected_question(self, chunks: List[np.ndarray]) -> None:
        if self.stop_event.is_set():
            return

        waveform = np.concatenate(chunks).astype(np.float32)
        if waveform.size < int(SAMPLE_RATE * MIN_QUESTION_SECONDS):
            return

        turn_id = self._advance_turn_id()
        with self.processing_lock:
            self.active_response_workers += 1
            self.turn_in_progress.set()
        self.preempt_requested.clear()
        self.preempt_voice_frames = 0
        response_thread = threading.Thread(
            target=self._handle_detected_question,
            args=(turn_id, waveform),
            daemon=True,
        )
        self.response_thread = response_thread
        response_thread.start()

    def _handle_detected_question(self, turn_id: int, waveform: np.ndarray) -> None:
        try:
            if not self._is_turn_current(turn_id):
                return

            self._emit_ui("status", "Transcribing question...")
            try:
                question = self._transcribe_audio(waveform)
            except Exception as exc:
                if self._is_turn_current(turn_id):
                    self._emit_api_error("Transcription failed", exc)
                return
            if not self._is_turn_current(turn_id):
                return

            if not question:
                self._emit_ui("log", "No clear question detected. Continuing to listen.")
                return

            self._emit_ui("log", f"Question heard: {question}")
            self._emit_ui("status", "Generating answer...")

            try:
                answer = self._generate_answer(question, turn_id)
            except Exception as exc:
                if self._is_turn_current(turn_id):
                    self._emit_api_error("Answer generation failed", exc)
                return
            if not self._is_turn_current(turn_id):
                return

            if not answer:
                self._emit_ui("log", "LLM returned empty answer. Skipping response.")
                return

            self._emit_ui("log", f"Answering: {answer}")
            self._emit_ui("status", "Speaking answer...")

            try:
                tts_bytes = self._text_to_speech(answer)
            except Exception as exc:
                if self._is_turn_current(turn_id):
                    self._emit_api_error("Speech synthesis failed", exc)
                return
            if not self._is_turn_current(turn_id):
                return

            try:
                interrupted = self._play_audio_bytes(tts_bytes, turn_id)
            except Exception as exc:
                if self._is_turn_current(turn_id):
                    self._emit_api_error("Speech playback failed", exc)
                return

            if interrupted or not self._is_turn_current(turn_id):
                return

            self._record_turn(question, answer)
            self._emit_ui("status", "Listening...")
        finally:
            with self.processing_lock:
                self.active_response_workers = max(0, self.active_response_workers - 1)
                no_active_workers = self.active_response_workers == 0
            if no_active_workers:
                self.turn_in_progress.clear()
                self.preempt_requested.clear()
                self.preempt_voice_frames = 0
                if self._is_turn_current(turn_id):
                    self._clear_audio_queue()

    def _transcribe_audio(self, waveform: np.ndarray) -> str:
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, waveform, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        wav_buffer.seek(0)
        file_tuple = ("question.wav", wav_buffer.read(), "audio/wav")

        transcription = self.client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=file_tuple,
            language="en",
            response_format="verbose_json",
            temperature=0.0,
        )

        text = getattr(transcription, "text", "") or ""
        text = text.strip()

        # ── Whisper hallucination guard ──
        if is_whisper_hallucination(text, transcription):
            self._emit_ui("log", f"Filtered likely hallucination: {text!r}")
            return ""

        return text

    def _compact_tts_text(self, text: str, max_words: int, max_chars: int) -> str:
        cleaned = " ".join(str(text).replace("\n", " ").replace("\r", " ").split())
        if not cleaned:
            return ""
        words = cleaned.split()
        if len(words) > max_words:
            cleaned = " ".join(words[:max_words]).rstrip(" ,.;:")
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars].rstrip(" ,.;:")
        if cleaned and cleaned[-1] not in ".!?":
            cleaned = f"{cleaned}."
        return cleaned

    def _generate_answer(self, question: str, turn_id: int) -> str:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        with self.history_lock:
            memory_summary = compact_model_text(
                self.memory_summary, max_chars=MEMORY_SUMMARY_MAX_CHARS
            )
            recent_history = list(self.history[-MAX_PROMPT_RECENT_MESSAGES:])
        if memory_summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"Conversation memory:\n{memory_summary}",
                }
            )
        messages.extend(recent_history)
        messages.append({"role": "user", "content": compact_model_text(question, max_chars=500)})

        stream = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.55,
            max_tokens=ANSWER_MAX_TOKENS,
            stream=True,
        )

        answer_parts: List[str] = []
        for chunk in stream:
            if not self._is_turn_current(turn_id):
                return ""
            try:
                delta = chunk.choices[0].delta
            except Exception:
                continue
            content = getattr(delta, "content", None)
            if not content:
                continue
            if isinstance(content, str):
                answer_parts.append(content)
                continue
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, str):
                        answer_parts.append(item)
                    elif isinstance(item, dict):
                        text = item.get("text")
                        if text:
                            answer_parts.append(text)
                    else:
                        text = getattr(item, "text", None)
                        if text:
                            answer_parts.append(text)

        answer = "".join(answer_parts)
        answer = self._compact_tts_text(answer, ANSWER_MAX_WORDS, ANSWER_MAX_CHARS)
        return answer

    def _record_turn(self, question: str, answer: str) -> None:
        latest_exchange: List[Dict[str, str]] = []
        with self.history_lock:
            compact_question = compact_model_text(question)
            compact_answer = compact_model_text(answer)
            self.history.append({"role": "user", "content": compact_question})
            self.history.append({"role": "assistant", "content": compact_answer})
            self.turn_count += 1
            if len(self.history) > MAX_HISTORY_MESSAGES:
                self.history = self.history[-MAX_HISTORY_MESSAGES:]
            latest_exchange = self.history[-2:]
        self._refresh_memory_summary(latest_exchange)

    def _refresh_memory_summary(self, latest_exchange: Optional[List[Dict[str, str]]] = None) -> None:
        if not latest_exchange:
            return
        try:
            with self.history_lock:
                existing_summary = compact_model_text(
                    self.memory_summary, max_chars=MEMORY_SUMMARY_MAX_CHARS
                )
            latest_text = format_history_for_memory(latest_exchange)
            if not latest_text:
                return
            user_content = latest_text
            if existing_summary:
                user_content = f"Existing memory:\n{existing_summary}\n\nLatest exchange:\n{latest_text}"
            completion = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                temperature=0.2,
                max_tokens=120,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Update the interview memory into compact bullet points. "
                            "Keep only facts that matter for future consistency: experience, projects, "
                            "skills, strengths, weaknesses, preferences, leadership examples, and open threads. "
                            "Avoid fluff and repetition."
                        ),
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
            )
            summary = compact_model_text(
                (completion.choices[0].message.content or "").strip(),
                max_chars=MEMORY_SUMMARY_MAX_CHARS,
            )
            if summary:
                with self.history_lock:
                    self.memory_summary = summary
        except Exception as exc:
            logger.debug("Memory summary refresh failed: %s", exc)

    def _text_to_speech(self, text: str) -> bytes:
        # Orpheus currently expects speed=1.0 in most deployments.
        speed = self.cfg.tts_speed
        speech_text = self._compact_tts_text(text, ANSWER_MAX_WORDS, ANSWER_MAX_CHARS)
        if not speech_text:
            raise RuntimeError("No text available for TTS after compaction.")

        if abs(speed - 1.0) > 1e-6:
            self._emit_ui(
                "log",
                "Note: Current Orpheus API may only support speed=1.0. Falling back automatically if needed.",
            )

        try:
            response = self.client.audio.speech.create(
                model="canopylabs/orpheus-v1-english",
                voice=self.cfg.tts_voice,
                input=speech_text,
                response_format="wav",
                speed=speed,
            )
        except Exception as exc:
            if is_tts_payload_too_large_error(exc):
                shorter_text = self._compact_tts_text(
                    speech_text, TTS_RETRY_MAX_WORDS, TTS_RETRY_MAX_CHARS
                )
                self._emit_ui("log", "TTS payload too large; retrying with shorter answer.")
                response = self.client.audio.speech.create(
                    model="canopylabs/orpheus-v1-english",
                    voice=self.cfg.tts_voice,
                    input=shorter_text,
                    response_format="wav",
                    speed=1.0,
                )
            # If speed is rejected by API, retry at 1.0 for reliability.
            elif abs(speed - 1.0) > 1e-6:
                self._emit_ui("log", f"TTS speed rejected by API ({exc}); retrying at 1.0.")
                response = self.client.audio.speech.create(
                    model="canopylabs/orpheus-v1-english",
                    voice=self.cfg.tts_voice,
                    input=speech_text,
                    response_format="wav",
                    speed=1.0,
                )
            else:
                raise

        if hasattr(response, "read"):
            audio_bytes = response.read()
        elif hasattr(response, "content"):
            audio_bytes = response.content
        else:
            raise RuntimeError("Unexpected TTS response type from Groq SDK.")

        if not audio_bytes:
            raise RuntimeError("Groq TTS returned empty audio.")

        return audio_bytes

    def _play_audio_bytes(self, audio_bytes: bytes, turn_id: int) -> bool:
        data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        playback_data = data if data.ndim == 1 else data[:, 0]
        if playback_data.size == 0:
            return False

        self._reset_interrupt_capture()
        self.is_speaking.set()

        interrupted = False
        stream, playback_data, sample_rate = self._open_output_stream(playback_data, int(sample_rate))
        try:
            stream.start()
            cursor = 0
            total = playback_data.shape[0]
            while cursor < total and not self.stop_event.is_set():
                if self.interrupt_requested.is_set() or not self._is_turn_current(turn_id):
                    interrupted = True
                    break
                end = min(cursor + OUTPUT_CHUNK_SAMPLES, total)
                chunk = playback_data[cursor:end]
                if chunk.shape[0] < OUTPUT_CHUNK_SAMPLES:
                    chunk = np.pad(chunk, (0, OUTPUT_CHUNK_SAMPLES - chunk.shape[0]))
                stream.write(chunk.reshape(-1, 1))
                cursor = end
        finally:
            try:
                stream.stop()
                stream.close()
            except Exception:
                logger.debug("Output stream close warning", exc_info=True)
            self.is_speaking.clear()
            if interrupted:
                self._promote_interrupt_buffer_to_pending()
            self._reset_interrupt_capture()

        return interrupted

    def _emit_api_error(self, context: str, exc: Exception) -> None:
        if is_rate_limit_error(exc):
            msg = f"{context}: Rate limit reached - wait a bit and retry."
        else:
            msg = f"{context}: {exc}"
        self._emit_ui("log", msg)
        self._emit_ui("log", traceback.format_exc().strip())
        self._emit_ui("status", "Listening...")
        if not self.preempt_requested.is_set():
            self._clear_audio_queue()
        logger.exception("%s", context)

    def _clear_audio_queue(self) -> None:
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        with self._pending_priority_lock:
            self.pending_priority_chunks = []


def run_app() -> None:
    # Basic sanity check before opening UI.
    try:
        sd.query_devices()
    except Exception as exc:
        print(f"Audio device error: {exc}")
        return

    initial_raw = load_config_raw(CONFIG_PATH)
    agent_ref: Dict[str, Optional[InterviewAgent]] = {"agent": None}

    def on_start() -> None:
        if agent_ref["agent"] is not None:
            return

        try:
            cfg = build_config(ui.get_config_raw(), require_required=True)
        except Exception as exc:
            ui.show_error("Invalid Setup", str(exc))
            return

        try:
            save_config(cfg, CONFIG_PATH)
        except Exception as exc:
            ui.show_error("Save Failed", f"Could not save config.json: {exc}")
            return

        ui.append_log("Saved setup to config.json.")
        ui.append_log("Starting interview agent...")
        ui.set_status("Starting...")

        agent: Optional[InterviewAgent] = None
        try:
            agent = InterviewAgent(cfg=cfg, ui=ui)
            agent_ref["agent"] = agent
            ui.set_running(True)
            agent.start()
        except Exception as exc:
            logger.exception("Failed to start agent")
            if agent is not None:
                try:
                    agent.stop()
                except Exception:
                    logger.debug("Agent cleanup failed after startup error.", exc_info=True)
            agent_ref["agent"] = None
            ui.set_running(False)
            ui.set_status("Ready")
            ui.show_error("Startup Error", str(exc))

    def on_stop() -> None:
        agent = agent_ref["agent"]
        if agent is None:
            return
        agent.stop()
        agent_ref["agent"] = None
        ui.set_running(False)
        ui.set_status("Stopped")
        ui.append_log("Agent stopped.")

    def on_exit() -> None:
        on_stop()
        ui.root.after(100, ui.root.destroy)

    ui = InterviewUI(
        on_start=on_start,
        on_stop=on_stop,
        on_exit=on_exit,
        initial_raw=initial_raw,
    )
    ui.append_log("Ready. Fill setup and click Start Interview.")
    ui.append_log("Input capture is system audio from your default playback device.")
    ui.append_log("No mic setup required.")

    def pump_ui_queue() -> None:
        agent = agent_ref["agent"]
        if agent is not None:
            while True:
                try:
                    kind, text = agent.ui_queue.get_nowait()
                except queue.Empty:
                    break
                if kind == "status":
                    ui.set_status(text)
                elif kind == "log":
                    ui.append_log(text)
                    print(text, flush=True)
            if agent.stop_event.is_set():
                agent_ref["agent"] = None
                ui.set_running(False)
                if ui.status_var.get() != "Stopped":
                    ui.set_status("Ready")
        ui.root.after(100, pump_ui_queue)

    ui.root.after(100, pump_ui_queue)
    ui.root.mainloop()


if __name__ == "__main__":
    run_app()

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

import sounddevice as sd
import soundfile as sf


ROOT = Path(__file__).parent
RECORDINGS = ROOT / "recordings"
MODELS = ROOT / "models"

# --- Models ---
WHISPER_MODEL = MODELS / "ggml-base.en.bin"

PIPER_VOICE_ONNX = MODELS / "piper_voice" / "voice.onnx"
PIPER_VOICE_JSON = MODELS / "piper_voice" / "voice.onnx.json"

# Downloaded GGUF (local file)
LLM_MODEL = MODELS / "llm" / "Llama-3.2-3B-Instruct-Q4_K_M.gguf"

SAMPLE_RATE = 16000
CHANNELS = 1

SYSTEM_PROMPT = (
    "You are a friendly teddy bear voice assistant. "
    "Keep responses concise (1-3 sentences). "
    "Be warm and conversational."
)

# ---------- helpers ----------

def require_file(path: Path, hint: str):
    if not path.exists():
        print(f"\nMissing: {path}\nHint: {hint}\n")
        sys.exit(1)

def record_wav(path: Path):
    """
    Push-to-talk MVP:
    - Press ENTER to start recording
    - Press ENTER again to stop
    """
    input("\nPress ENTER to START recording...")
    print("Recording... Press ENTER to STOP.")

    frames = []

    def callback(indata, frames_count, time_info, status):
        if status:
            print(status, file=sys.stderr)
        frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=callback,
    )

    with stream:
        input()  # stop on ENTER

    # Write WAV using soundfile
    import numpy as np
    data = np.concatenate(frames, axis=0)
    sf.write(str(path), data, SAMPLE_RATE)
    print(f"Saved recording to: {path}")

def stt_whisper_cpp(wav_path: Path) -> str:
    """
    Runs whisper.cpp CLI on a wav and returns the transcript (from the .txt file).
    """
    require_file(WHISPER_MODEL, "Download ggml-base.en.bin into ./models")

    out_base = wav_path.with_suffix("")  # whisper-cli uses -of as base path
    txt_path = Path(str(out_base) + ".txt")

    cmd = [
        "whisper-cli",
        "-m", str(WHISPER_MODEL),
        "-f", str(wav_path),
        "-otxt",
        "-of", str(out_base),
        "-l", "en",
    ]

    subprocess.run(cmd, check=True)

    if not txt_path.exists():
        raise RuntimeError(f"Expected transcript file not found: {txt_path}")

    return txt_path.read_text(encoding="utf-8").strip()

def _clean_llm_output(raw: str) -> str:
    """
    Make the model output usable for TTS:
    - remove special tokens
    - if the prompt was echoed, keep only after the last 'Assistant:'
    - stop if it starts inventing 'User:' / 'System:' / another 'Assistant:' turn
    """
    text = (raw or "").strip()

    # common special token
    text = text.replace("<|begin_of_text|>", "").strip()

    # If prompt got echoed, keep only completion after the last Assistant:
    if "Assistant:" in text:
        text = text.split("Assistant:")[-1].strip()

    # Hard-stop if it starts making up new turns
    for marker in ["\nUser:", "\nSystem:", "\nAssistant:"]:
        if marker in text:
            text = text.split(marker, 1)[0].strip()

    # Some models emit leading quotes/newlines
    return text.strip().strip('"').strip()

def llm_reply(user_text: str, history: list[tuple[str, str]]) -> str:
    """
    Uses llama-completion (one-shot) with a local GGUF model.
    No --stop flags (since your build doesn't accept them); we do post-processing instead.
    """
    require_file(LLM_MODEL, "Download the GGUF into ./models/llm/ (Llama-3.2-3B-Instruct-Q4_K_M.gguf)")

    convo = [f"System: {SYSTEM_PROMPT}"]
    for u, a in history[-6:]:
        convo.append(f"User: {u}")
        convo.append(f"Assistant: {a}")
    convo.append(f"User: {user_text}")
    convo.append("Assistant:")

    prompt = "\n".join(convo)


    # Use the CLI's `-sys` and `-p` flags and disable the interactive conversation
    # template so the tool returns only a single completion for our prompt.
    cmd = [
        "llama-completion",
        "-m", str(LLM_MODEL),
        "-n", "160",
        "--temp", "0.7",
        "--top-p", "0.9",
        "-sys", SYSTEM_PROMPT,
        "-p", prompt,
        "--no-conversation",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Fallback: some builds expect the prompt via stdin or file. Retry by sending the
    # prompt on stdin (no -p) if the CLI rejected our flags.
    if result.returncode != 0:
        stderr = (result.stderr or "").lower()
        if "invalid argument" in stderr or "unrecognized" in stderr or "--no-conversation" in stderr:
            result = subprocess.run(
                [
                    "llama-completion",
                    "-m", str(LLM_MODEL),
                    "-n", "160",
                    "--temp", "0.7",
                    "--top-p", "0.9",
                ],
                input=prompt,
                capture_output=True,
                text=True,
            )

    if result.returncode != 0:
        raise RuntimeError(result.stderr or result.stdout or "llama-completion failed")

    reply = _clean_llm_output(result.stdout)

    # Ensure we return a short reply (1-3 sentences). Truncate noisy multi-turn
    # output to the first 1-2 sentences to avoid the CLI inventing extra turns.
    def _truncate_sentences(text: str, max_sentences: int = 2) -> str:
        if not text:
            return text
        # Prefer the first non-empty line
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        first = lines[0] if lines else text
        # Split on sentence enders
        import re
        parts = re.split(r'(?<=[\.\?!])\s+', first)
        return " ".join(parts[:max_sentences]).strip()

    short_reply = _truncate_sentences(reply, max_sentences=2)

    return short_reply or "Hmm—can you say that again?"

def tts_piper(text: str):
    require_file(PIPER_VOICE_ONNX, "Download a Piper voice .onnx to ./models/piper_voice/voice.onnx")
    require_file(PIPER_VOICE_JSON, "Download the matching .onnx.json to ./models/piper_voice/voice.onnx.json")

    piper_bin = shutil.which("piper")
    if not piper_bin:
        raise RuntimeError("Could not find 'piper' on PATH. Try: pip install piper-tts")

    out_wav = RECORDINGS / f"tts_{int(time.time())}.wav"

    cmd = [
        piper_bin,
        "--model", str(PIPER_VOICE_ONNX),
        "--output_file", str(out_wav),
    ]

    subprocess.run(cmd, input=text, text=True, check=True)

    if not out_wav.exists() or out_wav.stat().st_size < 1000:
        raise RuntimeError(f"TTS output wav missing or too small: {out_wav}")

    subprocess.run(["afplay", str(out_wav)], check=True)

# ---------- main ----------

def main():
    RECORDINGS.mkdir(parents=True, exist_ok=True)

    history: list[tuple[str, str]] = []

    print("\n🧸 Teddy MVP 1 (local): Push-to-talk → STT → LLM → TTS\n")
    print("Ctrl+C to exit.\n")

    try:
        while True:
            wav_path = RECORDINGS / f"utt_{int(time.time())}.wav"
            record_wav(wav_path)

            print("Transcribing...")
            user_text = stt_whisper_cpp(wav_path)
            print(f"\nYou said: {user_text}")

            if not user_text.strip():
                print("Heard nothing—try again.")
                continue

            print("Thinking...")
            reply = llm_reply(user_text, history)
            print(f"\nTeddy: {reply}\n")

            print("Speaking...")
            tts_piper(reply)

            history.append((user_text, reply))

    except KeyboardInterrupt:
        print("\nBye 👋")

if __name__ == "__main__":
    main()

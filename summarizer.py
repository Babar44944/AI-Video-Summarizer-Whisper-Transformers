"""
AI Video Summarizer - SINGLE FILE VERSION
Sab kuch ek hi file mein: auto-install, auto-launch, Whisper transcription,
BART summarization, topic extraction, timestamps, TXT/PDF export.

USAGE (PowerShell):
    python video_summarizer.py

Bas itna chalao - khud packages install karega, khud streamlit launch karega.
"""

import importlib
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime


def _patch_windows_asyncio():
    """Stop harmless ConnectionResetError tracebacks when browser closes a socket."""
    if sys.platform != "win32":
        return
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport

        _original = _ProactorBasePipeTransport._call_connection_lost

        def _quiet_connection_lost(self, exc):
            try:
                _original(self, exc)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass

        _ProactorBasePipeTransport._call_connection_lost = _quiet_connection_lost
    except (ImportError, AttributeError):
        pass

# =================================================================
# STEP 0: Auto-install missing packages + auto-relaunch under Streamlit
# (Ye block tab chalta hai jab is file ko seedha `python video_summarizer.py` se run karein)
# =================================================================

REQUIRED_PACKAGES = {
    "whisper": "openai-whisper",
    "transformers": "transformers",
    "streamlit": "streamlit",
    "moviepy": "moviepy",
    "torch": "torch",
    "fpdf": "fpdf2",
}


def _ensure_packages():
    missing = []
    for module_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"📦 Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    else:
        print("✅ All Python packages already installed.")


def _get_ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def _ensure_ffmpeg():
    """Return True if ffmpeg is on PATH or bundled via imageio-ffmpeg (moviepy)."""
    return _get_ffmpeg_exe() is not None


def _check_ffmpeg():
    exe = _get_ffmpeg_exe()
    if exe and shutil.which("ffmpeg"):
        print("✅ ffmpeg found.")
    elif exe:
        print("✅ ffmpeg bundled via moviepy (will use for efficient audio decoding).")
    else:
        print("⚠️  ffmpeg not found — using built-in WAV loader for transcription.")
        print("   Optional: choco install ffmpeg   (ya ffmpeg.org se download karke PATH me add karo)")


def _ensure_dirs():
    for d in ["videos", "audio", "transcripts", "summaries"]:
        os.makedirs(d, exist_ok=True)


# Agar Streamlit ke andar nahi chal raha (yaani user ne `python video_summarizer.py` chalaya),
# to packages install karo aur khud ko `streamlit run` se relaunch karo.
if __name__ == "__main__" and os.environ.get("_RUNNING_UNDER_STREAMLIT") != "1":
    _patch_windows_asyncio()
    _ensure_packages()
    _check_ffmpeg()
    _ensure_dirs()
    print("🚀 Launching AI Video Summarizer in your browser...\n")
    env = os.environ.copy()
    env["PATH"] = os.environ.get("PATH", "")
    env["_RUNNING_UNDER_STREAMLIT"] = "1"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)],
        env=env,
    )
    sys.exit(0)


# =================================================================
# STEP 1+: Actual app (yahan se Streamlit ke andar code chalta hai)
# =================================================================

_patch_windows_asyncio()

import streamlit as st

try:
    from moviepy import VideoFileClip  # moviepy >= 2.0
except ImportError:
    from moviepy.editor import VideoFileClip  # moviepy < 2.0

import numpy as np
import torch
import whisper

_ensure_ffmpeg()

SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"  # smaller + faster than bart-large-cnn


def _get_transformers_classes():
    """Transformers 5.x removed some top-level Auto* exports — load from submodules."""
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError:
        from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM
        from transformers.models.auto.tokenization_auto import AutoTokenizer
    return AutoModelForSeq2SeqLM, AutoTokenizer


@st.cache_resource
def get_whisper_model(size="tiny"):
    return whisper.load_model(size)


@st.cache_resource
def get_summarizer():
    AutoModelForSeq2SeqLM, AutoTokenizer = _get_transformers_classes()
    tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL)
    model.eval()
    return tokenizer, model


# ---------- Audio extraction ----------
WHISPER_SAMPLE_RATE = 16000


def extract_audio(video_path, audio_path="audio/audio.wav"):
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    ffmpeg = _get_ffmpeg_exe()
    if ffmpeg:
        try:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-nostdin",
                    "-i",
                    video_path,
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    str(WHISPER_SAMPLE_RATE),
                    "-ac",
                    "1",
                    audio_path,
                ],
                check=True,
                capture_output=True,
            )
            return audio_path
        except subprocess.CalledProcessError:
            pass

    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(
        audio_path, fps=WHISPER_SAMPLE_RATE, nbytes=2, ffmpeg_params=["-ac", "1"], logger=None
    )
    clip.close()
    return audio_path


def _load_audio_for_whisper(file, sr=WHISPER_SAMPLE_RATE):
    """Decode audio to mono float32 via ffmpeg (streams + resamples, low memory)."""
    ffmpeg = _get_ffmpeg_exe()
    if not ffmpeg:
        return _load_wav_for_whisper(file, target_sr=sr)

    cmd = [
        ffmpeg,
        "-nostdin",
        "-threads",
        "0",
        "-i",
        file,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def _load_wav_for_whisper(path, target_sr=WHISPER_SAMPLE_RATE, chunk_seconds=30):
    """Chunked WAV reader — fallback when ffmpeg is unavailable."""
    import wave

    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        chunk_frames = int(chunk_seconds * framerate)

        if sample_width != 2:
            raise ValueError(
                f"Chunked WAV fallback only supports 16-bit PCM (got {sample_width} bytes/sample)"
            )

        resampled_parts = []
        while True:
            raw = wf.readframes(chunk_frames)
            if not raw:
                break

            samples = np.frombuffer(raw, dtype=np.int16)
            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)

            chunk = samples.astype(np.float32) / 32768.0
            if framerate != target_sr:
                target_len = int(len(chunk) * target_sr / framerate)
                if target_len > 0:
                    indices = np.linspace(0, len(chunk) - 1, target_len)
                    chunk = np.interp(indices, np.arange(len(chunk)), chunk)

            resampled_parts.append(chunk.astype(np.float32))

    if not resampled_parts:
        return np.array([], dtype=np.float32)
    return np.concatenate(resampled_parts)


def transcribe_audio(audio_path, model_size="tiny"):
    model = get_whisper_model(model_size)
    result = model.transcribe(
        _load_audio_for_whisper(audio_path),
        fp16=False,
        beam_size=1,
        best_of=1,
        condition_on_previous_text=False,
    )
    return result["text"], result["segments"]


# ---------- Summarization ----------
def _summarize_text(text, max_length=150, min_length=50):
    tokenizer, model = get_summarizer()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.inference_mode():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_length=max_length,
            min_length=min_length,
            num_beams=2,
            do_sample=False,
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def _chunk_text(text, max_words=800):
    words = text.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def _summarize_long_text(text, max_length=150, min_length=50):
    chunks = _chunk_text(text)
    if len(chunks) == 1:
        return _summarize_text(chunks[0], max_length=max_length, min_length=min_length)

    partial_summaries = []
    for chunk in chunks:
        partial_summaries.append(_summarize_text(chunk, max_length=120, min_length=30))

    combined = " ".join(partial_summaries)
    return _summarize_text(combined, max_length=max_length, min_length=min_length)


def generate_all_summaries(transcript):
    """One summarization pass, then format as short / detailed / executive."""
    core = _summarize_long_text(transcript, max_length=300, min_length=80)
    sentences = [s.strip() for s in core.replace("\n", " ").split(". ") if s.strip()]
    short = "\n".join(f"• {s.rstrip('.')}." for s in sentences[:5])
    detailed = core
    executive = (
        "Executive Summary\n------------------\n"
        f"{core}\n\nKey Action Items:\n"
        "(Review transcript for specific tasks, deadlines, and owners.)"
    )
    return short, detailed, executive


def generate_summary(transcript, style="short"):
    short, detailed, executive = generate_all_summaries(transcript)
    if style == "short":
        return short
    if style == "detailed":
        return detailed
    if style == "executive":
        return executive
    raise ValueError("style must be 'short', 'detailed', or 'executive'")


# ---------- Topic extraction ----------
def extract_topics(transcript, top_n=5):
    from collections import Counter
    import re

    stopwords = set("""a an the is are was were be been being to of in on for and or but
        with as at by from this that these those it its it's i you we they he she
        them his her our your their not no do does did so if then than there here
        will would can could should about into over under up down out just like
        also very really one two going gonna got get going s t re ve d ll""".split())

    words = re.findall(r"[a-zA-Z']+", transcript.lower())
    filtered = [w for w in words if w not in stopwords and len(w) > 3]
    common = Counter(filtered).most_common(top_n)
    return [w.capitalize() for w, _ in common]


# ---------- Timestamps ----------
def _format_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def generate_timestamps(segments, interval_seconds=300):
    timestamps = []
    last_marker = -interval_seconds
    for seg in segments:
        if seg["start"] - last_marker >= interval_seconds:
            text_preview = seg["text"].strip()[:60]
            timestamps.append(f"{_format_time(seg['start'])} - {text_preview}")
            last_marker = seg["start"]
    return timestamps


# ---------- Export ----------
def _safe_stem(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in stem) or "video"


def save_outputs(video_name, transcript, short, detailed, executive, topics, timestamps):
    """Save transcript + summaries to project folders."""
    base = _safe_stem(video_name)
    os.makedirs("transcripts", exist_ok=True)
    os.makedirs("summaries", exist_ok=True)

    paths = {
        "transcript": os.path.join("transcripts", f"{base}_transcript.txt"),
        "summary": os.path.join("summaries", f"{base}_summary.txt"),
        "short": os.path.join("summaries", f"{base}_short_summary.txt"),
        "executive": os.path.join("summaries", f"{base}_executive_summary.txt"),
        "report": os.path.join("summaries", f"{base}_full_report.txt"),
    }

    with open(paths["transcript"], "w", encoding="utf-8") as f:
        f.write(transcript)
    with open(paths["summary"], "w", encoding="utf-8") as f:
        f.write(detailed)
    with open(paths["short"], "w", encoding="utf-8") as f:
        f.write(short)
    with open(paths["executive"], "w", encoding="utf-8") as f:
        f.write(executive)

    report = (
        f"AI Video Summary Report\n"
        f"Video: {video_name}\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"Topics: {', '.join(topics)}\n\n"
        f"--- Short Summary ---\n{short}\n\n"
        f"--- Detailed Summary ---\n{detailed}\n\n"
        f"--- Executive Summary ---\n{executive}\n\n"
        f"--- Timestamps ---\n" + "\n".join(timestamps)
    )
    with open(paths["report"], "w", encoding="utf-8") as f:
        f.write(report)

    return paths


def build_outputs_zip(video_name, transcript, short, detailed, executive, topics, timestamps):
    base = _safe_stem(video_name)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{base}_transcript.txt", transcript)
        zf.writestr(f"{base}_summary.txt", detailed)
        zf.writestr(f"{base}_short_summary.txt", short)
        zf.writestr(f"{base}_executive_summary.txt", executive)
        zf.writestr(f"{base}_topics.txt", "\n".join(f"- {t}" for t in topics))
        zf.writestr(f"{base}_timestamps.txt", "\n".join(timestamps))
    return buf.getvalue()


def _auto_download_bytes(data: bytes, filename: str, mime="application/octet-stream"):
    """Trigger a browser download without an extra click (best-effort)."""
    import base64

    b64 = base64.b64encode(data).decode("ascii")
    st.components.v1.html(
        f"""<script>
        (function() {{
            const link = document.createElement("a");
            link.href = "data:{mime};base64,{b64}";
            link.download = {json.dumps(filename)};
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }})();
        </script>""",
        height=0,
    )


def export_pdf(content, filepath):
    from fpdf import FPDF
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in content.split("\n"):
        pdf.multi_cell(0, 8, line.encode("latin-1", "replace").decode("latin-1"))
    pdf.output(filepath)
    return filepath


# =================================================================
# STEP 2: Streamlit UI
# =================================================================

st.set_page_config(page_title="AI Video Summarizer", page_icon="🎥", layout="wide")
st.title("🎥 AI Video Summarizer")
st.caption("Upload a video → get transcript, summary, topics & timestamps")

with st.sidebar:
    st.header("⚙️ Settings")
    model_size = st.selectbox("Whisper model size", ["tiny", "base", "small", "medium"], index=0)
    st.caption("tiny = fastest · base/small/medium = slower but more accurate")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "mkv", "avi"])

if uploaded_video:
    os.makedirs("videos", exist_ok=True)
    video_path = os.path.join("videos", uploaded_video.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    st.video(video_path)

    if st.button("🚀 Summarize Video", type="primary"):
        with st.spinner("Extracting audio..."):
            audio_path = extract_audio(video_path)

        with st.spinner(f"Transcribing with Whisper ({model_size})..."):
            transcript, segments = transcribe_audio(audio_path, model_size)
        st.session_state["transcript"] = transcript
        st.session_state["segments"] = segments

        with st.spinner("Generating summaries..."):
            short, detailed, executive = generate_all_summaries(transcript)
            st.session_state["short"] = short
            st.session_state["detailed"] = detailed
            st.session_state["executive"] = executive

        with st.spinner("Extracting topics & timestamps..."):
            st.session_state["topics"] = extract_topics(transcript)
            st.session_state["timestamps"] = generate_timestamps(segments)

        saved = save_outputs(
            uploaded_video.name,
            transcript,
            st.session_state["short"],
            st.session_state["detailed"],
            st.session_state["executive"],
            st.session_state["topics"],
            st.session_state["timestamps"],
        )
        st.session_state["saved_paths"] = saved
        st.session_state["output_zip"] = build_outputs_zip(
            uploaded_video.name,
            transcript,
            st.session_state["short"],
            st.session_state["detailed"],
            st.session_state["executive"],
            st.session_state["topics"],
            st.session_state["timestamps"],
        )
        st.session_state["output_zip_name"] = f"{_safe_stem(uploaded_video.name)}_outputs.zip"
        st.session_state["auto_download"] = True

        st.success("Done! Files saved and download started.")

if "transcript" in st.session_state:
    if st.session_state.pop("auto_download", False):
        zip_name = st.session_state.get("output_zip_name", "video_outputs.zip")
        _auto_download_bytes(
            st.session_state["output_zip"],
            zip_name,
            mime="application/zip",
        )

    if "saved_paths" in st.session_state:
        p = st.session_state["saved_paths"]
        st.info(
            "Saved locally:\n"
            f"- `{p['transcript']}`\n"
            f"- `{p['summary']}`\n"
            f"- `{p['report']}`"
        )

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "📄 Transcript", "🎯 Topics", "⏱️ Timestamps"])

    with tab1:
        style = st.radio("Summary type", ["Short (bullets)", "Detailed", "Executive"], horizontal=True)
        if style == "Short (bullets)":
            st.markdown(st.session_state["short"])
            content = st.session_state["short"]
        elif style == "Detailed":
            st.write(st.session_state["detailed"])
            content = st.session_state["detailed"]
        else:
            st.text(st.session_state["executive"])
            content = st.session_state["executive"]

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Download Summary (TXT)",
                content,
                file_name="summary.txt",
                key="dl_summary",
            )
            if "output_zip" in st.session_state:
                st.download_button(
                    "⬇️ Download All (ZIP)",
                    st.session_state["output_zip"],
                    file_name=st.session_state.get("output_zip_name", "outputs.zip"),
                    mime="application/zip",
                    key="dl_zip",
                )
        with col2:
            if st.button("⬇️ Generate PDF Report"):
                pdf_path = export_pdf(
                    f"AI Video Summary Report\n\n"
                    f"Topics: {', '.join(st.session_state['topics'])}\n\n"
                    f"Summary:\n{content}\n\n"
                    f"Timestamps:\n" + "\n".join(st.session_state["timestamps"]),
                    "summaries/report.pdf",
                )
                with open(pdf_path, "rb") as f:
                    st.download_button("⬇️ Download PDF", f, file_name="summary_report.pdf")

    with tab2:
        st.text_area("Full Transcript", st.session_state["transcript"], height=400)
        st.download_button(
            "⬇️ Download Transcript", st.session_state["transcript"], file_name="transcript.txt"
        )

    with tab3:
        for topic in st.session_state["topics"]:
            st.markdown(f"- {topic}")

    with tab4:
        for ts in st.session_state["timestamps"]:
            st.markdown(f"`{ts}`")
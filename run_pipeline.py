"""
run_pipeline.py
──────────────────────────────────────────────────────────────────────────────
ONE command to rule them all.

Give it a topic -> get a fully rendered MP4 chalkboard video with
Hindi Trump-Modi narration.

Usage
-----
    python run_pipeline.py "Gradient Descent"
    python run_pipeline.py "Softmax Function" --voice-trump voices/trump.wav --voice-modi voices/modi.wav
    python run_pipeline.py "Attention Mechanism" --no-diagram --keep-workdir

What it does internally
-----------------------
    Step 1  generate_chalkboard.py Pass 1  -- Gemini generates title + diagram hint (JSON)
    Step 2  generate_chalkboard.py Pass 2  -- Gemini generates animated SVG diagram
    Step 3  Jinja2 renders HTML board      -- SVG + title injected into template
    Step 4  narrate_and_render.py Step 1   -- Gemini writes Hindi Trump-Modi dialogue
    Step 5  narrate_and_render.py Step 2   -- HuggingFace TTS generates audio per line
    Step 6  narrate_and_render.py Step 3   -- Playwright screenshots board per line
    Step 7  narrate_and_render.py Step 4   -- FFmpeg stitches frames + audio -> MP4

File layout expected
--------------------
    run_pipeline.py                  <- THIS FILE
    template.html                    <- Jinja2 HTML template
    generate_chalkboard.py           <- chalkboard generator (imported)
    narrate_and_render.py            <- narration + video renderer (imported)
    voices/
        trump.wav                    <- Trump voice sample (default path)
        modi.wav                     <- Modi voice sample (default path)
    avatar/
        trump.png                    <- Trump avatar image (default path)
        modi.png                     <- Modi avatar image (default path)
    output/                          <- all outputs land here (auto-created)

Requirements
------------
    pip install google-genai jinja2 python-dotenv gradio_client playwright
    playwright install chromium
    # ffmpeg must be on PATH

.env
----
    GEMINI_API_KEY=your-gemini-key
    HF_TOKEN=your-huggingface-token   (optional but helps with rate limits)
"""

import os
import sys
import json
import base64
import argparse
import threading
import socketserver
import http.server
import subprocess
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

# Ensure UTF-8 output on Windows to avoid cp1252 UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Make sure siblings are importable regardless of cwd
HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from generate_chalkboard import (
    generate_content,
    generate_diagram,
    build_context,
    render_template,
    save_html,
    save_json,
    slug,
    TEMPLATE_FILE,
)
from narrate_and_render import (
    generate_narration,
    generate_audio,
    render_html_frames,
    build_video,
    _get_audio_duration,
    _silent_wav,
    HF_SPACE,
    BOARD_WIDTH,
    BOARD_HEIGHT,
)


# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR           = Path(os.environ.get("PIPELINE_OUTPUT_DIR", str(Path(tempfile.gettempdir()) / "bytebrain-output")))
DEFAULT_TRUMP        = HERE / "voices" / "trump.wav"
DEFAULT_MODI         = HERE / "voices" / "modi.wav"
DEFAULT_TRUMP_AVATAR = HERE / "avatar" / "trump.png"
DEFAULT_MODI_AVATAR  = HERE / "avatar" / "modi.png"
HTTP_PORT            = 8765


# ── Local HTTP server (fixes Google Fonts over file://) ──────────────────────

class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def _start_http_server(directory: Path) -> socketserver.TCPServer:
    os.chdir(directory)
    httpd = socketserver.TCPServer(("", HTTP_PORT), _QuietHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    print(f"[http]   Local font server started on http://localhost:{HTTP_PORT}")
    return httpd


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Full chalkboard video pipeline -- just give a topic."
    )
    parser.add_argument(
        "topic",
        help="ML/CS topic to explain, e.g. 'Gradient Descent'",
    )
    parser.add_argument(
        "--voice-trump", default=str(DEFAULT_TRUMP),
        help=f"WAV voice sample for Trump (default: {DEFAULT_TRUMP})",
    )
    parser.add_argument(
        "--voice-modi", default=str(DEFAULT_MODI),
        help=f"WAV voice sample for Modi  (default: {DEFAULT_MODI})",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Final MP4 output path",
    )
    parser.add_argument(
        "--no-diagram", action="store_true",
        help="Skip SVG diagram generation (faster, uses placeholder)",
    )
    parser.add_argument(
        "--save-json", action="store_true",
        help="Save the intermediate content JSON",
    )
    parser.add_argument(
        "--save-script", action="store_true",
        help="Save the Hindi dialogue script as JSON",
    )
    parser.add_argument(
        "--keep-workdir", action="store_true",
        help="Keep intermediate frames/audio files for debugging",
    )
    parser.add_argument(
        "--hf-space", default=os.environ.get("HF_SPACE", HF_SPACE),
        help="HuggingFace Space ID for TTS",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
        help="HuggingFace API token (optional)",
    )
    parser.add_argument(
        "--avatar-trump", default=str(DEFAULT_TRUMP_AVATAR),
        help=f"Trump avatar image path (default: {DEFAULT_TRUMP_AVATAR})",
    )
    parser.add_argument(
        "--avatar-modi", default=str(DEFAULT_MODI_AVATAR),
        help=f"Modi avatar image path (default: {DEFAULT_MODI_AVATAR})",
    )
    parser.add_argument(
        "--openai-transcribe-model",
        default=os.environ.get("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe"),
        help="OpenAI transcription model for timing extraction",
    )
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _image_to_data_uri(image_path: Path) -> str:
    ext  = image_path.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _silent_wav_stereo(path: Path, duration: int = 2):
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "anullsrc=r=44100:cl=stereo",
        "-ar", "44100",
        "-t", str(duration),
        str(path),
    ], capture_output=True)


def _concat_audio_tracks(audio_paths: list[Path], output_audio_path: Path) -> Path:
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    concat_list = output_audio_path.parent / "audio_concat.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for ap in audio_paths:
            f.write(f"file '{ap.resolve()}'\n")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(output_audio_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-ar", "24000", "-ac", "1",
            str(output_audio_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    return output_audio_path


def _mux_recorded_video_with_audio(recorded_video_path: Path, audio_path: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(recorded_video_path),
        "-i", str(audio_path),
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def _build_speaker_timeline_with_openai(
    dialogue: list[dict],
    audio_paths: list[Path],
    durations: list[float | None],
    transcribe_model: str,
) -> list[dict]:
    try:
        openai_module = __import__("openai")
    except Exception:
        raise SystemExit("[error] openai package is required. Run: pip install openai")
    OpenAI  = openai_module.OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("[error] OPENAI_API_KEY is required for transcription-based avatar timing.")
    client = OpenAI(api_key=api_key)

    timeline = []
    start_at = 0.0
    for idx, entry in enumerate(dialogue):
        dur        = durations[idx] if idx < len(durations) and durations[idx] else 3.0
        audio_path = audio_paths[idx] if idx < len(audio_paths) else None
        transcript = ""
        if audio_path and audio_path.exists():
            try:
                with audio_path.open("rb") as f:
                    result = client.audio.transcriptions.create(
                        model=transcribe_model,
                        file=f,
                        response_format="verbose_json",
                        timestamp_granularities=["word"],
                    )
                transcript = getattr(result, "text", "") or ""
                words = getattr(result, "words", None) or []
                if words:
                    first_start = words[0].get("start", 0.0) if isinstance(words[0], dict) else getattr(words[0], "start", 0.0)
                    last_end    = words[-1].get("end",   0.0) if isinstance(words[-1], dict) else getattr(words[-1], "end",   0.0)
                    dur = max(float(last_end) - float(first_start), 0.1)
            except Exception as e:
                print(f"[warn]  OpenAI transcription failed for line {idx+1}: {e}")
        end_at = start_at + float(dur)
        timeline.append({
            "index":        idx + 1,
            "speaker":      "trump" if entry.get("speaker") == 1 else "modi",
            "start_sec":    round(start_at, 3),
            "end_sec":      round(end_at, 3),
            "duration_sec": round(float(dur), 3),
            "text":         entry.get("line", ""),
            "transcript":   transcript,
        })
        start_at = end_at
    return timeline


# ── Playwright recorder (accepts http:// URL directly) ────────────────────────

def _record_animation_via_http(
    html_url: str,
    timeline: list[dict],
    work_dir: Path,
    avatar_trump_path: Path,
    avatar_modi_path: Path,
) -> Path:
    from narrate_and_render import _require

    pw_module       = _require("playwright.sync_api", "playwright")
    sync_playwright = pw_module.sync_playwright

    recording_dir = work_dir / "recordings"
    recording_dir.mkdir(parents=True, exist_ok=True)

    avatar_trump_data = _image_to_data_uri(avatar_trump_path)
    avatar_modi_data  = _image_to_data_uri(avatar_modi_path)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(
            viewport={"width": BOARD_WIDTH, "height": BOARD_HEIGHT},
            record_video_dir=str(recording_dir),
            record_video_size={"width": BOARD_WIDTH, "height": BOARD_HEIGHT},
        )
        page = context.new_page()
        page.goto(html_url, wait_until="networkidle")
        page.wait_for_timeout(8500)

        page.add_style_tag(content="""
            .speaker-avatar {
                position: fixed;
                bottom: 1px;
                height: 100%;
                max-width: 68%;
                object-fit: contain;
                object-position: bottom center;
                z-index: 9998;
                filter: drop-shadow(0 6px 16px rgba(0,0,0,.45));
            }
            #avatar-trump { left: 20px; }
            #avatar-modi  { right: 20px; }
            .avatar-active { display: block;  opacity: 1; }
            .avatar-hidden { display: none;   opacity: 0; }
        """)

        page.evaluate(f"""
            const trump = document.createElement('img');
            trump.id = 'avatar-trump';
            trump.className = 'speaker-avatar avatar-hidden';
            trump.src = {json.dumps(avatar_trump_data)};
            document.body.appendChild(trump);

            const modi = document.createElement('img');
            modi.id = 'avatar-modi';
            modi.className = 'speaker-avatar avatar-hidden';
            modi.src = {json.dumps(avatar_modi_data)};
            document.body.appendChild(modi);
        """)

        for item in timeline:
            speaker     = item.get("speaker", "trump")
            duration_ms = int(max(0.1, float(item.get("duration_sec", 0.1))) * 1000)

            page.evaluate(f"""
                const trump  = document.getElementById('avatar-trump');
                const modi   = document.getElementById('avatar-modi');
                const active = {json.dumps(speaker)};
                if (active === 'trump') {{
                    trump.classList.add('avatar-active');
                    trump.classList.remove('avatar-hidden');
                    modi.classList.add('avatar-hidden');
                    modi.classList.remove('avatar-active');
                }} else {{
                    modi.classList.add('avatar-active');
                    modi.classList.remove('avatar-hidden');
                    trump.classList.add('avatar-hidden');
                    trump.classList.remove('avatar-active');
                }}
            """)
            page.wait_for_timeout(duration_ms)

        # Outro: hide avatars
        page.evaluate("""
            ['avatar-trump', 'avatar-modi'].forEach(id => {
                const el = document.getElementById(id);
                if (el) {
                    el.classList.add('avatar-hidden');
                    el.classList.remove('avatar-active');
                }
            });
        """)
        page.wait_for_timeout(600)

        video_path_str = page.video.path()
        page.close()
        context.close()
        browser.close()

    raw_video_path = Path(video_path_str)
    if not raw_video_path.exists():
        raise SystemExit("[error] Playwright recording failed: no video file produced.")
    return raw_video_path


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    args       = parse_args()
    topic      = args.topic
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_slug = slug(topic)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    work_dir = OUTPUT_DIR / f"_work_{topic_slug}_{ts}"
    work_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output) if args.output \
                  else OUTPUT_DIR / f"{topic_slug}_{ts}_video.mp4"

    print(f"\n{'='*60}")
    print(f"  TOPIC : {topic}")
    print(f"  OUTPUT: {output_path}")
    print(f"{'='*60}\n")

    # Validate voice files early
    for label, vpath in [("--voice-trump", args.voice_trump), ("--voice-modi", args.voice_modi)]:
        if not Path(vpath).exists():
            raise SystemExit(
                f"[error] Voice file not found ({label}): {Path(vpath).resolve()}\n"
                f"        Place WAV samples in voices/trump.wav and voices/modi.wav\n"
                f"        or pass --voice-trump / --voice-modi explicitly."
            )

    for label, ipath in [("--avatar-trump", args.avatar_trump), ("--avatar-modi", args.avatar_modi)]:
        if not Path(ipath).exists():
            raise SystemExit(
                f"[error] Avatar image not found ({label}): {Path(ipath).resolve()}\n"
                f"        Place images in avatar/trump.png and avatar/modi.png\n"
                f"        or pass --avatar-trump / --avatar-modi explicitly."
            )

    # =========================================================================
    # PHASE 1 -- Generate the chalkboard HTML
    # =========================================================================
    print("-- PHASE 1: Chalkboard HTML ----------------------------------\n")

    raw_data = generate_content(topic)

    if args.save_json:
        save_json(raw_data, OUTPUT_DIR / f"{topic_slug}_{ts}.json")

    diagram_svg = None
    if not args.no_diagram:
        hint = raw_data.get("diagram_hint", f"A visual diagram explaining {topic}")
        diagram_svg = generate_diagram(topic, hint)

    ctx  = build_context(raw_data, diagram_svg)
    html = render_template(ctx, template_file=TEMPLATE_FILE)

    html_path = OUTPUT_DIR / f"{topic_slug}_{ts}.html"
    save_html(html, html_path)

    print(f"\n[phase1] HTML ready -> {html_path.name}\n")

    # =========================================================================
    # PHASE 2 -- Narration + audio
    # =========================================================================
    print("-- PHASE 2: Narration & TTS Audio ----------------------------\n")

    dialogue = generate_narration(topic)

    if args.save_script:
        script_path = OUTPUT_DIR / f"{topic_slug}_{ts}_script.json"
        script_path.write_text(
            json.dumps(dialogue, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[narr]  Script saved -> {script_path.name}")

    print("\n-- Dialogue Script ------------------------------------------")
    for entry in dialogue:
        name = "Trump" if entry["speaker"] == 1 else "Modi "
        print(f"  [{name}]: {entry['line']}")
    print("-------------------------------------------------------------\n")

    audio_dir   = work_dir / "audio"
    audio_paths = generate_audio(
        dialogue,
        voice_trump=args.voice_trump,
        voice_modi=args.voice_modi,
        audio_dir=audio_dir,
        hf_space=args.hf_space,
        hf_token=args.hf_token,
    )

    durations = [_get_audio_duration(ap) for ap in audio_paths]
    timeline  = _build_speaker_timeline_with_openai(
        dialogue,
        audio_paths,
        durations,
        transcribe_model=args.openai_transcribe_model,
    )
    (work_dir / "speaker_timeline.json").write_text(
        json.dumps(timeline, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[phase2] {len(audio_paths)} audio clips ready\n")

    # =========================================================================
    # PHASE 3 -- Record animation via Playwright
    # =========================================================================
    print("-- PHASE 3: Recording Animation ------------------------------\n")

    httpd    = _start_http_server(OUTPUT_DIR)
    html_url = f"http://localhost:{HTTP_PORT}/{html_path.name}"

    recorded_video_path = _record_animation_via_http(
        html_url,
        timeline,
        work_dir,
        avatar_trump_path=Path(args.avatar_trump),
        avatar_modi_path=Path(args.avatar_modi),
    )

    httpd.shutdown()
    print(f"\n[phase3] Animation recorded -> {recorded_video_path.name}\n")

    # =========================================================================
    # PHASE 4 -- Build final MP4
    # =========================================================================
    print("-- PHASE 4: Building Video -----------------------------------\n")

    merged_audio_path = work_dir / "merged_audio.wav"
    _concat_audio_tracks(audio_paths, merged_audio_path)
    _mux_recorded_video_with_audio(recorded_video_path, merged_audio_path, output_path)

    if not args.keep_workdir:
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        print(f"[debug]  Work files kept at: {work_dir.resolve()}")

    print(f"\n{'='*60}")
    print(f"  DONE!")
    print(f"  Video -> {output_path.resolve()}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
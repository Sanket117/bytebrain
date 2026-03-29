"""
narrate_and_render.py
──────────────────────────────────────────────────────────────────────────────
Full pipeline:
  1. generate_narration()  -- Gemini writes a funny Trump vs Modi
                              Hindi dialogue for the given topic
  2. generate_audio()      -- sends each dialogue line to the HuggingFace
                              Gradio TTS (banao-tech/vibe-voice-custom-voices)
                              using speaker voice files you provide
  3. render_html_frames()  -- Playwright opens the generated HTML board,
                              takes a screenshot per dialogue beat
  4. build_video()         -- FFmpeg stitches frames + per-line audio into
                              the final MP4

Usage
-----
    python narrate_and_render.py                        \\
        --html   output/softmax_20260329.html           \\
        --topic  "Softmax Function"                     \\
        --voice-trump  voices/trump.wav                 \\
        --voice-modi   voices/modi.wav                  \\
        --output final/softmax_video.mp4

Requirements
------------
    pip install google-genai gradio_client playwright python-dotenv
    playwright install chromium

.env
----
    GEMINI_API_KEY=your-key-here
"""

import os
import re
import json
import shutil
import argparse
import subprocess
import uuid
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── lazy imports ──────────────────────────────────────────────────────────────

def _require(module: str, pip_name: str = None):
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError:
        pkg = pip_name or module
        raise SystemExit(
            f"[error] Missing package: '{pkg}'\n"
            f"        Run:  pip install {pkg}"
        )


# ── Config ────────────────────────────────────────────────────────────────────

HF_SPACE        = "banao-tech/vibe-voice-custom-voices"
NARRATION_MODEL = os.environ.get("NARRATION_MODEL", "gemini-2.5-pro")
BOARD_WIDTH     = 414
BOARD_HEIGHT    = 736


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 -- Generate Hindi Trump vs Modi narration via Gemini
# ══════════════════════════════════════════════════════════════════════════════

NARRATION_PROMPT = """
You are a satire comedy writer. Create a sarcastically funny Hindi dialogue between Trump [1] and Modi [2]
explaining the given ML/CS topic. Rules:
- Exactly 8-10 lines total, alternating [1] and [2]
- Output lines primarily in Hindi (Devanagari script), allowing English only for technical terms
- Trump is overconfident, often confused, and slightly dim-witted in a playful way; keep it witty and non-hateful
- Modi explains patiently with desi analogies
- Use most indian way of comedy, not english, also not include fake news term here
- Tone should be sarcastic and punchy, not plain funny
- Each line MAX 25 words
- End with both understanding the concept
- Output ONLY the dialogue lines, one per line, exactly in this format:
  [1]: <line>
  [2]: <line>
No extra text, no intro, no outro.

IMPORTANT: BOTH MUST SPEAK IN HINDI LANGUAGE. ONLY TECHNICAL TERMS OR JARGON IS ALLOWED.

Example style:
[1]: Modi bhai, yeh Gradient Descent kya hai? Kuch samajh nahi aaya!
[2]: Are Trump bhai! Socho -- ek pahad hai, gend ko neeche pahunchana hai.
[1]: Neeche kyun Modi? Main toh TOP pe rehta hoon -- America First!
[2]: Yahan ulta hai! Neeche matlab Loss kam hai -- yahi ML ka khel hai!
""".strip()


def generate_narration(topic: str) -> list[dict]:
    """
    Returns list of:  [{"speaker": 1, "line": "..."}, ...]
    speaker 1 = Trump, speaker 2 = Modi
    """
    from google import genai as google_genai
    from google.genai import types as google_types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("[error] GEMINI_API_KEY not set in .env")

    client = google_genai.Client(api_key=api_key)
    print(f"[narr]  Generating Hindi dialogue for: {topic!r} ...")
    prompt = f"{NARRATION_PROMPT}\n\nTopic: {topic}"

    raw = ""
    last_error = None
    for model_name in [NARRATION_MODEL, "gemini-2.0-flash"]:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=google_types.GenerateContentConfig(
                    temperature=0.9,
                    max_output_tokens=800,
                ),
            )
            raw = (response.text or "").strip()
            if raw:
                break
        except Exception as e:
            last_error = e
            print(f"[warn]  Narration model '{model_name}' failed: {e}")

    if not raw:
        raise SystemExit(f"[error] Narration generation failed: {last_error}")

    lines = []
    for raw_line in raw.splitlines():
        raw_line = raw_line.strip()
        m = re.match(r"\[([12])\]:\s*(.+)", raw_line)
        if m:
            lines.append({"speaker": int(m.group(1)), "line": m.group(2).strip()})

    if not lines:
        raise SystemExit("[error] Gemini returned no parseable dialogue lines.")

    print(f"[narr]  {len(lines)} lines generated")
    return lines


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 -- TTS via HuggingFace Gradio
# ══════════════════════════════════════════════════════════════════════════════

def generate_audio(
    dialogue: list[dict],
    voice_trump: str,
    voice_modi: str,
    audio_dir: Path,
    hf_space: str,
    hf_token: str | None,
) -> list[Path]:
    """
    Sends each dialogue line to the TTS Gradio space.
    Returns list of audio file paths in dialogue order.
    """
    gradio_client = _require("gradio_client")
    Client      = gradio_client.Client
    handle_file = gradio_client.handle_file

    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"[tts]   Connecting to HuggingFace space: {hf_space} ...")
    client_kwargs = {}
    if hf_token:
        client_kwargs["hf_token"] = hf_token
    try:
        client = Client(hf_space, **client_kwargs)
    except TypeError:
        client = Client(hf_space)

    trump_voice = handle_file(voice_trump)
    modi_voice  = handle_file(voice_modi)

    audio_paths = []
    for i, entry in enumerate(dialogue):
        speaker  = entry["speaker"]
        text     = entry["line"]
        out_file = audio_dir / f"line_{i+1:02d}_spk{speaker}.wav"

        spk1 = trump_voice if speaker == 1 else modi_voice
        spk2 = modi_voice  if speaker == 1 else trump_voice

        print(f"[tts]   Line {i+1}/{len(dialogue)} (Speaker {speaker}): {text[:50]}...")

        try:
            result = client.predict(
                text=text,
                speaker1_audio_path=spk1,
                speaker2_audio_path=spk2,
                speaker3_audio_path=spk1,
                speaker4_audio_path=spk2,
                seed=42,
                diffusion_steps=20,
                cfg_scale=1.3,
                use_sampling=False,
                temperature=0.95,
                top_p=0.95,
                max_words_per_chunk=250,
                api_name="/generate_speech_gradio",
            )
            if isinstance(result, dict):
                src = result.get("value") or result.get("path") or result.get("name")
            else:
                src = result

            shutil.copy2(src, out_file)
            print(f"[tts]   Saved -> {out_file}")
        except Exception as exc:
            print(f"[warn]  TTS failed for line {i+1}: {exc}")
            _silent_wav(out_file, duration=2)

        audio_paths.append(out_file)

    print(f"[tts]   All {len(audio_paths)} audio files ready")
    return audio_paths


def _silent_wav(path: Path, duration: int = 2):
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"anullsrc=r=44100:cl=stereo",
        "-ar", "44100",
        "-t", str(duration),
        str(path),
    ], capture_output=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 -- Playwright: render HTML board -> screenshots per dialogue beat
# ══════════════════════════════════════════════════════════════════════════════

def render_html_frames(
    html_path: Path,
    dialogue: list[dict],
    audio_paths: list[Path],
    frames_dir: Path,
) -> list[Path]:
    pw_module = _require("playwright.sync_api", "playwright")
    sync_playwright = pw_module.sync_playwright

    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []

    durations = []
    for ap in audio_paths:
        dur = _get_audio_duration(ap)
        durations.append(dur if dur else 3.0)

    html_url = html_path.resolve().as_uri()
    print(f"[frames] Launching Playwright -> {html_url}")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": BOARD_WIDTH, "height": BOARD_HEIGHT}
        )
        page.goto(html_url, wait_until="networkidle")
        page.wait_for_timeout(8500)

        page.add_style_tag(content="""
            #subtitle-overlay {
                position: fixed;
                bottom: 54px;
                left: 50%;
                transform: translateX(-50%);
                width: 88%;
                background: rgba(0,0,0,0.72);
                border-radius: 8px;
                padding: 8px 14px;
                font-family: 'Caveat', cursive;
                font-size: 15px;
                color: #f5f0e8;
                text-align: center;
                line-height: 1.5;
                z-index: 9999;
                display: none;
                border: 1px solid rgba(245,240,232,0.2);
            }
            #subtitle-overlay.visible { display: block; }
            #subtitle-speaker {
                font-size: 11px;
                letter-spacing: 1.5px;
                text-transform: uppercase;
                margin-bottom: 3px;
                opacity: 0.6;
            }
        """)

        page.evaluate("""
            const div = document.createElement('div');
            div.id = 'subtitle-overlay';
            div.innerHTML = '<div id="subtitle-speaker"></div><div id="subtitle-text"></div>';
            document.body.appendChild(div);
        """)

        for i, (entry, audio_path, duration) in enumerate(
            zip(dialogue, audio_paths, durations)
        ):
            speaker_name = "Trump" if entry["speaker"] == 1 else "Modi"
            line_text    = entry["line"]

            page.evaluate(f"""
                document.getElementById('subtitle-speaker').textContent = {json.dumps(speaker_name)};
                document.getElementById('subtitle-text').textContent    = {json.dumps(line_text)};
                document.getElementById('subtitle-overlay').classList.add('visible');
            """)

            frame_path = frames_dir / f"frame_{i+1:03d}.png"
            page.screenshot(path=str(frame_path), full_page=False)
            frame_paths.append(frame_path)
            print(f"[frames] Frame {i+1}/{len(dialogue)} -> {frame_path.name} ({duration:.1f}s)")

        page.evaluate("document.getElementById('subtitle-overlay').classList.remove('visible')")
        outro_path = frames_dir / f"frame_{len(dialogue)+1:03d}.png"
        page.screenshot(path=str(outro_path), full_page=False)
        frame_paths.append(outro_path)

        browser.close()

    print(f"[frames] {len(frame_paths)} frames rendered")
    return frame_paths


def _get_audio_duration(path: Path) -> float | None:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 -- FFmpeg: stitch frames + audio -> MP4
# ══════════════════════════════════════════════════════════════════════════════

def build_video(
    frame_paths: list[Path],
    audio_paths: list[Path],
    durations: list[float],
    output_path: Path,
    fps: int = 24,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.parent / f"_tmp_segments_{uuid.uuid4().hex[:8]}"
    tmp.mkdir(exist_ok=True)

    segment_paths = []

    for i, (frame, audio, dur) in enumerate(zip(frame_paths, audio_paths, durations)):
        seg = tmp / f"seg_{i:03d}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(frame),
            "-i", str(audio),
            "-c:v", "libx264", "-preset", "fast",
            "-tune", "stillimage",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-vf", f"scale={BOARD_WIDTH}:{BOARD_HEIGHT}:force_original_aspect_ratio=decrease,"
                   f"pad={BOARD_WIDTH}:{BOARD_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black",
            "-t", str(dur),
            "-shortest",
            str(seg),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"[error] FFmpeg segment encode failed at segment {i+1}.\n"
                f"stderr:\n{exc.stderr}"
            )
        segment_paths.append(seg)
        print(f"[video]  Segment {i+1}/{len(audio_paths)} encoded ({dur:.1f}s)")

    if len(frame_paths) > len(audio_paths):
        outro_frame = frame_paths[-1]
        outro_seg   = tmp / "seg_outro.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(outro_frame),
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            "-vf", f"scale={BOARD_WIDTH}:{BOARD_HEIGHT}:force_original_aspect_ratio=decrease,"
                   f"pad={BOARD_WIDTH}:{BOARD_HEIGHT}:(ow-iw)/2:(oh-ih)/2:color=black",
            "-t", "2",
            "-shortest",
            str(outro_seg),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"[error] FFmpeg outro encode failed.\nstderr:\n{exc.stderr}")
        segment_paths.append(outro_seg)

    concat_list = tmp / "concat.txt"
    with open(concat_list, "w") as f:
        for sp in segment_paths:
            f.write(f"file '{sp.resolve()}'\n")

    print(f"[video]  Concatenating {len(segment_paths)} segments -> {output_path}")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    print(f"[video]  Final video -> {output_path.resolve()}")

    shutil.rmtree(tmp, ignore_errors=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Hindi Trump-Modi narration, TTS audio, and render video from chalkboard HTML."
    )
    parser.add_argument("--html",        required=True, help="Path to the generated chalkboard HTML file")
    parser.add_argument("--topic",       required=True, help="Topic name")
    parser.add_argument("--voice-trump", required=True, help="WAV file for Trump's voice")
    parser.add_argument("--voice-modi",  required=True, help="WAV file for Modi's voice")
    parser.add_argument("--output",      default=None,  help="Output MP4 path")
    parser.add_argument("--save-script", action="store_true", help="Save dialogue script as JSON")
    parser.add_argument("--fps",         type=int, default=24, help="Output video FPS")
    parser.add_argument("--hf-space",    default=os.environ.get("HF_SPACE", HF_SPACE))
    parser.add_argument("--hf-token",    default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"))
    parser.add_argument("--run-root",    default="output")
    parser.add_argument("--keep-workdir", action="store_true")
    return parser.parse_args()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _resolve_html_path(raw_html_path: Path) -> Path:
    if not raw_html_path.exists():
        raise SystemExit(f"[error] HTML file not found: {raw_html_path.resolve()}")
    return raw_html_path


def main():
    args  = parse_args()
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_slug = slug(args.topic)

    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    html_path  = _resolve_html_path(Path(args.html))

    for label, vpath in [("--voice-trump", args.voice_trump), ("--voice-modi", args.voice_modi)]:
        if not Path(vpath).exists():
            raise SystemExit(f"[error] Voice file not found ({label}): {vpath}")

    work_dir = run_root / f"_work_{topic_slug}_{ts}"
    work_dir.mkdir(parents=True, exist_ok=True)

    out_dir    = run_root
    output_path = Path(args.output) if args.output else out_dir / f"{topic_slug}_{ts}_video.mp4"

    dialogue = generate_narration(args.topic)

    if args.save_script:
        script_path = out_dir / f"{topic_slug}_{ts}_script.json"
        script_path.write_text(json.dumps(dialogue, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[narr]  Script saved -> {script_path}")

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

    frames_dir  = work_dir / "frames"
    frame_paths = render_html_frames(html_path, dialogue, audio_paths, frames_dir)

    durations = [_get_audio_duration(ap) or 3.0 for ap in audio_paths]
    build_video(frame_paths, audio_paths, durations, output_path, fps=args.fps)

    if not args.keep_workdir:
        shutil.rmtree(work_dir, ignore_errors=True)
    else:
        print(f"[debug] Work files kept at: {work_dir.resolve()}")

    print(f"\nDone! Video saved -> {output_path.resolve()}")


if __name__ == "__main__":
    main()
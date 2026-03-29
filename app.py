
import os
import sys
import subprocess
import tempfile
import logging
from pathlib import Path
from datetime import datetime
import gradio as gr

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

HERE         = Path(__file__).parent.resolve()
LOGO_PATH    = HERE / "logo" / "logo.png"
RUN_PIPELINE = HERE / "run_pipeline.py"

log.info("=== ByteBrain starting up ===")
log.info(f"Working directory : {HERE}")
log.info(f"Python executable : {sys.executable}")
log.info(f"run_pipeline.py exists: {RUN_PIPELINE.exists()}")


def _slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def generate_video(topic: str):
    topic = (topic or "").strip()
    if not topic:
        raise gr.Error("Please enter a topic.")

    log.info(f"Generate requested — topic: '{topic}'")

    ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root  = Path(tempfile.gettempdir()) / "bytebrain-output"
    output_root.mkdir(parents=True, exist_ok=True)
    output_video = output_root / f"{_slug(topic)}_{ts}_video.mp4"

    log.info(f"Output path: {output_video}")

    env = os.environ.copy()  # passes all HF Secrets automatically
    env["PIPELINE_OUTPUT_DIR"] = str(output_root)
    env["PYTHONIOENCODING"]    = "utf-8"
    env["PYTHONUTF8"]          = "1"

    cmd = [
        sys.executable,
        "-X", "utf8",
        str(RUN_PIPELINE),
        topic,
        "--output", str(output_video),
    ]

    log.info(f"Running command: {' '.join(cmd)}")

    proc = subprocess.run(
        cmd,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(HERE),
        env=env,
    )

    logs = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")

    log.info(f"Pipeline exited with code: {proc.returncode}")
    if proc.stdout:
        log.info(f"STDOUT:\n{proc.stdout[-2000:]}")
    if proc.stderr:
        log.warning(f"STDERR:\n{proc.stderr[-2000:]}")

    if proc.returncode != 0:
        tail = "\n".join(logs.strip().splitlines()[-50:])
        log.error("Pipeline failed.")
        raise gr.Error(f"Generation failed.\n{tail}")

    if not output_video.exists():
        log.error("Output video file missing after pipeline run.")
        raise gr.Error("Pipeline finished but output video file is missing.")

    log.info("Video generation successful.")
    return str(output_video), logs


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="ByteBrain Video Generator") as demo:
    if LOGO_PATH.exists():
        gr.Image(value=str(LOGO_PATH), show_label=False, width=140, height=140)
    gr.Markdown("## ByteBrain — Topic to Video")
    gr.Markdown(
        "Enter any ML/CS topic and get a chalkboard explainer video "
        "with Hindi Trump-Modi narration."
    )
    topic_input  = gr.Textbox(label="Topic", placeholder="e.g. Softmax Function")
    generate_btn = gr.Button("Generate Video", variant="primary")
    video_output = gr.Video(label="Generated Video")
    logs_output  = gr.Textbox(label="Pipeline Logs", lines=20, max_lines=40)

    generate_btn.click(
        fn=generate_video,
        inputs=topic_input,
        outputs=[video_output, logs_output],
    )

log.info("Gradio UI built successfully.")

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        ssr_mode=False,
    )
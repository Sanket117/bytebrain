"""
api.py
------
FastAPI backend for ByteBrain pipeline.
Exposes endpoints to submit a topic, poll job status, and download the video.

Endpoints:
    POST /generate          -- submit a topic, returns job_id
    GET  /status/{job_id}   -- poll job status + logs
    GET  /download/{job_id} -- download the final MP4
    GET  /health            -- health check
    GET  /docs              -- auto Swagger UI (built into FastAPI)

Usage:
    pip install fastapi uvicorn python-multipart
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

From your HTML frontend:
    POST http://localhost:8000/generate   { "topic": "Gradient Descent" }
    GET  http://localhost:8000/status/<job_id>
    GET  http://localhost:8000/download/<job_id>
"""

import os
import sys
import uuid
import subprocess
import tempfile
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

HERE         = Path(__file__).parent.resolve()
RUN_PIPELINE = HERE / "run_pipeline.py"
OUTPUT_DIR   = Path(os.environ.get("PIPELINE_OUTPUT_DIR",
                    str(Path(tempfile.gettempdir()) / "bytebrain-output")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── In-memory job store ───────────────────────────────────────────────────────
# For production replace with Redis or a database

jobs: dict[str, dict] = {}
# job schema:
# {
#   "job_id":     str,
#   "topic":      str,
#   "status":     "queued" | "running" | "done" | "failed",
#   "created_at": str,
#   "finished_at": str | None,
#   "video_path": str | None,
#   "logs":       str,
#   "error":      str | None,
# }


# ── Pydantic models ───────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    topic: str
    voice_trump: Optional[str] = None   # override voice path
    voice_modi:  Optional[str] = None
    no_diagram:  bool          = False


class JobStatusResponse(BaseModel):
    job_id:      str
    topic:       str
    status:      str
    created_at:  str
    finished_at: Optional[str]
    logs:        str
    error:       Optional[str]
    download_url: Optional[str]


# ── App ───────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== ByteBrain API starting up ===")
    log.info(f"Working dir     : {HERE}")
    log.info(f"Output dir      : {OUTPUT_DIR}")
    log.info(f"run_pipeline.py : {RUN_PIPELINE.exists()}")
    yield
    log.info("=== ByteBrain API shutting down ===")


app = FastAPI(
    title="ByteBrain API",
    description="Submit a topic → get a chalkboard explainer video with Trump-Modi narration.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins for local dev / HF Spaces frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Background worker ─────────────────────────────────────────────────────────

def _slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def run_pipeline_job(job_id: str):
    job   = jobs[job_id]
    topic = job["topic"]

    job["status"] = "running"
    log.info(f"[{job_id}] Starting pipeline for topic: '{topic}'")

    ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video = OUTPUT_DIR / f"{_slug(topic)}_{ts}_video.mp4"

    env = os.environ.copy()
    env["PIPELINE_OUTPUT_DIR"] = str(OUTPUT_DIR)
    env["PYTHONIOENCODING"]    = "utf-8"
    env["PYTHONUTF8"]          = "1"

    cmd = [
        sys.executable,
        "-X", "utf8",
        str(RUN_PIPELINE),
        topic,
        "--output", str(output_video),
    ]

    # Optional overrides
    if job.get("voice_trump"):
        cmd += ["--voice-trump", job["voice_trump"]]
    if job.get("voice_modi"):
        cmd += ["--voice-modi", job["voice_modi"]]
    if job.get("no_diagram"):
        cmd.append("--no-diagram")

    log.info(f"[{job_id}] CMD: {' '.join(cmd)}")

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(HERE),
            env=env,
        )

        logs = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        job["logs"] = logs

        log.info(f"[{job_id}] Pipeline exited with code {proc.returncode}")

        if proc.returncode != 0:
            job["status"] = "failed"
            job["error"]  = "\n".join(logs.strip().splitlines()[-50:])
            log.error(f"[{job_id}] Pipeline failed.")
        elif not output_video.exists():
            job["status"] = "failed"
            job["error"]  = "Pipeline finished but output video file is missing."
            log.error(f"[{job_id}] Video file missing.")
        else:
            job["status"]     = "done"
            job["video_path"] = str(output_video)
            log.info(f"[{job_id}] Done -> {output_video}")

    except Exception as e:
        job["status"] = "failed"
        job["error"]  = str(e)
        log.exception(f"[{job_id}] Unexpected error")

    finally:
        job["finished_at"] = datetime.utcnow().isoformat()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Quick health check."""
    return {
        "status":           "ok",
        "pipeline_exists":  RUN_PIPELINE.exists(),
        "output_dir":       str(OUTPUT_DIR),
        "active_jobs":      len([j for j in jobs.values() if j["status"] == "running"]),
    }


@app.post("/generate", response_model=dict)
def generate(req: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Submit a topic for video generation.
    Returns a job_id you can poll with GET /status/{job_id}.
    """
    topic = (req.topic or "").strip()
    if not topic:
        raise HTTPException(status_code=400, detail="topic cannot be empty.")

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id":      job_id,
        "topic":       topic,
        "status":      "queued",
        "created_at":  datetime.utcnow().isoformat(),
        "finished_at": None,
        "video_path":  None,
        "logs":        "",
        "error":       None,
        "voice_trump": req.voice_trump,
        "voice_modi":  req.voice_modi,
        "no_diagram":  req.no_diagram,
    }

    background_tasks.add_task(run_pipeline_job, job_id)

    log.info(f"Job created: {job_id} — topic: '{topic}'")

    return {
        "job_id":      job_id,
        "status":      "queued",
        "status_url":  f"/status/{job_id}",
        "download_url": f"/download/{job_id}",
    }


@app.get("/status/{job_id}", response_model=JobStatusResponse)
def status(job_id: str):
    """
    Poll job status.
    status values: queued | running | done | failed
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    download_url = f"/download/{job_id}" if job["status"] == "done" else None

    return JobStatusResponse(
        job_id       = job["job_id"],
        topic        = job["topic"],
        status       = job["status"],
        created_at   = job["created_at"],
        finished_at  = job["finished_at"],
        logs         = job["logs"],
        error        = job["error"],
        download_url = download_url,
    )


@app.get("/download/{job_id}")
def download(job_id: str):
    """
    Download the generated MP4 once status == done.
    """
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Job is not done yet. Status: {job['status']}")

    video_path = Path(job["video_path"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk.")

    filename = f"bytebrain_{_slug(job['topic'])}.mp4"
    return FileResponse(
        path        = str(video_path),
        media_type  = "video/mp4",
        filename    = filename,
    )


@app.get("/jobs")
def list_jobs():
    """List all jobs (for debugging)."""
    return [
        {
            "job_id":     j["job_id"],
            "topic":      j["topic"],
            "status":     j["status"],
            "created_at": j["created_at"],
        }
        for j in jobs.values()
    ]
# ByteBrain

ByteBrain is a topic-to-video generator that creates chalkboard-style explainer videos with:
- AI-generated lesson board (HTML + animated SVG)
- Hindi Trump/Modi style narration
- Voice synthesis
- Avatar-aware on-screen speaker visualization
- Final MP4 export

## Live Demo

- Hugging Face Space: https://huggingface.co/spaces/Sanket17/bytebrains

## Demo Videos

<table>
<tr>
<td width="50%">

### CNN Explainer
---
https://github.com/Sanket117/bytebrain/blob/main/demo/cnn.mp4

</td>
<td width="50%">

### Linear Regression Explainer
---
https://github.com/Sanket117/bytebrain/blob/main/demo/linear_regression.mp4

</td>
</tr>
</table>

## Features

- Simple Gradio UI with ByteBrain branding (`logo/logo.png`)
- Single text input: enter topic and generate full video
- Docker-ready for Hugging Face Spaces
- Temp-first output design for hosted environments

## Project Structure
...

```text
bytebrain/
├─ app.py
├─ run_pipeline.py
├─ generate_chalkboard.py
├─ narrate_and_render.py
├─ template.html
├─ logo/
│  └─ logo.png
├─ avatar/
│  ├─ trump.png
│  └─ modi.png
├─ voices/
│  ├─ trump.wav
│  └─ modi.wav
├─ demo/
│  └─ bytebrain_demo.mp4
├─ Dockerfile
└─ requirements.txt
```

## Environment Variables

Required:
- `GEMINI_API_KEY`
- `OPENAI_API_KEY`

Recommended:
- `HF_TOKEN`

Optional:
- `HF_SPACE` (default: `banao-tech/vibe-voice-custom-voices`)
- `OPENAI_TRANSCRIBE_MODEL` (default: `gpt-4o-mini-transcribe`)
- `NARRATION_MODEL` (default: `gemini-2.5-pro`)
- `PIPELINE_OUTPUT_DIR` (default: temp dir, e.g. `/tmp/bytebrain-output`)

## Output Storage

Generated files go to temp storage by default:
- Linux/Hugging Face: `/tmp/bytebrain-output`
- Windows/macOS: system temp + `bytebrain-output`

## Local Run

```bash
pip install -r requirements.txt
playwright install chromium
python app.py
```

Open: `http://localhost:7860`

## Docker Run

```bash
docker build -t bytebrain-app .
docker run -p 7860:7860 \
  -e GEMINI_API_KEY=your_key \
  -e OPENAI_API_KEY=your_key \
  -e HF_TOKEN=your_hf_token \
  bytebrain-app
```

## Deploy to Hugging Face Space

1. Create/Use Docker Space: `Sanket17/bytebrain`
2. Push this `bytebrain/` directory contents to the Space repo
3. Add secrets in Space settings:
   - `GEMINI_API_KEY`
   - `OPENAI_API_KEY`
   - `HF_TOKEN`
4. Space will build via `Dockerfile` and run `python app.py`

---
For Space config fields reference: https://huggingface.co/docs/hub/spaces-config-reference

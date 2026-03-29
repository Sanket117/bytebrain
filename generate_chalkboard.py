"""
generate_chalkboard.py
──────────────────────────────────────────────────────────────────────────────
Two-pass Gemini pipeline:
  Pass 1 — structured JSON  (title, bullets, formula, key terms ...)
  Pass 2 — animated SVG diagram  (chalk-style, topic-specific)

Both are injected into the Jinja2 HTML template to produce a complete board.

Usage
-----
    python generate_chalkboard.py "Backpropagation"
    python generate_chalkboard.py "Attention Mechanism" --save-json
    python generate_chalkboard.py --from-json output/backprop_....json

Requirements
------------
    pip install google-genai jinja2 python-dotenv

.env
----
    GEMINI_API_KEY=your-key-here
"""

import os
import re
import json
import argparse
from pathlib import Path
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google import genai
from google.genai import types
from jinja2 import Environment, FileSystemLoader


# ── Config ────────────────────────────────────────────────────────────────────

MODEL         = "gemini-2.5-pro"
TEMPLATE_FILE = "template.html"
TEMPLATE_DIR  = Path(__file__).parent
OUTPUT_DIR    = Path(__file__).parent / "output"


# ── Pass 1 prompt — structured content JSON ───────────────────────────────────

CONTENT_PROMPT = """
You are an expert ML/CS educator creating chalk-board explainers.
Output ONLY a valid JSON object -- no markdown fences, no prose, no extra keys.

Schema (follow exactly):
{
  "title":        "<emoji + topic name, <=40 chars>",
  "subtitle":     "<domain · tagline, <=60 chars>",
  "idea_items":   [
    {"bullet": "->", "html": "<note with <span class='colour'> tags>"},
    {"bullet": "->", "html": "..."},
    {"bullet": "->", "html": "..."}
  ],
  "minima_label": "<section label <=30 chars>",
  "minima_items": [
    {"bullet": "x", "html": "<span class=\"pink ul\">bad thing</span> -- why bad"},
    {"bullet": "v", "html": "<span class=\"yellow ul\">good thing</span> -- why good"}
  ],
  "formula":      "<core equation, plain text + <span> ok>",
  "key_terms":    [
    {"bullet": "<sym>", "html": "<span class=\"colour ul\">term</span> -- def"},
    {"bullet": "<sym>", "html": "..."},
    {"bullet": "<sym>", "html": "..."}
  ],
  "footnote":     "* <tip <=80 chars>",
  "extra_label":  "<bottom-right label <=40 chars>",
  "extra_sub":    "<bottom-right hint <=60 chars>",
  "diagram_hint": "<one sentence: what the diagram should visually show>"
}

Allowed colour classes: yellow  pink  blue  orange
Underline class: ul (combine with colour, e.g. class="blue ul")
Only <span> tags inside html values. Output ONLY the JSON.
""".strip()


# ── Pass 2 prompt — animated SVG diagram ─────────────────────────────────────

DIAGRAM_PROMPT = """
You are an SVG animation expert creating chalk-style educational diagrams.

CONTEXT
-------
Topic   : {topic}
Hint    : {hint}
Colours :
  --chalk-white  #f5f0e8
  --chalk-yellow #f7e06a
  --chalk-pink   #f4a0b0
  --chalk-blue   #a0c4f4
  --chalk-orange #f4b87a
  board bg       #2d5a27  (dark green chalkboard)

OUTPUT RULES
------------
- Output ONLY a raw <svg> element -- no wrapper, no markdown, no explanation.
- viewBox="0 0 354 300"  width="354"  height="300"
- style="filter:url(#chalk-filter)"  (already defined in the page)
- All strokes/fills use the colour values above.
- Every line/path that "draws on" must use this pattern:
    stroke-dasharray="<len>"  stroke-dashoffset="<len>"
    style="animation: drawOn <dur>s ease forwards <delay>s"
- Every element that pops in must use:
    opacity="0"  style="animation: popIn 0.3s ease forwards <delay>s"
- Use font-family="Patrick Hand, cursive" or "Caveat, cursive" for labels.
- Keep the diagram clear and readable -- axes, curves, nodes, arrows, labels.
- The diagram must be TOPIC-SPECIFIC and visually explain the concept.
- Total animation duration should be 5-8 seconds.
- Add a <style> block INSIDE the <svg> with ONLY these two keyframes
  (do NOT redefine any other keyframe -- they already exist globally):
    @keyframes drawOn  {{ to {{ stroke-dashoffset: 0; }} }}
    @keyframes popIn   {{ from {{ opacity:0; transform:scale(.5); }} to {{ opacity:1; transform:scale(1); }} }}

Output the <svg>...</svg> block only. Nothing else.
""".strip()


# ── Gemini client ─────────────────────────────────────────────────────────────

def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("[error] GEMINI_API_KEY not set. Add it to your .env file.")
        raise SystemExit(1)
    return genai.Client(api_key=api_key)


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


# ── Pass 1 — content JSON ─────────────────────────────────────────────────────

def generate_content(topic: str) -> dict:
    print(f"[pass1] Generating content JSON for: {topic!r} ...")
    client = _get_client()
    response = client.models.generate_content(
        model=MODEL,
        contents=f"{CONTENT_PROMPT}\n\nTopic: {topic}",
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=4096,
        ),
    )
    raw = _strip_fences(response.text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print("[error] Pass 1 did not return valid JSON:")
        print(raw[:800])
        raise SystemExit(1) from exc
    print("[pass1] Content JSON OK")
    return data


# ── Pass 2 — animated SVG ─────────────────────────────────────────────────────

def generate_diagram(topic: str, hint: str) -> str | None:
    print("[pass2] Generating SVG diagram ...")
    client = _get_client()
    prompt = DIAGRAM_PROMPT.format(topic=topic, hint=hint)
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.9,
            max_output_tokens=8192,
        ),
    )
    raw = _strip_fences(response.text)

    if not raw.strip().startswith("<svg"):
        match = re.search(r"(<svg[\s\S]+?</svg>)", raw, re.IGNORECASE)
        if match:
            raw = match.group(1)
        else:
            print("[warn]  Pass 2 did not return a valid <svg>. Diagram will use placeholder.")
            return None

    print("[pass2] SVG diagram OK")
    return raw


# ── Jinja rendering ───────────────────────────────────────────────────────────

def render_template(ctx: dict, template_file: str = TEMPLATE_FILE) -> str:
    template_path = TEMPLATE_DIR / template_file
    if not template_path.exists():
        raise FileNotFoundError(
            f"Template not found: {template_path}\n"
            f"Ensure '{template_file}' is in the same folder as this script."
        )
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=False,
    )
    return env.get_template(template_file).render(**ctx)


# ── Output helpers ────────────────────────────────────────────────────────────

def save_html(html: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"[out]   HTML -> {path}")


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[out]   JSON -> {path}")


# ── Build Jinja context ───────────────────────────────────────────────────────

def build_context(data: dict, diagram_svg: str | None) -> dict:

    def render_list(items: list) -> str:
        lis = []
        for item in items:
            bullet = item.get("bullet", "->")
            html   = item.get("html",   "")
            lis.append(
                f'<li data-b="{bullet}" '
                f'style="font-family:\'Indie Flower\',cursive;font-size:15px;'
                f'line-height:1.4;padding-left:22px;position:relative;'
                f'margin-bottom:3px;color:var(--chalk-white);">'
                f'{html}</li>'
            )
        return "\n".join(lis)

    return {
        # Header
        "title":             data.get("title",        "Topic"),
        "subtitle":          data.get("subtitle",     ""),

        # Top-right side notes
        "idea_items_html":   render_list(data.get("idea_items",   [])),
        "minima_label":      data.get("minima_label", "Key Contrast"),
        "minima_items_html": render_list(data.get("minima_items", [])),

        # Bottom-left
        "formula":           data.get("formula",      ""),
        "key_terms_html":    render_list(data.get("key_terms",    [])),
        "footnote":          data.get("footnote",     ""),

        # Bottom-right placeholder
        "extra_label":       data.get("extra_label",  "Your content here"),
        "extra_sub":         data.get("extra_sub",    "Pass extra_content to fill"),
        "extra_content":     None,

        # Diagram slot -- filled by Pass 2 SVG (or None -> placeholder)
        "diagram_label":     data.get("diagram_label", "Diagram"),
        "diagram_sub":       data.get("diagram_sub",   ""),
        "diagram_content":   diagram_svg,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a full chalk-board HTML explainer via Gemini (2-pass)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("topic", nargs="?",
                       help="Topic to explain e.g. 'Backpropagation'")
    group.add_argument("--from-json", metavar="FILE",
                       help="Skip Pass 1 -- load content from JSON file (still runs Pass 2)")

    parser.add_argument("--output", "-o", metavar="FILE",
                        help="Output HTML path (default: output/<slug>_<ts>.html)")
    parser.add_argument("--save-json", action="store_true",
                        help="Save raw content JSON alongside the HTML")
    parser.add_argument("--no-diagram", action="store_true",
                        help="Skip Pass 2 -- leave diagram as placeholder")
    parser.add_argument("--template", default=TEMPLATE_FILE,
                        help=f"Jinja2 template file (default: {TEMPLATE_FILE})")
    return parser.parse_args()


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # -- Pass 1: content JSON
    if args.from_json:
        json_path  = Path(args.from_json)
        print(f"[load]  Reading JSON from {json_path} ...")
        raw_data   = json.loads(json_path.read_text(encoding="utf-8"))
        topic      = raw_data.get("title", json_path.stem)
        topic_slug = slug(topic)
    else:
        topic      = args.topic
        raw_data   = generate_content(topic)
        topic_slug = slug(topic)

    if args.save_json and not args.from_json:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_json(raw_data, OUTPUT_DIR / f"{topic_slug}_{ts}.json")

    # -- Pass 2: animated SVG diagram
    diagram_svg = None
    if not args.no_diagram:
        hint = raw_data.get("diagram_hint", f"A visual diagram explaining {topic}")
        diagram_svg = generate_diagram(topic, hint)

    # -- Build context + render
    ctx  = build_context(raw_data, diagram_svg)
    html = render_template(ctx, template_file=args.template)

    # -- Save output
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else OUTPUT_DIR / f"{topic_slug}_{ts}.html"
    save_html(html, out_path)

    print(f"\nDone! Open in browser -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
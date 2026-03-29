FROM python:3.10-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install playwright + chromium in one step with progress
RUN playwright install-deps chromium && \
    playwright install chromium

COPY . .

RUN mkdir -p /tmp/bytebrain-output

ENV PYTHONUNBUFFERED=1
ENV PYTHONUTF8=1
ENV PYTHONIOENCODING=utf-8
ENV PIPELINE_OUTPUT_DIR=/tmp/bytebrain-output

EXPOSE 7860

CMD ["python", "app.py"]
# syntax=docker/dockerfile:1
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN --mount=type=secret,id=build_ca \
    if [ -f /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca; fi \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch==2.8.0+cpu" \
    && pip install --no-cache-dir -r requirements.txt
COPY --chmod=0644 app.py transcribe.py server.py ./
COPY web ./web
RUN chmod -R a+rX web && useradd --create-home openscribe && mkdir outputs && chown openscribe:openscribe outputs
USER openscribe
ENV GRADIO_SERVER_NAME=0.0.0.0 GRADIO_ANALYTICS_ENABLED=False
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]

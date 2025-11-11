FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py
COPY templates /app/templates
COPY static /app/static

# Defaults (override at runtime or via .env)
ENV SOURCE_URL="http://esp32-s3.local/stream"
ENV VERBOSE="0"
ENV TARGET_FPS="25"
ENV MQTT_HOST="mqtt.local"
ENV MQTT_PORT="1883"
ENV MQTT_USER="mqtt_user"
ENV MQTT_PASSWORD=""
ENV MQTT_BASE_TOPIC="gesture32"
ENV MQTT_DISCOVERY_PREFIX="homeassistant"
ENV MQTT_CLIENT_ID="gesture32-server"
ENV PINCH_DEADZONE_PX="8"
ENV PINCH_HISTORY="8"
ENV CUSTOM_GESTURE_MAP="{}"
ENV PYTHONUNBUFFERED=1

EXPOSE 12345
CMD ["gunicorn", "-b", "0.0.0.0:12345", "app:app", "--workers", "1", "--threads", "4"]

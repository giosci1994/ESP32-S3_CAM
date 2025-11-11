FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends     ffmpeg libsm6 libxext6 curl     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py
COPY templates /app/templates
COPY static /app/static

# Defaults (override at runtime)
ENV SOURCE_URL="http://192.168.1.24/stream"
ENV VERBOSE="1"
ENV TARGET_FPS="25"
ENV MQTT_HOST="192.168.1.100"
ENV MQTT_PORT="1883"
<<<<<<< HEAD
ENV MQTT_USER="## mqtt user ##"
ENV MQTT_PASSWORD=""
ENV MQTT_BASE_TOPIC="## base topic mqtt ##"
=======
ENV MQTT_USER="mqtt_user"
ENV MQTT_PASSWORD="Casafiga94?"
ENV MQTT_BASE_TOPIC="gesture32"
>>>>>>> ef27f72 (ritorno alle origini)
ENV MQTT_DISCOVERY_PREFIX="homeassistant"
ENV PINCH_DEADZONE_PX="8"
ENV PINCH_HISTORY="8"
ENV CUSTOM_GESTURE_MAP="{}"
ENV PYTHONUNBUFFERED=1

EXPOSE 12345
CMD ["gunicorn", "-b", "0.0.0.0:12345", "app:app", "--workers", "1", "--threads", "4"]

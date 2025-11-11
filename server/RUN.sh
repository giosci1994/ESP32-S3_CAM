#!/usr/bin/env bash
# auto-load .env if present
[ -f .env ] && export $(grep -v '^#' .env | xargs)

set -euo pipefail

# Override via env if needed
SOURCE_URL="${SOURCE_URL:-http://## indirizzo ip esp32 ##/stream}"
VERBOSE="${VERBOSE:-1}"
TARGET_FPS="${TARGET_FPS:-25}"
MQTT_HOST="${MQTT_HOST:-## indirizzo ip mqtt ##}"
MQTT_PORT="${MQTT_PORT:-1883}"
## mqtt user ##="${## mqtt user ##:-## mqtt user ##}"
MQTT_PASSWORD="${MQTT_PASSWORD:-## tua password ##}"
MQTT_BASE_TOPIC="${MQTT_BASE_TOPIC:-## base topic mqtt ##}"
MQTT_DISCOVERY_PREFIX="${MQTT_DISCOVERY_PREFIX:-homeassistant}"
PINCH_DEADZONE_PX="${PINCH_DEADZONE_PX:-8}"
PINCH_HISTORY="${PINCH_HISTORY:-8}"
CUSTOM_GESTURE_MAP="${CUSTOM_GESTURE_MAP:-{}}"
PORT="${PORT:-12345}"

echo "[*] Building image http-gesture-mqtt:latest ..."
docker build -t http-gesture-mqtt:latest .

echo "[*] Starting container on port ${PORT} ..."
docker rm -f http-gesture-mqtt >/dev/null 2>&1 || true
docker run --name http-gesture-mqtt   -e SOURCE_URL="${SOURCE_URL}"   -e VERBOSE="${VERBOSE}"   -e TARGET_FPS="${TARGET_FPS}"   -e MQTT_HOST="${MQTT_HOST}"   -e MQTT_PORT="${MQTT_PORT}"   -e ## mqtt user ##="${## mqtt user ##}"   -e MQTT_PASSWORD="${MQTT_PASSWORD}"   -e MQTT_BASE_TOPIC="${MQTT_BASE_TOPIC}"   -e MQTT_DISCOVERY_PREFIX="${MQTT_DISCOVERY_PREFIX}"   -e PINCH_DEADZONE_PX="${PINCH_DEADZONE_PX}"   -e PINCH_HISTORY="${PINCH_HISTORY}"   -e CUSTOM_GESTURE_MAP='${CUSTOM_GESTURE_MAP}'   -p ${PORT}:12345 --restart unless-stopped   http-gesture-mqtt:latest

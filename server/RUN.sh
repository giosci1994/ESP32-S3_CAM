#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

load_env_file() {
  local env_file="$1"
  if [[ -f "${env_file}" ]]; then
    echo "[*] Loading environment from ${env_file}" >&2
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
  fi
}

load_env_file "${ROOT_DIR}/.env"
load_env_file "${SCRIPT_DIR}/.env"

# Override via env if needed
SOURCE_URL="${SOURCE_URL:-http://esp32-s3.local/stream}"
VERBOSE="${VERBOSE:-0}"
TARGET_FPS="${TARGET_FPS:-25}"
MQTT_HOST="${MQTT_HOST:-mqtt.local}"
MQTT_PORT="${MQTT_PORT:-1883}"
MQTT_USER="${MQTT_USER:-mqtt_user}"
MQTT_PASSWORD="${MQTT_PASSWORD:-}"
MQTT_BASE_TOPIC="${MQTT_BASE_TOPIC:-gesture32}"
MQTT_DISCOVERY_PREFIX="${MQTT_DISCOVERY_PREFIX:-homeassistant}"
PINCH_DEADZONE_PX="${PINCH_DEADZONE_PX:-8}"
PINCH_HISTORY="${PINCH_HISTORY:-8}"
CUSTOM_GESTURE_MAP="${CUSTOM_GESTURE_MAP:-{}}"
PORT="${PORT:-12345}"

IMAGE_NAME="http-gesture-mqtt:latest"
CONTAINER_NAME="http-gesture-mqtt"

echo "[*] Building image ${IMAGE_NAME} ..."
docker build -t "${IMAGE_NAME}" "${SCRIPT_DIR}"

echo "[*] Starting container on port ${PORT} ..."
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
docker run --name "${CONTAINER_NAME}" \
  -e SOURCE_URL="${SOURCE_URL}" \
  -e VERBOSE="${VERBOSE}" \
  -e TARGET_FPS="${TARGET_FPS}" \
  -e MQTT_HOST="${MQTT_HOST}" \
  -e MQTT_PORT="${MQTT_PORT}" \
  -e MQTT_USER="${MQTT_USER}" \
  -e MQTT_PASSWORD="${MQTT_PASSWORD}" \
  -e MQTT_BASE_TOPIC="${MQTT_BASE_TOPIC}" \
  -e MQTT_DISCOVERY_PREFIX="${MQTT_DISCOVERY_PREFIX}" \
  -e PINCH_DEADZONE_PX="${PINCH_DEADZONE_PX}" \
  -e PINCH_HISTORY="${PINCH_HISTORY}" \
  -e CUSTOM_GESTURE_MAP='${CUSTOM_GESTURE_MAP}' \
  -p "${PORT}:12345" \
  --restart unless-stopped \
  "${IMAGE_NAME}"

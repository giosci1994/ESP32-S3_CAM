# ESP32-S3_CAM

Questo pacchetto include **firmware** (ESP32-S3) e **server** (Docker Flask + MediaPipe + MQTT)

## Struttura
- `firmware/esp32s3-cam.ino` — placeholder: `## WIFI SSID ##`, `## WIFI PASSWORD ##`, `## indirizzo ip mqtt ##`
- `server/` — app completa con placeholder (usa `.env`):
  - `app.py`, `Dockerfile`, `docker-compose.yml` (con `env_file: .env`), `RUN.sh` (autoload .env), `requirements.txt`
  - `templates/index.html`, `static/style.css`
  - `.env.example` da copiare in `.env`

## Avvio
```bash
cd server
cp .env.example .env
# compila i placeholder
docker compose up -d
```

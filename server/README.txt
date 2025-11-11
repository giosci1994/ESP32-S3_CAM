ESP32 Gesture + MediaPipe + MQTT — Pinch Distance (no Face Detection)
====================================================================

Funzioni:
- Gesture mani (estese) + MQTT autodiscovery
- Pinch (pollice-indice): distanza in px e normalizzata (0..1) + trend (opening/closing/steady)
- No face detection

Endpoints:
- http://<host>:12345/         (UI)
- http://<host>:12345/stream   (video MJPEG con overlay)
- http://<host>:12345/status   (JSON con gesture+pinch)
- http://<host>:12345/health   (diagnostica)
- http://<host>:12345/snapshot.jpg (singolo frame)

MQTT (default):
- Base topic: gesture32
- availability: gesture32/availability (online/offline)
- state topics:
  - gesture32/state/gesture
  - gesture32/state/confidence
  - gesture32/state/hands
  - gesture32/state/pinch_distance_px
  - gesture32/state/pinch_distance_norm
  - gesture32/state/pinch_state

Avvio rapido:
  docker build -t http-gesture-mqtt:latest .
  SOURCE_URL="http://## Indirizzo IP ESP32 ##/stream" VERBOSE=1 ./RUN.sh
  # oppure:
  docker compose up -d

Configurazione (.env):
  cp .env.example .env
  # modifica i valori con le impostazioni locali
  # il server Flask, docker compose e RUN.sh caricano automaticamente questo file

Tuning pinch via ENV:
- PINCH_DEADZONE_PX (default 8)
- PINCH_HISTORY (default 8)

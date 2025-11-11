ESP32 Gesture + MediaPipe + MQTT — Pinch Distance (no Face Detection)
====================================================================

Funzioni:
- Gesture mani (estese) + MQTT autodiscovery
- Pinch (pollice-indice): distanza in px e normalizzata (0..1) + trend (opening/closing/steady)
- No face detection

Endpoints:
- http://<host>:12345/         (UI)
- http://<host>:12345/stream   (video MJPEG con overlay)
- http://<host>:12345/status   (JSON con gesture, pinch, stato video/MQTT e log se debug)
- http://<host>:12345/health   (diagnostica)
- http://<host>:12345/snapshot.jpg (singolo frame)
- http://<host>:12345/gestures  (GET/POST configurazione gesture MQTT)

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
  cp .env.example .env  # personalizza e poi rimuovi dal VCS
  ./RUN.sh
  # oppure:
  docker compose up -d

Tuning pinch via ENV:
- PINCH_DEADZONE_PX (default 8)
- PINCH_HISTORY (default 8)

Altre variabili utili:
- DEBUG_MODE / VERBOSE per attivare il pannello log in UI
- ACTIVE_GESTURES per limitare le gesture inviate via MQTT (usa "*" per tutte)
- LOG_HISTORY_SIZE per regolare il buffer di log esposto dal pannello debug

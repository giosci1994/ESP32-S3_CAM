import os, time, re, cv2, numpy as np, requests, logging, json, threading
from pathlib import Path
from flask import Flask, Response, render_template, jsonify, request
from collections import deque

# --- MediaPipe (hands only) ---
mp_ok = True
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles
except Exception as e:
    mp_ok = False

# --- Env / settings ---

def _load_env_from_file():
    """Populate os.environ with values from a .env file if present.

    The lookup order is: current working directory, repository root, server
    directory. Only missing variables are set so that explicit environment
    values (e.g. docker, shell) win.
    """

    env_name = os.getenv("ENV_FILE", ".env")
    candidate_dirs = [
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent,
        Path.cwd(),
    ]
    seen = set()
    for base in candidate_dirs:
        env_path = (base / env_name).resolve()
        if env_path in seen or not env_path.exists() or not env_path.is_file():
            continue
        seen.add(env_path)
        try:
            for raw_line in env_path.read_text().splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                    value = value[1:-1]
                os.environ.setdefault(key, value)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[env] Failed loading {env_path}: {exc}")


_load_env_from_file()

SOURCE_URL = os.getenv("SOURCE_URL", "http://esp32-s3.local/stream")
TARGET_FPS = float(os.getenv("TARGET_FPS", "25"))
FRAME_DELAY = 1.0 / TARGET_FPS
TARGET_SIZE = (800, 600)
VERBOSE = os.getenv("VERBOSE", "0").lower() in {"1", "true", "yes", "on"}
DEBUG_MODE = os.getenv("DEBUG_MODE", "0").lower() in {"1", "true", "yes", "on"} or VERBOSE
CUSTOM_GESTURE_MAP = json.loads(os.getenv("CUSTOM_GESTURE_MAP", "{}"))

# Pinch tuning
PINCH_DEADZONE_PX = int(os.getenv("PINCH_DEADZONE_PX", "8"))
PINCH_HISTORY = int(os.getenv("PINCH_HISTORY", "8"))

# MQTT
MQTT_HOST = os.getenv("MQTT_HOST", "mqtt.local")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER", "mqtt_user")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")
MQTT_BASE = os.getenv("MQTT_BASE_TOPIC", "gesture32")
DISCOVERY_PREFIX = os.getenv("MQTT_DISCOVERY_PREFIX", "homeassistant")
MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "gesture32-server")

logging.basicConfig(
    level=logging.DEBUG if VERBOSE or DEBUG_MODE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("gesture-server")

_LOG_HISTORY = deque(maxlen=int(os.getenv("LOG_HISTORY_SIZE", "200")))


class _MemoryLogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            _LOG_HISTORY.append({
                "ts": record.created,
                "level": record.levelname,
                "msg": msg,
            })
        except Exception:  # pragma: no cover - defensive
            pass


logging.getLogger().addHandler(_MemoryLogHandler())

app = Flask(__name__)
last_gesture = {"label":"unknown","confidence":0.0,"num_hands":0}
last_pinch   = {"distance_px":0.0,"distance_norm":0.0,"trend":"steady","delta_px":0.0}
last_error = ""
last_frame_ts = 0.0

AVAILABLE_GESTURES = {
    "unknown",
    "none",
    "open_palm",
    "fist",
    "peace",
    "rock",
    "ok",
    "point_left",
    "point_right",
    "point",
    "pinch",
    "one",
    "0_fingers",
    "1_fingers",
    "2_fingers",
    "3_fingers",
    "4_fingers",
}
AVAILABLE_GESTURES.update(CUSTOM_GESTURE_MAP.values())

def _parse_active_gestures(raw: str):
    if not raw:
        return set()
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return {item for item in items if item in AVAILABLE_GESTURES or item == "*"}


_DEFAULT_ACTIVE = _parse_active_gestures(os.getenv("ACTIVE_GESTURES", ""))
ACTIVE_GESTURES = set(AVAILABLE_GESTURES if "*" in _DEFAULT_ACTIVE else _DEFAULT_ACTIVE)
if not ACTIVE_GESTURES:
    ACTIVE_GESTURES = set(AVAILABLE_GESTURES)

# ------------- MQTT: Home Assistant Discovery -------------
mqtt_client = None
mqtt_connected = False
mqtt_last_error = ""
def mqtt_connect_and_discover():
    global mqtt_client, mqtt_connected, mqtt_last_error
    try:
        import paho.mqtt.client as mqtt
    except Exception as e:
        log.warning(f"MQTT not available: {e}")
        return

    def on_connect(client, userdata, flags, rc):
        global mqtt_connected, mqtt_last_error
        log.info(f"[MQTT] Connected (rc={rc})")
        mqtt_connected = True
        mqtt_last_error = ""
        client.publish(f"{MQTT_BASE}/availability", "online", retain=True)
        device = {
            "identifiers": [MQTT_BASE],
            "name": "ESP32 Gesture Server",
            "manufacturer": "Giosci Lab",
            "model": "ESP32+MediaPipe",
        }
        sensors = [
            {
                "uniq": f"{MQTT_BASE}_gesture",
                "name": "ESP32 Gesture",
                "state_topic": f"{MQTT_BASE}/state/gesture",
                "icon": "mdi:hand-back-right",
            },
            {
                "uniq": f"{MQTT_BASE}_confidence",
                "name": "ESP32 Gesture Confidence",
                "state_topic": f"{MQTT_BASE}/state/confidence",
                "unit_of_measurement": "%",
                "state_class": "measurement",
                "suggested_display_precision": 2
            },
            {
                "uniq": f"{MQTT_BASE}_hands",
                "name": "ESP32 Hands Count",
                "state_topic": f"{MQTT_BASE}/state/hands",
                "state_class": "measurement"
            },
            # Pinch sensors
            {
                "uniq": f"{MQTT_BASE}_pinch_distance_px",
                "name": "ESP32 Pinch Distance (px)",
                "state_topic": f"{MQTT_BASE}/state/pinch_distance_px",
                "unit_of_measurement": "px",
                "state_class": "measurement",
                "suggested_display_precision": 1
            },
            {
                "uniq": f"{MQTT_BASE}_pinch_distance_norm",
                "name": "ESP32 Pinch Distance (norm)",
                "state_topic": f"{MQTT_BASE}/state/pinch_distance_norm",
                "state_class": "measurement",
                "suggested_display_precision": 4
            },
            {
                "uniq": f"{MQTT_BASE}_pinch_state",
                "name": "ESP32 Pinch State",
                "state_topic": f"{MQTT_BASE}/state/pinch_state",
                "icon": "mdi:gesture",
            }
        ]
        for s in sensors:
            obj = {
                "name": s["name"],
                "state_topic": s["state_topic"],
                "unique_id": s["uniq"],
                "availability_topic": f"{MQTT_BASE}/availability",
                "device": device,
            }
            if "unit_of_measurement" in s: obj["unit_of_measurement"] = s["unit_of_measurement"]
            if "state_class" in s: obj["state_class"] = s["state_class"]
            if "icon" in s: obj["icon"] = s["icon"]
            if "suggested_display_precision" in s: obj["suggested_display_precision"] = s["suggested_display_precision"]
            disc_topic = f"{DISCOVERY_PREFIX}/sensor/{s['uniq']}/config"
            client.publish(disc_topic, json.dumps(obj), retain=True)
            log.info(f"[MQTT] Discovery published: {disc_topic}")

    def on_disconnect(client, userdata, rc):
        global mqtt_connected, mqtt_last_error
        log.warning(f"[MQTT] Disconnected (rc={rc})")
        mqtt_connected = False
        mqtt_last_error = f"disconnect_rc_{rc}"

    import paho.mqtt.client as mqtt
    mqtt_client = mqtt.Client(client_id=MQTT_CLIENT_ID, clean_session=True)
    mqtt_client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
    mqtt_client.will_set(f"{MQTT_BASE}/availability", "offline", retain=True)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_disconnect = on_disconnect

    try:
        mqtt_client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
        threading.Thread(target=mqtt_client.loop_forever, daemon=True).start()
    except Exception as e:
        mqtt_last_error = f"connect_failed: {e}"
        log.warning(f"[MQTT] Connect failed: {e}")
        mqtt_client = None
        mqtt_connected = False

def mqtt_publish_state():
    if mqtt_client is None:
        return
    try:
        current_label = last_gesture.get("label", "unknown")
        publish_label = current_label
        publish_conf = last_gesture.get('confidence', 0.0)
        publish_hands = last_gesture.get('num_hands', 0)
        if ACTIVE_GESTURES and current_label not in ACTIVE_GESTURES and current_label not in {"unknown", "none"}:
            publish_label = "inactive"
            publish_conf = 0.0
            publish_hands = 0
        mqtt_client.publish(f"{MQTT_BASE}/state/gesture", publish_label, qos=0, retain=False)
        mqtt_client.publish(f"{MQTT_BASE}/state/confidence", f"{publish_conf*100:.2f}", qos=0, retain=False)
        mqtt_client.publish(f"{MQTT_BASE}/state/hands", str(publish_hands), qos=0, retain=False)
        # Pinch
        mqtt_client.publish(f"{MQTT_BASE}/state/pinch_distance_px", f"{last_pinch.get('distance_px',0.0):.1f}", qos=0, retain=False)
        mqtt_client.publish(f"{MQTT_BASE}/state/pinch_distance_norm", f"{last_pinch.get('distance_norm',0.0):.4f}", qos=0, retain=False)
        mqtt_client.publish(f"{MQTT_BASE}/state/pinch_state", last_pinch.get("trend","steady"), qos=0, retain=False)
    except Exception as e:
        log.error(f"[MQTT] publish error: {e}", exc_info=True)

# ------------- Video readers -------------
def _mjpeg_http_reader(url, timeout=8, chunk=4096):
    headers = {"User-Agent": "Mozilla/5.0 (ESP32-Gesture-Server)"}
    with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        m = re.search(r'boundary=([^;]+)', ctype, re.IGNORECASE)
        boundary = m.group(1).strip() if m else None
        if boundary and not boundary.startswith("--"):
            boundary = "--" + boundary

        buf = bytearray()
        soi = b"\xff\xd8"; eoi = b"\xff\xd9"
        for chunk_bytes in r.iter_content(chunk_size=chunk):
            if not chunk_bytes: continue
            buf.extend(chunk_bytes)

            if boundary and boundary.encode() in buf:
                parts = buf.split(boundary.encode())
                buf = bytearray(parts[-1])
                for part in parts[:-1]:
                    hdr_end = part.find(b"\r\n\r\n")
                    if hdr_end != -1:
                        jpg = part[hdr_end+4:]
                        if len(jpg) > 2 and jpg.startswith(soi) and jpg.endswith(eoi):
                            yield bytes(jpg)
                continue

            while True:
                i = buf.find(soi)
                j = buf.find(eoi, i+2)
                if i != -1 and j != -1:
                    frame = bytes(buf[i:j+2])
                    del buf[:j+2]
                    yield frame
                else:
                    break

def _frames():
    url = SOURCE_URL
    while True:
        try:
            for jpg in _mjpeg_http_reader(url, timeout=8):
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                yield frame
        except Exception as e:
            global last_error
            last_error = f"HTTP reader: {type(e).__name__}: {e}"
            log.error("Stream failure: %s", last_error, exc_info=True)
            time.sleep(0.6)

# --- Gesture helpers ---
def _finger_up(pts, tip, pip, delta=0.02):
    return pts[tip,1] < pts[pip,1] - delta

def _classify_gesture(landmarks, handedness_label):
    import numpy as np
    pts = np.array([(lm.x, lm.y) for lm in landmarks.landmark], dtype=np.float32)
    WRIST=0; THUMB_TIP=4; THUMB_IP=3; INDEX_TIP=8; INDEX_PIP=6; MIDDLE_TIP=12; MIDDLE_PIP=10; RING_TIP=16; RING_PIP=14; PINKY_TIP=20; PINKY_PIP=18; INDEX_MCP=5

    def dist(a, b): return float(np.linalg.norm(pts[a]-pts[b]))
    def horiz_dir(a, b): return pts[b,0] - pts[a,0]

    fingers = {
        "thumb": (pts[THUMB_TIP,0] < pts[THUMB_IP,0]-0.02) if handedness_label=="Right"
                 else (pts[THUMB_TIP,0] > pts[THUMB_IP,0]+0.02),
        "index": _finger_up(pts, INDEX_TIP, INDEX_PIP, 0.02),
        "middle": _finger_up(pts, MIDDLE_TIP, MIDDLE_PIP, 0.02),
        "ring": _finger_up(pts, RING_TIP, RING_PIP, 0.02),
        "pinky": _finger_up(pts, PINKY_TIP, PINKY_PIP, 0.02),
    }
    up_count = sum(1 for v in fingers.values() if v)
    d_thumb_index = dist(THUMB_TIP, INDEX_TIP)

    if up_count == 5: return "open_palm", 0.95
    if up_count == 0: return "fist", 0.95
    if fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
        return "peace", 0.9
    if fingers["index"] and not fingers["middle"] and not fingers["ring"] and fingers["pinky"]:
        return "rock", 0.85
    if d_thumb_index < 0.06:
        return "ok", 0.85
    if fingers["index"] and not any([fingers["middle"], fingers["ring"], fingers["pinky"]]):
        dx = horiz_dir(WRIST, INDEX_MCP)
        if dx < -0.015: return "point_left", 0.8
        if dx >  0.015: return "point_right", 0.8
        return "point", 0.75
    if d_thumb_index < 0.04 and not fingers["index"]:
        return "pinch", 0.8
    if fingers["index"] and up_count == 1:
        return "one", 0.8
    return f"{up_count}_fingers", 0.6

# --- Pinch calculation ---
_pinch_hist = deque(maxlen=PINCH_HISTORY)

def _pinch_distance(lm, w, h):
    t = lm.landmark[4]   # THUMB_TIP
    i = lm.landmark[8]   # INDEX_TIP
    dx = (t.x - i.x) * w
    dy = (t.y - i.y) * h
    dist_px = (dx*dx + dy*dy) ** 0.5
    dist_norm = ((t.x - i.x)**2 + (t.y - i.y)**2) ** 0.5
    return dist_px, dist_norm

def _pinch_trend(new_px):
    _pinch_hist.append(new_px)
    if len(_pinch_hist) < 4:
        return "steady", 0.0
    delta = _pinch_hist[-1] - _pinch_hist[0]
    if abs(delta) < PINCH_DEADZONE_PX:
        return "steady", delta
    return ("opening" if delta > 0 else "closing"), delta

# ------------- Stream / processing -------------
def frame_generator():
    global last_gesture, last_pinch, last_error, last_frame_ts
    if not mp_ok:
        log.warning("MediaPipe not available, streaming raw frames.")

    ctx = mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=1,
                         min_detection_confidence=0.5, min_tracking_confidence=0.5) if mp_ok else None
    prev = time.time()
    last_pub = 0.0

    for frame in _frames():
        label, conf, nh = "none", 0.0, 0
        last_frame_ts = time.time()
        last_error = ""

        if ctx:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = ctx.process(rgb)

            if res.multi_hand_landmarks and res.multi_handedness:
                for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    g, c = _classify_gesture(lm, hd.classification[0].label)
                    if c > conf: label, conf = g, c
                    nh += 1
                    mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS,
                                              mp_styles.get_default_hand_landmarks_style(),
                                              mp_styles.get_default_hand_connections_style())

                # Pinch: usa la prima mano
                h, w = frame.shape[:2]
                lm0 = res.multi_hand_landmarks[0]
                pinch_px, pinch_norm = _pinch_distance(lm0, w, h)
                pinch_trend, pinch_delta = _pinch_trend(pinch_px)
                last_pinch = {
                    "distance_px": float(pinch_px),
                    "distance_norm": float(pinch_norm),
                    "trend": pinch_trend,
                    "delta_px": float(pinch_delta)
                }

        mapped_label = CUSTOM_GESTURE_MAP.get(label, label)
        last_gesture = {"label": mapped_label, "confidence": float(conf), "num_hands": int(nh)}

        # MQTT publish (throttled)
        nowt = time.time()
        if nowt - last_pub > 0.4:
            mqtt_publish_state()
            last_pub = nowt

        frame = cv2.resize(frame, (800, 600), interpolation=cv2.INTER_AREA)
        # Overlay info
        cv2.rectangle(frame, (10,10), (560,72), (0,0,0), -1)
        txt1 = f"Gesture: {last_gesture['label']}  Conf: {last_gesture['confidence']:.2f}  Hands: {last_gesture['num_hands']}"
        cv2.putText(frame, txt1, (16,36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        txt2 = f"Pinch: {last_pinch['distance_px']:.1f}px  {last_pinch['trend']} ({last_pinch['delta_px']:+.0f}px)"
        cv2.putText(frame, txt2, (16,64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120,220,255), 2, cv2.LINE_AA)

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue

        now = time.time(); dt = now - prev
        if dt < FRAME_DELAY: time.sleep(FRAME_DELAY - dt)
        prev = time.time()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

# ------------- Flask endpoints -------------
@app.route("/")
def index(): return render_template("index.html")

@app.route("/stream")
def stream(): return Response(frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/status")
def status():
    now = time.time()
    video_ok = bool(last_frame_ts and (now - last_frame_ts) < max(2.0, FRAME_DELAY * 4))
    payload = {
        "gesture": last_gesture,
        "pinch": last_pinch,
        "video": {
            "ok": video_ok,
            "last_frame_ts": last_frame_ts,
            "source": SOURCE_URL,
        },
        "mqtt": {
            "connected": mqtt_connected,
            "last_error": mqtt_last_error,
            "host": MQTT_HOST,
            "port": MQTT_PORT,
        },
        "debug": DEBUG_MODE or VERBOSE,
        "last_error": last_error,
        "active_gestures": sorted(ACTIVE_GESTURES),
    }
    if DEBUG_MODE or VERBOSE:
        payload["logs"] = list(_LOG_HISTORY)
    return jsonify(payload)


@app.route("/gestures", methods=["GET", "POST"])
def gestures_config():
    global ACTIVE_GESTURES
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        active = data.get("active", [])
        if not isinstance(active, list):
            log.warning("Invalid gestures payload: %s", data)
            return jsonify({"ok": False, "error": "Invalid payload"}), 400
        filtered = [g for g in active if g in AVAILABLE_GESTURES]
        if not filtered:
            log.warning("Attempt to set empty or invalid gestures: %s", active)
            return jsonify({"ok": False, "error": "No valid gestures"}), 400
        ACTIVE_GESTURES = set(filtered)
        log.info("Active gestures updated: %s", sorted(ACTIVE_GESTURES))
        return jsonify({"ok": True, "active": sorted(ACTIVE_GESTURES)})
    return jsonify({
        "available": sorted(AVAILABLE_GESTURES),
        "active": sorted(ACTIVE_GESTURES),
    })

@app.route("/health")
def health():
    return jsonify({"ok": True, "source": SOURCE_URL, "last_error": last_error})

def _one_frame():
    for frame in _frames():
        return frame
    return None

@app.route("/snapshot.jpg")
def snapshot():
    frame = _one_frame()
    if frame is None:
        return ("", 503)
    ok, buf = cv2.imencode(".jpg", cv2.resize(frame, (800,600)), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok: return ("", 503)
    return Response(buf.tobytes(), mimetype="image/jpeg")

# ------------- App boot -------------
def _boot():
    logging.getLogger().info(f"Boot with SOURCE_URL={SOURCE_URL}")
    mqtt_connect_and_discover()

_boot()

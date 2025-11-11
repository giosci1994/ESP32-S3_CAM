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
FRAME_DELAY = 1.0 / max(TARGET_FPS, 1.0)
TARGET_SIZE = (800, 600)
VERBOSE = os.getenv("VERBOSE", "0").lower() in {"1", "true", "yes", "on"}
DEBUG_MODE = os.getenv("DEBUG_MODE", "0").lower() in {"1", "true", "yes", "on"}
if VERBOSE:
    DEBUG_MODE = True
DEBUG_LOCKED = VERBOSE
# Confidence filtering
def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

try:
    _conf_threshold_env = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
except (TypeError, ValueError):
    _conf_threshold_env = 0.7
CONFIDENCE_THRESHOLD = _clamp(_conf_threshold_env, 0.0, 1.0)

# Gesture catalog
GESTURE_CHOICES = [
    "Mano aperta",
    "V di vittoria",
    "OK",
    "Angolo in alto a sinistra",
    "Angolo in alto a destra",
    "Angolo in basso a sinistra",
    "Angolo in basso a destra",
]

DEFAULT_GESTURE_MAP = {
    "open_palm": "Mano aperta",
    "victory": "V di vittoria",
    "ok": "OK",
    "point_top_left": "Angolo in alto a sinistra",
    "point_top_right": "Angolo in alto a destra",
    "point_bottom_left": "Angolo in basso a sinistra",
    "point_bottom_right": "Angolo in basso a destra",
}

try:
    _custom_map_env = json.loads(os.getenv("CUSTOM_GESTURE_MAP", "{}"))
    if not isinstance(_custom_map_env, dict):
        _custom_map_env = {}
except json.JSONDecodeError:
    _custom_map_env = {}

CUSTOM_GESTURE_MAP = DEFAULT_GESTURE_MAP.copy()
CUSTOM_GESTURE_MAP.update({k: v for k, v in _custom_map_env.items() if isinstance(k, str) and isinstance(v, str)})

# Pinch tuning
PINCH_DEADZONE_PX = int(os.getenv("PINCH_DEADZONE_PX", "8"))
PINCH_HISTORY = int(os.getenv("PINCH_HISTORY", "8"))

# MQTT
MQTT_HOST = os.getenv("MQTT_HOST", "mqtt.local")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER", "mqtt_user")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD", "")
MQTT_BASE = os.getenv("MQTT_BASE_TOPIC", "gesture32")
_MQTT_BASE_INITIAL = MQTT_BASE
DISCOVERY_PREFIX = os.getenv("MQTT_DISCOVERY_PREFIX", "homeassistant")
MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "gesture32-server")

logging.basicConfig(
    level=logging.DEBUG if VERBOSE or DEBUG_MODE else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("gesture-server")

_LOG_HISTORY = deque(maxlen=int(os.getenv("LOG_HISTORY_SIZE", "200")))

_FRAME_SIZE_CHOICES = {
    "320x240": (320, 240),
    "640x480": (640, 480),
    "800x600": (800, 600),
}


def _advanced_defaults():
    return {
        "confidence_min": CONFIDENCE_THRESHOLD,
        "movement_sensitivity_px": 8,
        "temporal_smoothing": 5,
        "hold_delay_ms": 700,
        "pinch_threshold_norm": 0.05,
        "pinch_stability_px": 12,
        "pinch_confirm_ms": 1000,
        "corner_area_pct": 25,
        "processing_fps": int(round(TARGET_FPS)),
        "frame_size": "{}x{}".format(*TARGET_SIZE),
        "brightness_contrast": 0,
        "auto_exposure": True,
        "mqtt_publish_interval": 400,
        "float_precision": 2,
        "mqtt_base_topic": _MQTT_BASE_INITIAL,
        "show_landmarks": True,
        "visual_feedback": True,
    }


ADVANCED_SETTINGS = _advanced_defaults()


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


def _apply_logging_level():
    logging.getLogger().setLevel(logging.DEBUG if (VERBOSE or DEBUG_MODE) else logging.INFO)


_apply_logging_level()

app = Flask(__name__)
last_gesture = {"label":"unknown","confidence":0.0,"num_hands":0}
last_pinch   = {"distance_px":0.0,"distance_norm":0.0,"trend":"steady","delta_px":0.0}
last_error = ""
last_frame_ts = 0.0

AVAILABLE_GESTURES = list(GESTURE_CHOICES)
for mapped in CUSTOM_GESTURE_MAP.values():
    if mapped not in AVAILABLE_GESTURES:
        AVAILABLE_GESTURES.append(mapped)
AVAILABLE_GESTURES_SET = set(AVAILABLE_GESTURES)
AVAILABLE_GESTURES_SET.update({"unknown", "none"})

def _parse_active_gestures(raw: str):
    if not raw:
        return set()
    items = [item.strip() for item in raw.split(",") if item.strip()]
    parsed = set()
    for item in items:
        if item == "*":
            parsed.add(item)
            continue
        if item in AVAILABLE_GESTURES_SET:
            parsed.add(item)
            continue
        mapped = CUSTOM_GESTURE_MAP.get(item)
        if mapped and mapped in AVAILABLE_GESTURES_SET:
            parsed.add(mapped)
    return parsed


_DEFAULT_ACTIVE = _parse_active_gestures(os.getenv("ACTIVE_GESTURES", ""))
if "*" in _DEFAULT_ACTIVE:
    ACTIVE_GESTURES = set(AVAILABLE_GESTURES)
else:
    ACTIVE_GESTURES = set(_DEFAULT_ACTIVE)
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
        if publish_conf < CONFIDENCE_THRESHOLD:
            publish_label = "none"
            publish_conf = 0.0
            publish_hands = 0
        elif ACTIVE_GESTURES and publish_label not in {"unknown", "none"} and publish_label not in ACTIVE_GESTURES:
            publish_label = "inactive"
            publish_conf = 0.0
            publish_hands = 0
        prec = int(ADVANCED_SETTINGS.get("float_precision", 2))
        prec = min(4, max(1, prec))
        fmt = "{:." + str(prec) + "f}"
        fmt_norm = "{:." + str(max(prec, 3)) + "f}"
        mqtt_client.publish(f"{MQTT_BASE}/state/gesture", publish_label, qos=0, retain=False)
        mqtt_client.publish(f"{MQTT_BASE}/state/confidence", fmt.format(publish_conf * 100.0), qos=0, retain=False)
        mqtt_client.publish(f"{MQTT_BASE}/state/hands", str(publish_hands), qos=0, retain=False)
        # Pinch
        mqtt_client.publish(
            f"{MQTT_BASE}/state/pinch_distance_px",
            fmt.format(last_pinch.get('distance_px', 0.0)),
            qos=0,
            retain=False,
        )
        mqtt_client.publish(
            f"{MQTT_BASE}/state/pinch_distance_norm",
            fmt_norm.format(last_pinch.get('distance_norm', 0.0)),
            qos=0,
            retain=False,
        )
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

    # Mano aperta: tutte le dita sollevate e ben distanziate
    if all(fingers.values()):
        span = dist(INDEX_TIP, PINKY_TIP)
        conf = 0.92 + min(0.06, span) * 0.5
        return "open_palm", min(conf, 0.99)

    # V di vittoria: indice e medio sollevati e separati, altre dita chiuse
    if fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
        separation = dist(INDEX_TIP, MIDDLE_TIP)
        if separation > 0.045:
            conf = 0.88 + min(0.07, separation - 0.045) * 1.5
            return "victory", min(conf, 0.99)

    # OK: pollice e indice a contatto, indice piegato, altre dita prevalentemente sollevate
    index_bent = pts[INDEX_TIP,1] > pts[INDEX_PIP,1] - 0.005
    support_fingers = sum(1 for name in ("middle", "ring", "pinky") if fingers[name])
    if d_thumb_index < 0.05 and index_bent and support_fingers >= 2:
        conf = 0.86 + (0.05 - d_thumb_index) * 2.2
        return "ok", min(conf, 0.99)

    # Indicazione angoli: solo indice sollevato, direzione diagonale ben definita
    if fingers["index"] and not any(fingers[name] for name in ("middle", "ring", "pinky")):
        idx_vec = pts[INDEX_TIP] - pts[INDEX_MCP]
        dx, dy = float(idx_vec[0]), float(idx_vec[1])
        mag = (dx * dx + dy * dy) ** 0.5
        dir_thr = 0.04
        if mag > 0.1:
            horiz = None
            vert = None
            if dx <= -dir_thr:
                horiz = "left"
            elif dx >= dir_thr:
                horiz = "right"
            if dy <= -dir_thr:
                vert = "top"
            elif dy >= dir_thr:
                vert = "bottom"
            if horiz and vert:
                conf = 0.82 + min(0.08, mag - 0.1) * 1.2
                return f"point_{vert}_{horiz}", min(conf, 0.96)

    return "unknown", 0.3

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
        settings_snapshot = dict(ADVANCED_SETTINGS)
        label, conf, nh = "none", 0.0, 0
        last_frame_ts = time.time()
        last_error = ""

        target_size = _FRAME_SIZE_CHOICES.get(settings_snapshot.get("frame_size"), TARGET_SIZE)
        if target_size != frame.shape[1::-1]:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

        bc_val = float(settings_snapshot.get("brightness_contrast", 0) or 0)
        if abs(bc_val) > 0.01:
            alpha = 1.0 + (bc_val / 100.0)
            beta = bc_val
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        if ctx:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = ctx.process(rgb)

            if res.multi_hand_landmarks and res.multi_handedness:
                for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    g, c = _classify_gesture(lm, hd.classification[0].label)
                    if c > conf: label, conf = g, c
                    nh += 1
                    if settings_snapshot.get("show_landmarks", True):
                        mp_drawing.draw_landmarks(
                            frame,
                            lm,
                            mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style(),
                        )

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
        interval = max(0.05, float(settings_snapshot.get("mqtt_publish_interval", 400)) / 1000.0)
        if nowt - last_pub > interval:
            mqtt_publish_state()
            last_pub = nowt

        if frame.shape[1::-1] != target_size:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        # Overlay info
        info_box_w = max(220, target_size[0] - 40)
        cv2.rectangle(frame, (10,10), (10 + info_box_w, 72), (0,0,0), -1)
        display_prec = int(settings_snapshot.get("float_precision", 2))
        display_prec = min(4, max(1, display_prec))
        conf_txt = f"{last_gesture['confidence']:.{display_prec}f}"
        txt1 = f"Gesture: {last_gesture['label']}  Conf: {conf_txt}  Hands: {last_gesture['num_hands']}"
        cv2.putText(frame, txt1, (16,36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        pinch_px = last_pinch.get('distance_px', 0.0)
        pinch_delta = last_pinch.get('delta_px', 0.0)
        pinch_txt = f"{pinch_px:.{display_prec}f}"
        txt2 = f"Pinch: {pinch_txt}px  {last_pinch['trend']} ({pinch_delta:+.0f}px)"
        cv2.putText(frame, txt2, (16,64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120,220,255), 2, cv2.LINE_AA)

        if settings_snapshot.get("visual_feedback", True):
            color = (60, 220, 120) if last_gesture['confidence'] >= CONFIDENCE_THRESHOLD else (70, 70, 220)
            center = (target_size[0] - 60, target_size[1] - 60)
            cv2.circle(frame, center, 32, color, 4)

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
        "debug_locked": DEBUG_LOCKED,
        "last_error": last_error,
        "active_gestures": sorted(ACTIVE_GESTURES),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "advanced": dict(ADVANCED_SETTINGS),
    }
    if DEBUG_MODE or VERBOSE:
        payload["logs"] = list(_LOG_HISTORY)
    return jsonify(payload)


@app.route("/gestures", methods=["GET", "POST"])
def gestures_config():
    global ACTIVE_GESTURES, CONFIDENCE_THRESHOLD
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        active = data.get("active", [])
        if not isinstance(active, list):
            log.warning("Invalid gestures payload: %s", data)
            return jsonify({"ok": False, "error": "Invalid payload"}), 400
        filtered = [g for g in active if g in AVAILABLE_GESTURES_SET]
        if not filtered:
            log.warning("Attempt to set empty or invalid gestures: %s", active)
            return jsonify({"ok": False, "error": "No valid gestures"}), 400
        threshold = data.get("confidence_threshold", CONFIDENCE_THRESHOLD)
        try:
            threshold = float(threshold)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "Confidence threshold non valido"}), 400
        threshold = _clamp(threshold, 0.0, 1.0)
        ACTIVE_GESTURES = set(filtered)
        CONFIDENCE_THRESHOLD = threshold
        ADVANCED_SETTINGS["confidence_min"] = CONFIDENCE_THRESHOLD
        log.info("Active gestures updated: %s", sorted(ACTIVE_GESTURES))
        log.info("Confidence threshold set to %.2f", CONFIDENCE_THRESHOLD)
        return jsonify({
            "ok": True,
            "active": sorted(ACTIVE_GESTURES),
            "confidence_threshold": CONFIDENCE_THRESHOLD,
        })
    return jsonify({
        "available": AVAILABLE_GESTURES,
        "active": sorted(ACTIVE_GESTURES),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    })


@app.route("/debug", methods=["POST"])
def debug_config():
    global DEBUG_MODE
    data = request.get_json(silent=True) or {}
    if DEBUG_LOCKED:
        return jsonify({
            "ok": False,
            "error": "Debug bloccato dalla configurazione",  # UI handles message
            "debug": True,
            "locked": True,
        }), 400
    enabled = bool(data.get("enabled"))
    DEBUG_MODE = enabled
    _apply_logging_level()
    return jsonify({
        "ok": True,
        "debug": DEBUG_MODE or VERBOSE,
        "locked": DEBUG_LOCKED,
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
    size = _FRAME_SIZE_CHOICES.get(ADVANCED_SETTINGS.get("frame_size"), TARGET_SIZE)
    ok, buf = cv2.imencode(".jpg", cv2.resize(frame, size), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok: return ("", 503)
    return Response(buf.tobytes(), mimetype="image/jpeg")

# ------------- App boot -------------
def _boot():
    logging.getLogger().info(f"Boot with SOURCE_URL={SOURCE_URL}")
    mqtt_connect_and_discover()

_boot()
@app.route("/advanced-settings", methods=["GET", "POST"])
def advanced_settings():
    global ADVANCED_SETTINGS, CONFIDENCE_THRESHOLD, TARGET_FPS, FRAME_DELAY, TARGET_SIZE, MQTT_BASE
    if request.method == "GET":
        return jsonify({"settings": ADVANCED_SETTINGS})

    data = request.get_json(silent=True) or {}
    if data.get("action") == "reset":
        ADVANCED_SETTINGS = _advanced_defaults()
        CONFIDENCE_THRESHOLD = ADVANCED_SETTINGS["confidence_min"]
        TARGET_FPS = float(ADVANCED_SETTINGS["processing_fps"])
        FRAME_DELAY = 1.0 / max(TARGET_FPS, 1.0)
        TARGET_SIZE = _FRAME_SIZE_CHOICES.get(ADVANCED_SETTINGS["frame_size"], TARGET_SIZE)
        MQTT_BASE = ADVANCED_SETTINGS["mqtt_base_topic"] or MQTT_BASE
        log.info("Advanced settings reset to defaults")
        return jsonify({"ok": True, "settings": ADVANCED_SETTINGS})

    errors = {}
    updated = {}

    def _parse_float(name, low, high, default):
        raw = data.get(name, default)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors[name] = "Valore non valido"
            return
        updated[name] = _clamp(value, low, high)

    def _parse_int(name, low, high, default):
        raw = data.get(name, default)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            errors[name] = "Valore non valido"
            return
        updated[name] = max(low, min(high, value))

    _parse_float("confidence_min", 0.5, 0.95, ADVANCED_SETTINGS["confidence_min"])
    _parse_int("movement_sensitivity_px", 4, 30, ADVANCED_SETTINGS["movement_sensitivity_px"])
    _parse_int("temporal_smoothing", 1, 10, ADVANCED_SETTINGS["temporal_smoothing"])
    _parse_int("hold_delay_ms", 200, 1500, ADVANCED_SETTINGS["hold_delay_ms"])
    _parse_float("pinch_threshold_norm", 0.03, 0.08, ADVANCED_SETTINGS["pinch_threshold_norm"])
    _parse_int("pinch_stability_px", 4, 20, ADVANCED_SETTINGS["pinch_stability_px"])
    _parse_int("pinch_confirm_ms", 300, 2000, ADVANCED_SETTINGS["pinch_confirm_ms"])
    _parse_int("corner_area_pct", 10, 50, ADVANCED_SETTINGS["corner_area_pct"])
    _parse_int("processing_fps", 10, 30, ADVANCED_SETTINGS["processing_fps"])
    _parse_int("mqtt_publish_interval", 200, 1000, ADVANCED_SETTINGS["mqtt_publish_interval"])
    _parse_int("float_precision", 1, 4, ADVANCED_SETTINGS["float_precision"])

    frame_size = data.get("frame_size", ADVANCED_SETTINGS["frame_size"])
    if frame_size not in _FRAME_SIZE_CHOICES:
        errors["frame_size"] = "Valore non supportato"
    else:
        updated["frame_size"] = frame_size

    bc_raw = data.get("brightness_contrast", ADVANCED_SETTINGS["brightness_contrast"])
    try:
        bc_value = float(bc_raw)
    except (TypeError, ValueError):
        errors["brightness_contrast"] = "Valore non valido"
    else:
        updated["brightness_contrast"] = max(-50.0, min(50.0, bc_value))

    for name in ("auto_exposure", "show_landmarks", "visual_feedback"):
        updated[name] = bool(data.get(name, ADVANCED_SETTINGS[name]))

    mqtt_base_topic = data.get("mqtt_base_topic", ADVANCED_SETTINGS["mqtt_base_topic"]) or ADVANCED_SETTINGS["mqtt_base_topic"]
    if isinstance(mqtt_base_topic, str):
        mqtt_base_topic = mqtt_base_topic.strip()
    else:
        errors["mqtt_base_topic"] = "Valore non valido"
    if not errors.get("mqtt_base_topic"):
        if mqtt_base_topic:
            updated["mqtt_base_topic"] = mqtt_base_topic
        else:
            errors["mqtt_base_topic"] = "Valore obbligatorio"

    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    ADVANCED_SETTINGS.update(updated)
    CONFIDENCE_THRESHOLD = ADVANCED_SETTINGS["confidence_min"]
    TARGET_FPS = float(ADVANCED_SETTINGS["processing_fps"])
    FRAME_DELAY = 1.0 / max(TARGET_FPS, 1.0)
    TARGET_SIZE = _FRAME_SIZE_CHOICES.get(ADVANCED_SETTINGS["frame_size"], TARGET_SIZE)
    MQTT_BASE = ADVANCED_SETTINGS["mqtt_base_topic"]
    log.info("Advanced settings updated: %s", json.dumps(ADVANCED_SETTINGS))
    return jsonify({"ok": True, "settings": ADVANCED_SETTINGS})

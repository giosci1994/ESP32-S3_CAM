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
    "Swipe verso sinistra",
    "Swipe verso destra",
]

DEFAULT_GESTURE_MAP = {
    "open_palm": "Mano aperta",
    "victory": "V di vittoria",
    "ok": "OK",
    "point_top_left": "Angolo in alto a sinistra",
    "point_top_right": "Angolo in alto a destra",
    "point_bottom_left": "Angolo in basso a sinistra",
    "point_bottom_right": "Angolo in basso a destra",
    "swipe_left": "Swipe verso sinistra",
    "swipe_right": "Swipe verso destra",
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

_STATE_PATH = Path(__file__).resolve().parent / "state.json"
_STATE_LOCK = threading.Lock()

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
        "pinch_stability_px": 16,
        "pinch_confirm_ms": 1000,
        "corner_area_pct": 25,
        "pinch_corner_hold_s": 1.5,
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


def _persist_state():
    payload = {
        "active_gestures": sorted(ACTIVE_GESTURES),
        "advanced_settings": ADVANCED_SETTINGS,
    }
    tmp_path = _STATE_PATH.with_suffix(".tmp")
    try:
        with _STATE_LOCK:
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            tmp_path.replace(_STATE_PATH)
    except Exception as exc:  # pragma: no cover - I/O failure should not crash
        log.warning("Failed to persist state: %s", exc)


def _coerce_advanced_updates(data, *, strict=False):
    updates = {}
    errors = {}

    float_fields = {
        "confidence_min": (0.5, 0.95),
        "pinch_threshold_norm": (0.03, 0.08),
        "pinch_corner_hold_s": (0.5, 3.0),
        "brightness_contrast": (-50.0, 50.0),
    }
    int_fields = {
        "movement_sensitivity_px": (4, 30),
        "temporal_smoothing": (1, 10),
        "hold_delay_ms": (200, 1500),
        "pinch_stability_px": (4, 20),
        "pinch_confirm_ms": (300, 2000),
        "corner_area_pct": (10, 50),
        "processing_fps": (10, 30),
        "mqtt_publish_interval": (200, 1000),
        "float_precision": (1, 4),
    }

    for name, bounds in float_fields.items():
        if name not in data:
            continue
        low, high = bounds
        raw = data.get(name)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            if strict:
                errors[name] = "Valore non valido"
            continue
        updates[name] = _clamp(value, low, high)

    for name, bounds in int_fields.items():
        if name not in data:
            continue
        low, high = bounds
        raw = data.get(name)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            if strict:
                errors[name] = "Valore non valido"
            continue
        updates[name] = max(low, min(high, value))

    frame_size = data.get("frame_size")
    if frame_size is not None:
        if frame_size in _FRAME_SIZE_CHOICES:
            updates["frame_size"] = frame_size
        elif strict:
            errors["frame_size"] = "Valore non supportato"

    for name in ("auto_exposure", "show_landmarks", "visual_feedback"):
        if name in data:
            updates[name] = bool(data.get(name))

    if "mqtt_base_topic" in data:
        topic = data.get("mqtt_base_topic")
        if isinstance(topic, str):
            topic = topic.strip()
            if topic:
                updates["mqtt_base_topic"] = topic
            elif strict:
                errors["mqtt_base_topic"] = "Valore obbligatorio"
        elif strict:
            errors["mqtt_base_topic"] = "Valore non valido"

    return updates, errors


def _set_advanced_settings(updates, *, persist=True):
    global ADVANCED_SETTINGS, CONFIDENCE_THRESHOLD, TARGET_FPS, FRAME_DELAY, TARGET_SIZE, MQTT_BASE
    if not updates:
        if persist:
            _persist_state()
        return False

    previous = dict(ADVANCED_SETTINGS)
    ADVANCED_SETTINGS.update(updates)
    CONFIDENCE_THRESHOLD = ADVANCED_SETTINGS["confidence_min"]
    TARGET_FPS = float(ADVANCED_SETTINGS["processing_fps"])
    FRAME_DELAY = 1.0 / max(TARGET_FPS, 1.0)
    TARGET_SIZE = _FRAME_SIZE_CHOICES.get(ADVANCED_SETTINGS["frame_size"], TARGET_SIZE)
    new_base = ADVANCED_SETTINGS.get("mqtt_base_topic") or MQTT_BASE
    base_changed = new_base != MQTT_BASE
    MQTT_BASE = new_base
    if base_changed and mqtt_client is not None:
        mqtt_publish_discovery()
    if persist:
        _persist_state()
    return previous != ADVANCED_SETTINGS


def _load_persisted_state():
    if not _STATE_PATH.exists() or not _STATE_PATH.is_file():
        return
    try:
        data = json.loads(_STATE_PATH.read_text())
    except Exception as exc:  # pragma: no cover - ignore corrupt files
        log.warning("Unable to read state file %s: %s", _STATE_PATH, exc)
        return

    gestures = data.get("active_gestures")
    if isinstance(gestures, (list, tuple)):
        filtered = [g for g in gestures if g in AVAILABLE_GESTURES_SET]
        if filtered:
            ACTIVE_GESTURES.clear()
            ACTIVE_GESTURES.update(filtered)

    adv = data.get("advanced_settings")
    if isinstance(adv, dict):
        updates, _ = _coerce_advanced_updates(adv, strict=False)
        _set_advanced_settings(updates, persist=False)
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
last_gesture = {"label": "unknown", "confidence": 0.0, "num_hands": 0}
last_pinch = {
    "distance_px": 0.0,
    "distance_norm": 0.0,
    "trend": None,
    "delta_px": 0.0,
    "mode": None,
    "mode_enabled": False,
    "mode_since": 0.0,
}
pinch_mode_state = {
    "active": None,
    "since": 0.0,
    "steady_since": None,
    "feedback_until": 0.0,
    "feedback_text": "",
}
_corner_hold_tracker = {
    "point_top_left": {"start": None},
    "point_top_right": {"start": None},
}
last_error = ""
last_frame_ts = 0.0

_frame_condition = threading.Condition()
_latest_frame_seq = 0
_latest_frame_jpeg = None
_processing_thread = None

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

_load_persisted_state()

# ------------- MQTT: Home Assistant Discovery -------------
mqtt_client = None
mqtt_connected = False
mqtt_last_error = ""


def mqtt_publish_discovery():
    if mqtt_client is None:
        return

    try:
        device = {
            "identifiers": [MQTT_BASE],
            "name": "ESP32 Gesture Server",
            "manufacturer": "Giosci Lab",
            "model": "ESP32+MediaPipe",
        }
        sensors = [
            {
                "component": "sensor",
                "uniq": f"{MQTT_BASE}_gesture",
                "name": "ESP32 Gesture",
                "state_topic": f"{MQTT_BASE}/state/gesture",
                "icon": "mdi:hand-back-right",
            },
            {
                "component": "sensor",
                "uniq": f"{MQTT_BASE}_confidence",
                "name": "ESP32 Gesture Confidence",
                "state_topic": f"{MQTT_BASE}/state/confidence",
                "unit_of_measurement": "%",
                "state_class": "measurement",
                "suggested_display_precision": 2,
            },
            {
                "component": "sensor",
                "uniq": f"{MQTT_BASE}_hands",
                "name": "ESP32 Hands Count",
                "state_topic": f"{MQTT_BASE}/state/hands",
                "state_class": "measurement",
            },
            {
                "component": "sensor",
                "uniq": f"{MQTT_BASE}_pinch_distance_px",
                "name": "ESP32 Pinch Distance (px)",
                "state_topic": f"{MQTT_BASE}/state/pinch_distance_px",
                "unit_of_measurement": "px",
                "state_class": "measurement",
                "suggested_display_precision": 1,
            },
            {
                "component": "sensor",
                "uniq": f"{MQTT_BASE}_pinch_distance_norm",
                "name": "ESP32 Pinch Distance (norm)",
                "state_topic": f"{MQTT_BASE}/state/pinch_distance_norm",
                "state_class": "measurement",
                "suggested_display_precision": 4,
            },
            {
                "component": "sensor",
                "uniq": f"{MQTT_BASE}_pinch_state",
                "name": "ESP32 Pinch State",
                "state_topic": f"{MQTT_BASE}/state/pinch_state",
                "icon": "mdi:gesture",
            },
            {
                "component": "sensor",
                "uniq": f"{MQTT_BASE}_pinch_mode",
                "name": "ESP32 Pinch Mode",
                "state_topic": f"{MQTT_BASE}/state/pinch_mode",
                "icon": "mdi:gesture-tap-hold",
            },
            {
                "component": "binary_sensor",
                "uniq": f"{MQTT_BASE}_pinch_mode_left",
                "name": "ESP32 Pinch Mode Left",
                "state_topic": f"{MQTT_BASE}/state/pinch_mode_left",
                "device_class": "occupancy",
                "payload_on": "ON",
                "payload_off": "OFF",
                "icon": "mdi:hand-back-left",
            },
            {
                "component": "binary_sensor",
                "uniq": f"{MQTT_BASE}_pinch_mode_right",
                "name": "ESP32 Pinch Mode Right",
                "state_topic": f"{MQTT_BASE}/state/pinch_mode_right",
                "device_class": "occupancy",
                "payload_on": "ON",
                "payload_off": "OFF",
                "icon": "mdi:hand-back-right",
            },
        ]

        mqtt_client.publish(f"{MQTT_BASE}/availability", "online", retain=True)

        for sensor in sensors:
            sensor_copy = dict(sensor)
            component = sensor_copy.pop("component")
            payload = {
                "name": sensor_copy.pop("name"),
                "state_topic": sensor_copy.pop("state_topic"),
                "unique_id": sensor_copy.pop("uniq"),
                "availability_topic": f"{MQTT_BASE}/availability",
                "device": device,
            }
            payload.update(sensor_copy)
            topic = f"{DISCOVERY_PREFIX}/{component}/{payload['unique_id']}/config"
            mqtt_client.publish(topic, json.dumps(payload), retain=True)
            log.info("[MQTT] Discovery published: %s", topic)
    except Exception as exc:
        log.warning("[MQTT] Discovery failed: %s", exc)


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
        mqtt_publish_discovery()

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
        pinch_state = last_pinch.get("trend")
        mqtt_client.publish(
            f"{MQTT_BASE}/state/pinch_state",
            pinch_state if pinch_state is not None else "none",
            qos=0,
            retain=False,
        )
        active_mode = pinch_mode_state.get("active")
        mqtt_client.publish(
            f"{MQTT_BASE}/state/pinch_mode",
            active_mode or "none",
            qos=0,
            retain=False,
        )
        mqtt_client.publish(
            f"{MQTT_BASE}/state/pinch_mode_left",
            "ON" if active_mode == "left" else "OFF",
            qos=0,
            retain=False,
        )
        mqtt_client.publish(
            f"{MQTT_BASE}/state/pinch_mode_right",
            "ON" if active_mode == "right" else "OFF",
            qos=0,
            retain=False,
        )
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
    WRIST=0; THUMB_TIP=4; THUMB_IP=3; INDEX_TIP=8; INDEX_PIP=6; MIDDLE_TIP=12; MIDDLE_PIP=10; RING_TIP=16; RING_PIP=14; PINKY_TIP=20; PINKY_PIP=18; INDEX_MCP=5; MIDDLE_MCP=9; RING_MCP=13; PINKY_MCP=17

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
    palm_center = tuple(np.mean(pts[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]], axis=0))
    meta = {
        "fingers": fingers,
        "is_open_palm": False,
        "palm_center": palm_center,
        "index_tip": tuple(pts[INDEX_TIP]),
    }

    # Mano aperta: tutte le dita sollevate e ben distanziate
    if all(fingers.values()):
        span = dist(INDEX_TIP, PINKY_TIP)
        conf = 0.92 + min(0.06, span) * 0.5
        meta["is_open_palm"] = True
        return "open_palm", min(conf, 0.99), meta

    # V di vittoria: indice e medio sollevati e separati, altre dita chiuse
    if fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
        separation = dist(INDEX_TIP, MIDDLE_TIP)
        if separation > 0.045:
            conf = 0.88 + min(0.07, separation - 0.045) * 1.5
            return "victory", min(conf, 0.99), meta

    # OK: pollice e indice a contatto, indice piegato, altre dita prevalentemente sollevate
    index_bent = pts[INDEX_TIP,1] > pts[INDEX_PIP,1] - 0.005
    support_fingers = sum(1 for name in ("middle", "ring", "pinky") if fingers[name])
    if d_thumb_index < 0.05 and index_bent and support_fingers >= 2:
        conf = 0.86 + (0.05 - d_thumb_index) * 2.2
        return "ok", min(conf, 0.99), meta

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
                return f"point_{vert}_{horiz}", min(conf, 0.96), meta

    return "unknown", 0.3, meta

# --- Pinch calculation & swipe detection ---
_pinch_hist = deque(maxlen=PINCH_HISTORY)
_swipe_state = {}


def _swipe_tracker(hand_label):
    key = hand_label or "Unknown"
    if key not in _swipe_state:
        _swipe_state[key] = {
            "samples": deque(maxlen=14),
            "last_direction": None,
            "last_time": 0.0,
        }
    return _swipe_state[key]


def _detect_swipe(hand_label, meta, frame_width, settings_snapshot, confidence):
    if not meta:
        return None
    min_conf = float(settings_snapshot.get("confidence_min", CONFIDENCE_THRESHOLD))
    if confidence < min_conf:
        tracker = _swipe_tracker(hand_label)
        tracker["samples"].clear()
        return None
    if not meta.get("is_open_palm"):
        tracker = _swipe_tracker(hand_label)
        tracker["samples"].clear()
        return None

    now = time.time()
    palm_center = meta.get("palm_center") or (0.5, 0.5)
    center_x = float(palm_center[0]) * frame_width
    tracker = _swipe_tracker(hand_label)
    samples = tracker["samples"]
    samples.append((now, center_x))
    window = 0.6
    while samples and now - samples[0][0] > window:
        samples.popleft()
    if len(samples) < 3:
        return None
    start_time, start_x = samples[0]
    end_time, end_x = samples[-1]
    duration = end_time - start_time
    if duration < 0.12 or duration > window:
        return None
    movement = end_x - start_x
    threshold = float(settings_snapshot.get("movement_sensitivity_px", 8) or 8)
    if abs(movement) < threshold:
        return None

    direction = "right" if movement > 0 else "left"
    if tracker["last_direction"] == direction and (now - tracker["last_time"]) < 0.5:
        return None

    tracker["last_direction"] = direction
    tracker["last_time"] = now
    samples.clear()
    samples.append((now, center_x))

    bonus = max(0.0, abs(movement) - threshold)
    conf = 0.82 + min(0.15, bonus / max(threshold, 1.0) * 0.15)
    conf = min(0.99, conf)
    return ("swipe_right" if direction == "right" else "swipe_left"), conf


def _store_frame_bytes(jpeg_bytes: bytes):
    global _latest_frame_seq, _latest_frame_jpeg
    if not jpeg_bytes:
        return
    with _frame_condition:
        _latest_frame_jpeg = jpeg_bytes
        _latest_frame_seq += 1
        _frame_condition.notify_all()


def _wait_for_frame(last_seq: int, timeout: float = 2.0):
    end_time = time.time() + timeout
    with _frame_condition:
        while True:
            if _latest_frame_jpeg is not None and _latest_frame_seq > max(last_seq, -1):
                chunk = (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + _latest_frame_jpeg
                    + b"\r\n"
                )
                return chunk, _latest_frame_seq
            remaining = end_time - time.time()
            if remaining <= 0:
                return None, last_seq
            _frame_condition.wait(remaining)


def _get_latest_jpeg():
    with _frame_condition:
        return _latest_frame_jpeg


def _pinch_distance(lm, w, h):
    t = lm.landmark[4]   # THUMB_TIP
    i = lm.landmark[8]   # INDEX_TIP
    dx = (t.x - i.x) * w
    dy = (t.y - i.y) * h
    dist_px = (dx*dx + dy*dy) ** 0.5
    dist_norm = ((t.x - i.x)**2 + (t.y - i.y)**2) ** 0.5
    return dist_px, dist_norm

def _pinch_trend(new_px, stability_px=None):
    if stability_px is None:
        stability_px = PINCH_DEADZONE_PX
    try:
        threshold = max(1.0, float(stability_px))
    except (TypeError, ValueError):
        threshold = PINCH_DEADZONE_PX

    _pinch_hist.append(new_px)
    if len(_pinch_hist) < 6:
        return "steady", 0.0

    half = len(_pinch_hist) // 2
    if half < 3:
        half = len(_pinch_hist) - 1
    history = list(_pinch_hist)
    start_avg = sum(history[:half]) / max(1, half)
    end_avg = sum(history[-half:]) / max(1, half)
    delta = end_avg - start_avg
    if abs(delta) < threshold:
        return "steady", delta
    return ("opening" if delta > 0 else "closing"), delta


def _localize_pinch_trend(trend):
    mapping = {
        "opening": "apertura",
        "closing": "chiusura",
        "steady": "stabile",
    }
    if trend is None:
        return "-"
    return mapping.get(trend, trend)


def _mode_label(mode):
    return {"left": "sinistra", "right": "destra"}.get(mode, "")


def _set_pinch_mode(mode, *, now=None, reason=None):
    if now is None:
        now = time.time()
    current = pinch_mode_state.get("active")
    if mode == current:
        return False
    pinch_mode_state["active"] = mode
    pinch_mode_state["since"] = now
    pinch_mode_state["steady_since"] = None
    message = ""
    if mode is None and current:
        label = _mode_label(current)
        if label:
            message = reason or f"Modalità pinch {label} disattivata"
    elif mode:
        label = _mode_label(mode)
        if label:
            message = reason or f"Modalità pinch {label} attivata"
    pinch_mode_state["feedback_text"] = message
    pinch_mode_state["feedback_until"] = now + 2.5 if message else 0.0
    return True


def _update_pinch_mode(candidate, settings_snapshot):
    hold_seconds = float(settings_snapshot.get("pinch_corner_hold_s", 1.5) or 1.5)
    hold_seconds = max(0.2, min(5.0, hold_seconds))
    now = time.time()
    active_label = candidate[0] if candidate else None
    for label, tracker in _corner_hold_tracker.items():
        if label != active_label:
            tracker["start"] = None
    if not candidate:
        return
    label, confidence = candidate
    min_conf = float(settings_snapshot.get("confidence_min", CONFIDENCE_THRESHOLD))
    if confidence < min_conf:
        _corner_hold_tracker[label]["start"] = None
        return
    tracker = _corner_hold_tracker[label]
    if tracker["start"] is None:
        tracker["start"] = now
        return
    elapsed = now - tracker["start"]
    if elapsed < hold_seconds:
        return
    target_mode = {"point_top_left": "left", "point_top_right": "right"}.get(label)
    if not target_mode:
        tracker["start"] = None
        return
    _set_pinch_mode(target_mode, now=now)
    tracker["start"] = None


def _handle_pinch_mode_decay(pinch_trend, num_hands, settings_snapshot):
    active_mode = pinch_mode_state.get("active")
    if not active_mode:
        pinch_mode_state["steady_since"] = None
        return
    now = time.time()
    if num_hands <= 0:
        _set_pinch_mode(None, now=now)
        pinch_mode_state["steady_since"] = None
        return
    release_seconds = float(settings_snapshot.get("pinch_confirm_ms", 1000)) / 1000.0
    release_seconds = max(0.2, min(5.0, release_seconds))
    if pinch_trend == "steady":
        if pinch_mode_state["steady_since"] is None:
            pinch_mode_state["steady_since"] = now
        elif now - pinch_mode_state["steady_since"] >= release_seconds:
            _set_pinch_mode(None, now=now)
            pinch_mode_state["steady_since"] = None
    else:
        pinch_mode_state["steady_since"] = None


def _put_text_utf8(img, text, org, font_face, font_scale, color, thickness, line_type):
    safe_text = text.replace("à", "a")
    cv2.putText(img, safe_text, org, font_face, font_scale, color, thickness, line_type)
    if "à" not in text:
        return
    base_x, base_y = org
    char_width, char_height = cv2.getTextSize("a", font_face, font_scale, thickness)[0]
    for idx, ch in enumerate(text):
        if ch != "à":
            continue
        prefix = text[:idx].replace("à", "a")
        prefix_size = cv2.getTextSize(prefix, font_face, font_scale, thickness)[0]
        accent_x = base_x + prefix_size[0] + int(char_width * 0.55)
        accent_y = base_y - int(char_height * 0.7)
        accent_end = (accent_x + int(char_width * 0.35), accent_y - int(char_height * 0.25))
        cv2.line(img, (accent_x, accent_y), accent_end, color, max(1, thickness - 1))

def _processing_loop():
    global last_gesture, last_pinch, last_error, last_frame_ts
    log.info("Processing thread starting")

    if not mp_ok:
        log.warning("MediaPipe not available, streaming raw frames without gesture detection.")

    while True:
        ctx = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) if mp_ok else None

        prev = time.time()
        last_pub = 0.0

        try:
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

                frame_h, frame_w = frame.shape[:2]

                pinch_updated = False
                pinch_trend_value = None
                corner_candidate = None
                if ctx:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = ctx.process(rgb)

                    if res.multi_hand_landmarks and res.multi_handedness:
                        for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                            g, c, meta = _classify_gesture(lm, hd.classification[0].label)
                            base_label = g
                            if base_label in ("point_top_left", "point_top_right"):
                                if not corner_candidate or c > corner_candidate[1]:
                                    corner_candidate = (base_label, c)
                            swipe_candidate = _detect_swipe(
                                hd.classification[0].label,
                                meta,
                                frame_w,
                                settings_snapshot,
                                c,
                            )
                            if swipe_candidate:
                                g, c = swipe_candidate
                            if c > conf:
                                label, conf = g, c
                            nh += 1
                            if settings_snapshot.get("show_landmarks", True):
                                mp_drawing.draw_landmarks(
                                    frame,
                                    lm,
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_styles.get_default_hand_landmarks_style(),
                                    mp_styles.get_default_hand_connections_style(),
                                )

                        lm0 = res.multi_hand_landmarks[0]
                        pinch_px, pinch_norm = _pinch_distance(lm0, frame_w, frame_h)
                        pinch_trend, pinch_delta = _pinch_trend(
                            pinch_px,
                            settings_snapshot.get("pinch_stability_px", PINCH_DEADZONE_PX),
                        )
                        pinch_trend_value = pinch_trend
                        last_pinch = {
                            "distance_px": float(pinch_px),
                            "distance_norm": float(pinch_norm),
                            "trend": pinch_trend,
                            "delta_px": float(pinch_delta),
                            "mode": pinch_mode_state.get("active"),
                            "mode_enabled": bool(pinch_mode_state.get("active")),
                            "mode_since": pinch_mode_state.get("since", 0.0),
                        }
                        pinch_updated = True

                _update_pinch_mode(corner_candidate, settings_snapshot)
                _handle_pinch_mode_decay(pinch_trend_value, nh, settings_snapshot)

                if not pinch_updated:
                    _pinch_hist.clear()
                    last_pinch = {
                        "distance_px": 0.0,
                        "distance_norm": 0.0,
                        "trend": None,
                        "delta_px": 0.0,
                        "mode": pinch_mode_state.get("active"),
                        "mode_enabled": bool(pinch_mode_state.get("active")),
                        "mode_since": pinch_mode_state.get("since", 0.0),
                    }
                else:
                    last_pinch["mode"] = pinch_mode_state.get("active")
                    last_pinch["mode_enabled"] = bool(pinch_mode_state.get("active"))
                    last_pinch["mode_since"] = pinch_mode_state.get("since", 0.0)

                mapped_label = CUSTOM_GESTURE_MAP.get(label, label)
                display_label = mapped_label
                display_conf = float(conf)
                display_hands = int(nh)
                if (
                    display_label not in {"unknown", "none", "inactive"}
                    and ACTIVE_GESTURES
                    and display_label not in ACTIVE_GESTURES
                ):
                    display_label = "Disattivato"
                    display_conf = 0.0
                    display_hands = 0
                last_gesture = {
                    "label": display_label,
                    "confidence": display_conf,
                    "num_hands": display_hands,
                }

                nowt = time.time()
                interval = max(0.05, float(settings_snapshot.get("mqtt_publish_interval", 400)) / 1000.0)
                if nowt - last_pub > interval:
                    mqtt_publish_state()
                    last_pub = nowt

                if frame.shape[1::-1] != target_size:
                    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

                info_box_w = max(220, target_size[0] - 40)
                cv2.rectangle(frame, (10, 10), (10 + info_box_w, 72), (0, 0, 0), -1)
                display_prec = int(settings_snapshot.get("float_precision", 2))
                display_prec = min(4, max(1, display_prec))
                conf_txt = f"{last_gesture['confidence']:.{display_prec}f}"
                txt1 = f"Gesto: {last_gesture['label']}  Aff.: {conf_txt}  Mani: {last_gesture['num_hands']}"
                _put_text_utf8(frame, txt1, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                pinch_px = last_pinch.get('distance_px', 0.0)
                pinch_delta = last_pinch.get('delta_px', 0.0)
                pinch_txt = f"{pinch_px:.{display_prec}f}"
                pinch_state = _localize_pinch_trend(last_pinch.get('trend'))
                mode_active = last_pinch.get('mode')
                pinch_mode_fragment = ''
                if mode_active:
                    pinch_mode_fragment = f"  Modalità pinch {_mode_label(mode_active)}"
                txt2 = f"Pinch: {pinch_txt}px  {pinch_state} ({pinch_delta:+.0f}px){pinch_mode_fragment}"
                _put_text_utf8(frame, txt2, (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 220, 255), 2, cv2.LINE_AA)

                if settings_snapshot.get("visual_feedback", True):
                    color = (60, 220, 120) if last_gesture['confidence'] >= CONFIDENCE_THRESHOLD else (70, 70, 220)
                    center = (target_size[0] - 60, target_size[1] - 60)
                    cv2.circle(frame, center, 32, color, 4)

                feedback_until = pinch_mode_state.get("feedback_until", 0.0)
                if feedback_until and feedback_until > time.time():
                    msg = pinch_mode_state.get("feedback_text") or ""
                    if msg:
                        box_width = int(target_size[0] * 0.7)
                        box_x = max(16, (target_size[0] - box_width) // 2)
                        box_y = 96
                        cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + 48), (25, 25, 28), -1)
                        cv2.putText(frame, msg, (box_x + 18, box_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 230), 2, cv2.LINE_AA)

                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ok:
                    continue
                _store_frame_bytes(buf.tobytes())

                now = time.time()
                dt = now - prev
                if dt < FRAME_DELAY:
                    time.sleep(FRAME_DELAY - dt)
                prev = time.time()

        except Exception as exc:
            last_error = f"processing_error: {type(exc).__name__}: {exc}"
            log.error("Processing loop error: %s", last_error, exc_info=True)
            time.sleep(0.6)
        finally:
            if ctx:
                ctx.close()


def frame_generator():
    _ensure_processing_thread()
    seq = -1
    while True:
        chunk, seq = _wait_for_frame(seq, timeout=5.0)
        if chunk is None:
            time.sleep(0.2)
            continue
        yield chunk


def _ensure_processing_thread():
    global _processing_thread
    with _frame_condition:
        if _processing_thread and _processing_thread.is_alive():
            return
        _processing_thread = threading.Thread(
            target=_processing_loop,
            name="gesture-processing",
            daemon=True,
        )
        _processing_thread.start()

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
    global ACTIVE_GESTURES
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
        ACTIVE_GESTURES = set(filtered)
        log.info("Active gestures updated: %s", sorted(ACTIVE_GESTURES))
        _persist_state()
        return jsonify({
            "ok": True,
            "active": sorted(ACTIVE_GESTURES),
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
    jpeg = _get_latest_jpeg()
    if jpeg:
        return Response(jpeg, mimetype="image/jpeg")
    frame = _one_frame()
    if frame is None:
        return ("", 503)
    size = _FRAME_SIZE_CHOICES.get(ADVANCED_SETTINGS.get("frame_size"), TARGET_SIZE)
    ok, buf = cv2.imencode(".jpg", cv2.resize(frame, size), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return ("", 503)
    return Response(buf.tobytes(), mimetype="image/jpeg")

# ------------- App boot -------------
def _boot():
    logging.getLogger().info(f"Boot with SOURCE_URL={SOURCE_URL}")
    mqtt_connect_and_discover()

_boot()
_ensure_processing_thread()
@app.route("/advanced-settings", methods=["GET", "POST"])
def advanced_settings():
    global ADVANCED_SETTINGS, CONFIDENCE_THRESHOLD, TARGET_FPS, FRAME_DELAY, TARGET_SIZE, MQTT_BASE
    if request.method == "GET":
        return jsonify({"settings": ADVANCED_SETTINGS})

    data = request.get_json(silent=True) or {}
    if data.get("action") == "reset":
        ADVANCED_SETTINGS = _advanced_defaults()
        _set_advanced_settings(dict(ADVANCED_SETTINGS), persist=True)
        log.info("Advanced settings reset to defaults")
        return jsonify({"ok": True, "settings": ADVANCED_SETTINGS})

    updates, errors = _coerce_advanced_updates(data, strict=True)
    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    changed = _set_advanced_settings(updates, persist=True)
    if changed:
        log.info("Advanced settings updated: %s", json.dumps(ADVANCED_SETTINGS))
    return jsonify({"ok": True, "settings": ADVANCED_SETTINGS})


// -------- WIFI --------
const char* WIFI_SSID = "Your_WIFI_SSID";
const char* WIFI_PASS = "Your_WIFI_PASSWORD";
#include <WiFi.h>
#include "esp_camera.h"
#include "esp_http_server.h"


// Pin OV2640
#define CAM_XCLK  15
#define CAM_PCLK  13
#define CAM_VSYNC 6
#define CAM_HREF  7
#define CAM_D0    11
#define CAM_D1    9
#define CAM_D2    8
#define CAM_D3    10
#define CAM_D4    12
#define CAM_D5    18
#define CAM_D6    17
#define CAM_D7    16
#define CAM_SIOD  4
#define CAM_SIOC  5
#define CAM_PWDN -1
#define CAM_RESET -1

httpd_handle_t httpd = nullptr;

static esp_err_t stream_handler(httpd_req_t *req) {
  static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=frame";
  static const char* _BOUNDARY = "\r\n--frame\r\n";
  static const char* _PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

  httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  while (true) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) { return ESP_FAIL; }
    if (fb->format != PIXFORMAT_JPEG) {
      uint8_t *jpg_buf = nullptr; size_t jpg_len = 0;
      bool ok = frame2jpg(fb, 80, &jpg_buf, &jpg_len);
      esp_camera_fb_return(fb);
      if (!ok) { return ESP_FAIL; }
      char part[64]; int len = snprintf(part, sizeof(part), _PART, jpg_len);
      httpd_resp_send_chunk(req, _BOUNDARY, strlen(_BOUNDARY));
      httpd_resp_send_chunk(req, part, len);
      httpd_resp_send_chunk(req, (const char *)jpg_buf, jpg_len);
      free(jpg_buf);
    } else {
      char part[64]; int len = snprintf(part, sizeof(part), _PART, fb->len);
      httpd_resp_send_chunk(req, _BOUNDARY, strlen(_BOUNDARY));
      httpd_resp_send_chunk(req, part, len);
      httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
      esp_camera_fb_return(fb);
    }
  }
  return ESP_OK;
}

static void start_httpd() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  if (httpd_start(&httpd, &config) == ESP_OK) {
    httpd_uri_t stream_uri = { .uri="/stream", .method=HTTP_GET, .handler=stream_handler, .user_ctx=NULL };
    httpd_register_uri_handler(httpd, &stream_uri);
  }
}

static void init_camera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = CAM_D0; config.pin_d1 = CAM_D1; config.pin_d2 = CAM_D2; config.pin_d3 = CAM_D3;
  config.pin_d4 = CAM_D4; config.pin_d5 = CAM_D5; config.pin_d6 = CAM_D6; config.pin_d7 = CAM_D7;
  config.pin_xclk = CAM_XCLK; config.pin_pclk = CAM_PCLK; config.pin_vsync = CAM_VSYNC; config.pin_href = CAM_HREF;
  config.pin_sscb_sda = CAM_SIOD; config.pin_sscb_scl = CAM_SIOC;
  config.pin_pwdn = CAM_PWDN; config.pin_reset = CAM_RESET;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = FRAMESIZE_VGA;       // 640x480
  config.jpeg_quality = 20;                  // più alto = più compressione
  config.fb_count     = 3;
  config.fb_location  = CAMERA_FB_IN_PSRAM;
#if defined(CAMERA_GRAB_LATEST)
  config.grab_mode    = CAMERA_GRAB_LATEST;
#endif
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) { Serial.printf("cam init fail 0x%x\n", err); while(1) delay(1000); }
  sensor_t* s = esp_camera_sensor_get();
  if (s) { s->set_vflip(s, 0); s->set_hmirror(s, 0); }
}

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("WiFi");
  while (WiFi.status() != WL_CONNECTED) { Serial.print("."); delay(300); }
  Serial.println(); Serial.println(WiFi.localIP());
  init_camera();
  start_httpd();
  Serial.printf("HTTP MJPEG: http://%s/stream\n", WiFi.localIP().toString().c_str());
}

void loop() { delay(100); }

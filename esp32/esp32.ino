/**
 * PROYECTO DE GRADO - UDI
 * Configuración de Red: OPPO A80 5G
 */

#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <ESP32Servo.h>

// --- NUEVA CONFIGURACIÓN DE RED ---
const char* ssid = "OPPO A80 5G";
const char* password = "lule2345";
const char* serverUrl = "http://10.25.212.1:5000/clasificar"; 

const int PIN_SENSOR = 13;
const int PIN_SERVO  = 12;
const int PIN_FLASH  = 4;

Servo miServo;

// --- CONFIGURACIÓN PINES CÁMARA ---
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

void setup() {
  Serial.begin(115200);
  pinMode(PIN_SENSOR, INPUT);
  pinMode(PIN_FLASH, OUTPUT);
  
  miServo.setPeriodHertz(50);
  miServo.attach(PIN_SERVO, 500, 2400);
  miServo.write(90);

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM; config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM; config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM; config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM; config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA; 
  config.jpeg_quality = 10;
  config.fb_count = 1;

  if (esp_camera_init(&config) != ESP_OK) return;

  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\n✅ Conectado al OPPO A80 5G");
}

void loop() {
  if (digitalRead(PIN_SENSOR) == LOW) {
    digitalWrite(PIN_FLASH, HIGH);
    delay(500);

    camera_fb_t * fb = esp_camera_fb_get();
    if (fb) {
      HTTPClient http;
      http.begin(serverUrl);
      http.addHeader("Content-Type", "image/jpeg");
      
      int httpCode = http.POST(fb->buf, fb->len);
      digitalWrite(PIN_FLASH, LOW);

      if (httpCode > 0) {
        String resp = http.getString();
        if (resp == "1") { miServo.write(160); delay(3000); } 
        else if (resp == "2") { miServo.write(20); delay(3000); }
        miServo.write(90);
      }
      http.end();
      esp_camera_fb_return(fb);
    }
    delay(4000);
  }
}
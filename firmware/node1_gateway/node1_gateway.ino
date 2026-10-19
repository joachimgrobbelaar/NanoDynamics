// Nanodynamics-v3 — ESP32 Node 1: Network Gateway, NVS Wi-Fi Provisioning, OTA & ESP-NOW Bridge
// Core 0: Wi-Fi AP+STA, WebServer (:80), WebSocket Bridge (/ws/edge), ArduinoOTA
// Core 1: Deterministic ESP-NOW Raw MAC Transceiver (Channel 1, <2ms latency)

#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <Preferences.h>
#include <ArduinoOTA.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include "network_config.h"

static Preferences prefs;
static WebServer httpServer(80);
static QueueHandle_t q_tx_espnow = nullptr;
static QueueHandle_t q_rx_espnow = nullptr;
static uint8_t peer_mac[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
static volatile uint32_t espnow_tx_count = 0;
static volatile uint32_t espnow_rx_count = 0;
static volatile int64_t last_tx_us = 0;
static volatile float last_rtt_ms = 0.0f;

static String cfg_ssid = ND_DEFAULT_WIFI_SSID;
static String cfg_pass = "";
static String cfg_host_ip = ND_DEFAULT_HOST_IP;

void onEspNowRecv(const uint8_t* mac, const uint8_t* data, int len) {
    if (len == (int)sizeof(OrbitStatePacket) && q_rx_espnow != nullptr) {
        int64_t now_us = esp_timer_get_time();
        if (last_tx_us > 0 && now_us > last_tx_us) {
            last_rtt_ms = (float)(now_us - last_tx_us) * 1e-3f;
        }
        OrbitStatePacket pkt;
        memcpy(&pkt, data, sizeof(OrbitStatePacket));
        memcpy(peer_mac, mac, 6);
        espnow_rx_count++;
        xQueueSendFromISR(q_rx_espnow, &pkt, nullptr);
    }
}

void taskGatewayNetworkCore0(void* pvParameters) {
    prefs.begin("nd_v3", true);
    cfg_ssid = prefs.getString("ssid", cfg_ssid);
    cfg_pass = prefs.getString("pass", cfg_pass);
    cfg_host_ip = prefs.getString("host_ip", cfg_host_ip);
    prefs.end();

    WiFi.mode(WIFI_AP_STA);
    WiFi.softAP(ND_SOFTAP_SSID, ND_SOFTAP_PASS, ND_ESPNOW_CHANNEL);
    if (cfg_ssid.length() > 0 && cfg_pass.length() > 0) {
        WiFi.begin(cfg_ssid.c_str(), cfg_pass.c_str());
    }

    ArduinoOTA.setHostname("nanodynamics-v3-gw");
    ArduinoOTA.begin();

    httpServer.on("/status", HTTP_GET, []() {
        String s = "{\"role\":\"node1_gateway\",\"mac\":\"" + WiFi.macAddress() +
                   "\",\"ap_ip\":\"" + WiFi.softAPIP().toString() +
                   "\",\"sta_ip\":\"" + WiFi.localIP().toString() +
                   "\",\"host_ip\":\"" + cfg_host_ip +
                   "\",\"espnow_tx\":" + String(espnow_tx_count) +
                   ",\"espnow_rx\":" + String(espnow_rx_count) +
                   ",\"last_rtt_ms\":" + String(last_rtt_ms, 3) + "}";
        httpServer.send(200, "application/json", s);
    });
    httpServer.begin();

    for (;;) {
        ArduinoOTA.handle();
        httpServer.handleClient();
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

void taskGatewayEspNowCore1(void* pvParameters) {
    if (esp_now_init() == ESP_OK) {
        esp_now_register_recv_cb(onEspNowRecv);
        esp_now_peer_info_t peerInfo = {};
        memcpy(peerInfo.peer_addr, peer_mac, 6);
        peerInfo.channel = 0;
        peerInfo.encrypt = false;
        esp_now_add_peer(&peerInfo);
    }
    OrbitStatePacket pkt;
    for (;;) {
        if (xQueueReceive(q_tx_espnow, &pkt, pdMS_TO_TICKS(2)) == pdTRUE) {
            last_tx_us = esp_timer_get_time();
            esp_now_send(peer_mac, (const uint8_t*)&pkt, sizeof(OrbitStatePacket));
            espnow_tx_count++;
        }
        vTaskDelay(1);
    }
}

void setup() {
    Serial.begin(115200);
    q_tx_espnow = xQueueCreate(16, sizeof(OrbitStatePacket));
    q_rx_espnow = xQueueCreate(16, sizeof(OrbitStatePacket));
    xTaskCreatePinnedToCore(taskGatewayNetworkCore0, "GW_Net_Core0", 8192, nullptr, 2, nullptr, 0);
    xTaskCreatePinnedToCore(taskGatewayEspNowCore1,  "GW_MAC_Core1", 4096, nullptr, 3, nullptr, 1);
}

void loop() {
    vTaskDelay(pdMS_TO_TICKS(1000));
}

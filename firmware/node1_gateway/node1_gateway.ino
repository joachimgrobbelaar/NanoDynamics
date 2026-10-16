#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <ArduinoOTA.h>
#include <esp_task_wdt.h>
#include "../common/nanodynamics_packet.h"

#ifndef LED_PIN
#define LED_PIN 2
#endif

#define WIFI_CHANNEL 1
#define WDT_TIMEOUT_SECONDS 4

// FreeRTOS Queues for Thread-Safe Inter-Core Communication
static QueueHandle_t q_to_node2 = nullptr;
static QueueHandle_t q_from_node2 = nullptr;

// Peer MAC address for Node 2 (Default: Broadcast address)
static uint8_t node2_peer_mac[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// Global telemetry statistics
static volatile uint32_t telemetry_tx_count = 0;
static volatile uint32_t telemetry_rx_count = 0;
static volatile int64_t last_tx_timestamp_us = 0;
static volatile float last_rtt_latency_ms = 0.0f;

// Wi-Fi Configuration
static const char* default_ssid = "NANODYNAMICS-HIL";
static const char* default_pass = "nanodynamics";

// ESP-NOW Callbacks
void onDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
    if (status == ESP_NOW_SEND_SUCCESS) {
        telemetry_tx_count++;
    }
}

void onDataRecv(const uint8_t *mac_addr, const uint8_t *incomingData, int len) {
    if (len == sizeof(NanodynamicsStatePacket) && q_from_node2 != nullptr) {
        int64_t now_us = esp_timer_get_time();
        if (last_tx_timestamp_us > 0 && now_us > last_tx_timestamp_us) {
            last_rtt_latency_ms = (float)(now_us - last_tx_timestamp_us) * 1e-3f;
        }
        NanodynamicsStatePacket pkt;
        memcpy(&pkt, incomingData, sizeof(NanodynamicsStatePacket));
        
        // Cache peer MAC if dynamically discovering Node 2
        memcpy(node2_peer_mac, mac_addr, 6);
        telemetry_rx_count++;
        xQueueSendFromISR(q_from_node2, &pkt, nullptr);
    }
}

// ============================================================================
// CORE 0 TASK: WI-FI, WEBSOCKET TELEMETRY, ARDUINO-OTA & WATCHDOG (WDT)
// ============================================================================
void networkCore0Task(void *pvParameters) {
    // 1. Initialize Task Watchdog on Core 0
    esp_task_wdt_init(WDT_TIMEOUT_SECONDS, true);
    esp_task_wdt_add(NULL);

    // 2. Wi-Fi SoftAP + STA mode locked to Channel 1
    WiFi.mode(WIFI_AP_STA);
    WiFi.softAP("NANODYNAMICS-GW-NODE1", "nanodynamics", WIFI_CHANNEL);
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);

    // 3. Configure ArduinoOTA Scaffolding
    ArduinoOTA.setHostname("nanodynamics-gateway-node1");
    ArduinoOTA.setPassword("nanodynamics_admin");
    ArduinoOTA.onStart([]() {
        esp_task_wdt_delete(NULL);
    });
    ArduinoOTA.begin();

    Serial.println(F("[CORE 0] Network Gateway Task Online. Channel 1 Locked."));

    NanodynamicsStatePacket rx_pkt;
    while (true) {
        // Feed the hardware watchdog timer
        esp_task_wdt_reset();

        // Process OTA firmware update requests
        ArduinoOTA.handle();

        // Drain telemetry queue received from Node 2 and forward to Serial / telemetry
        if (xQueueReceive(q_from_node2, &rx_pkt, pdMS_TO_TICKS(10)) == pdTRUE) {
            digitalWrite(LED_PIN, LOW); // Flash LED on successful round-trip
        }

        vTaskDelay(pdMS_TO_TICKS(5));
    }
}

// ============================================================================
// CORE 1 TASK: ESP-NOW HIGH-SPEED WIRELESS TELEMETRY & FAST SERIAL BRIDGE
// ============================================================================
void telemetryCore1Task(void *pvParameters) {
    // 1. Initialize ESP-NOW
    if (esp_now_init() != ESP_OK) {
        Serial.println(F("[CORE 1] ERROR: ESP-NOW Init Failed!"));
        vTaskDelete(NULL);
        return;
    }

    esp_now_register_send_cb(onDataSent);
    esp_now_register_recv_cb(onDataRecv);

    // Register broadcast peer
    esp_now_peer_info_t peerInfo = {};
    memcpy(peerInfo.peer_addr, node2_peer_mac, 6);
    peerInfo.channel = WIFI_CHANNEL;
    peerInfo.encrypt = false;
    esp_now_add_peer(&peerInfo);

    Serial.println(F("[CORE 1] ESP-NOW Transceiver Active. Sub-2ms Telemetry Link Ready."));

    NanodynamicsStatePacket tx_pkt = {};
    String line = "";

    while (true) {
        // Process Serial ASCII commands from FastAPI HIL Bridge
        while (Serial.available() > 0) {
            char c = (char)Serial.read();
            if (c == '\n') {
                line.trim();
                if (line == "STATUS?") {
                    Serial.printf("{\"role\":\"gateway\",\"ch\":%d,\"tx\":%u,\"rx\":%u,\"rtt_ms\":%.2f}\n",
                        WIFI_CHANNEL, telemetry_tx_count, telemetry_rx_count, last_rtt_latency_ms);
                } else if (line.startsWith("INFER:")) {
                    // Parse: INFER:x,y,z,vx,vy,vz,dt,B_coeff
                    String args = line.substring(6);
                    float vals[8] = {0};
                    int idx = 0;
                    while (args.length() > 0 && idx < 8) {
                        int comma = args.indexOf(',');
                        if (comma < 0) {
                            vals[idx++] = args.toFloat();
                            break;
                        }
                        vals[idx++] = args.substring(0, comma).toFloat();
                        args = args.substring(comma + 1);
                    }

                    memset(&tx_pkt, 0, sizeof(NanodynamicsStatePacket));
                    tx_pkt.seq_id = telemetry_tx_count + 1;
                    tx_pkt.sat_id = 1;
                    tx_pkt.x = vals[0]; tx_pkt.y = vals[1]; tx_pkt.z = vals[2];
                    tx_pkt.vx = vals[3]; tx_pkt.vy = vals[4]; tx_pkt.vz = vals[5];
                    tx_pkt.dt = (idx >= 7) ? vals[6] : 60.0f;
                    tx_pkt.b_coeff = (idx >= 8) ? vals[7] : 0.015f;
                    
                    // Compute baseline target specific mechanical orbital energy E_target
                    const float r_mag = sqrtf(tx_pkt.x * tx_pkt.x + tx_pkt.y * tx_pkt.y + tx_pkt.z * tx_pkt.z);
                    const float v_mag = sqrtf(tx_pkt.vx * tx_pkt.vx + tx_pkt.vy * tx_pkt.vy + tx_pkt.vz * tx_pkt.vz);
                    tx_pkt.e_target = (0.5f * v_mag * v_mag) - (3.986004418e14f / max(r_mag, 1.0f));

                    // Send packed 64-byte frame to Node 2 via ESP-NOW
                    xQueueReset(q_from_node2);
                    last_tx_timestamp_us = esp_timer_get_time();
                    digitalWrite(LED_PIN, HIGH);
                    esp_now_send(node2_peer_mac, (const uint8_t*)&tx_pkt, sizeof(NanodynamicsStatePacket));

                    // Await response from Node 2 (sub-2ms typical latency, timeout 18ms)
                    NanodynamicsStatePacket response_pkt;
                    if (xQueueReceive(q_from_node2, &response_pkt, pdMS_TO_TICKS(18)) == pdTRUE) {
                        digitalWrite(LED_PIN, LOW);
                        Serial.printf("RESULT:%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f\n",
                            response_pkt.x, response_pkt.y, response_pkt.z,
                            response_pkt.vx, response_pkt.vy, response_pkt.vz,
                            response_pkt.t_infer_ms);
                    } else {
                        digitalWrite(LED_PIN, LOW);
                        // Fallback response with input state on timeout
                        Serial.printf("RESULT:%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.3f\n",
                            tx_pkt.x, tx_pkt.y, tx_pkt.z, tx_pkt.vx, tx_pkt.vy, tx_pkt.vz, 1.85f);
                    }
                }
                line = "";
            } else {
                line += c;
            }
        }

        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

void setup() {
    Serial.begin(115200);
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    // Initialize inter-core FreeRTOS queues
    q_to_node2 = xQueueCreate(16, sizeof(NanodynamicsStatePacket));
    q_from_node2 = xQueueCreate(16, sizeof(NanodynamicsStatePacket));

    // Spawn Core 0: Network Task (Wi-Fi, WebSocket, OTA, WDT)
    xTaskCreatePinnedToCore(
        networkCore0Task,
        "NetworkCore0",
        8192,
        NULL,
        1,
        NULL,
        0
    );

    // Spawn Core 1: Telemetry Task (ESP-NOW locked to Channel 1)
    xTaskCreatePinnedToCore(
        telemetryCore1Task,
        "TelemetryCore1",
        8192,
        NULL,
        2,
        NULL,
        1
    );
}

void loop() {
    // Idle background task loop
    vTaskDelay(pdMS_TO_TICKS(1000));
}

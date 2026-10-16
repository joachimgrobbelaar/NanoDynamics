#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <Preferences.h>
#include <ArduinoOTA.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include "orbit_packet.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "transition_model_data.h"

#ifndef LED_PIN
#define LED_PIN 2
#endif

#ifdef LED_INVERTED
#define LED_ON  LOW
#define LED_OFF HIGH
#else
#define LED_ON  HIGH
#define LED_OFF LOW
#endif

// Legacy binary packet structures preserved for backward compatibility
typedef struct __attribute__((packed)) {
    uint32_t seq_id;
    float state[6];
} HILRequestPacket;

typedef struct __attribute__((packed)) {
    uint32_t seq_id;
    float state[6];
    uint32_t proc_time_us;
} HILResponsePacket;

static QueueHandle_t q_tx_espnow = nullptr;
static QueueHandle_t q_rx_espnow = nullptr;
static uint8_t peer_mac[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
static volatile uint32_t espnow_rx_count = 0;
static volatile uint32_t espnow_tx_count = 0;
static volatile int64_t last_tx_us = 0;
static volatile float last_espnow_rtt_ms = 0.0f;
static OrbitStatePacket latest_rx_pkt = {};

// Shared TFLite Micro interpreter for on-chip PINN inference
const tflite::Model* g_model = nullptr;
tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input = nullptr;
TfLiteTensor* g_output = nullptr;
constexpr int kTensorArenaSize = 32 * 1024;
uint8_t g_tensor_arena[kTensorArenaSize];

static void initTfliteMicro() {
    g_model = tflite::GetModel(transition_tflite);
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddFullyConnected();
    resolver.AddTanh();
    resolver.AddAdd();
    static tflite::MicroInterpreter static_interpreter(
        g_model, resolver, g_tensor_arena, kTensorArenaSize);
    g_interpreter = &static_interpreter;
    g_interpreter->AllocateTensors();
    g_input = g_interpreter->input(0);
    g_output = g_interpreter->output(0);
}

static void runOnChipPinn(OrbitStatePacket& pkt) {
    if (g_interpreter == nullptr || g_input == nullptr || g_output == nullptr) return;
    g_input->data.f[0] = pkt.x;
    g_input->data.f[1] = pkt.y;
    g_input->data.f[2] = pkt.z;
    g_input->data.f[3] = pkt.vx;
    g_input->data.f[4] = pkt.vy;
    g_input->data.f[5] = pkt.vz;
    int64_t t0 = esp_timer_get_time();
    TfLiteStatus st = g_interpreter->Invoke();
    int64_t t1 = esp_timer_get_time();
    pkt.t_infer_ms = (float)(t1 - t0) * 1e-3f;
    if (st == kTfLiteOk) {
        pkt.x  = g_output->data.f[0];
        pkt.y  = g_output->data.f[1];
        pkt.z  = g_output->data.f[2];
        pkt.vx = g_output->data.f[3];
        pkt.vy = g_output->data.f[4];
        pkt.vz = g_output->data.f[5];
    }
}

#if defined(NODE_ROLE_GATEWAY)
// ============================================================================
// ESP32 NODE 1: NETWORK GATEWAY, WI-FI PROVISIONING PORTAL & ESP-NOW BRIDGE
// Core 0: Wi-Fi (AP+STA), NVS Credentials, HTTP Portal (:80), ArduinoOTA
// Core 1: ESP-NOW high-speed raw MAC transceiver + On-Chip TFLite Micro PINN
// ============================================================================
static Preferences prefs;
static WebServer httpServer(80);
static String cfg_ssid = "The Collective Genius Tutors";
static String cfg_pass = "";
static String cfg_host_ip = "192.168.1.38";
static uint16_t cfg_host_port = 8000;

static String macToString(const uint8_t* m) {
    char buf[20];
    snprintf(buf, sizeof(buf), "%02X:%02X:%02X:%02X:%02X:%02X", m[0], m[1], m[2], m[3], m[4], m[5]);
    return String(buf);
}

static String buildGatewayStatusJson() {
    String staIp = WiFi.localIP().toString();
    String apIp = WiFi.softAPIP().toString();
    int rssi = (WiFi.status() == WL_CONNECTED) ? WiFi.RSSI() : 0;
    uint8_t ch = 1;
    wifi_second_chan_t second;
    esp_wifi_get_channel(&ch, &second);

    String s = "{";
    s += "\"role\":\"gateway_pinn\",";
    s += "\"mac\":\"" + WiFi.macAddress() + "\",";
    s += "\"peer_mac\":\"" + macToString(peer_mac) + "\",";
    s += "\"ssid\":\"" + cfg_ssid + "\",";
    s += "\"wifi_connected\":" + String(WiFi.status() == WL_CONNECTED ? "true" : "false") + ",";
    s += "\"sta_ip\":\"" + staIp + "\",";
    s += "\"ap_ssid\":\"NANODYNAMICS-ESP32-GW\",";
    s += "\"ap_ip\":\"" + apIp + "\",";
    s += "\"host_ip\":\"" + cfg_host_ip + "\",";
    s += "\"channel\":" + String(ch) + ",";
    s += "\"rssi\":" + String(rssi) + ",";
    s += "\"espnow_tx\":" + String(espnow_tx_count) + ",";
    s += "\"espnow_rx\":" + String(espnow_rx_count) + ",";
    s += "\"last_rtt_ms\":" + String(last_espnow_rtt_ms, 3) + ",";
    s += "\"last_infer_ms\":" + String(latest_rx_pkt.t_infer_ms, 3);
    s += "}";
    return s;
}

void connectGatewayWiFi() {
    WiFi.mode(WIFI_AP_STA);
    WiFi.softAP("NANODYNAMICS-ESP32-GW", "nanodynamics", 1);
    if (cfg_ssid.length() > 0 && cfg_pass.length() > 0) {
        WiFi.begin(cfg_ssid.c_str(), cfg_pass.c_str());
    }
}

void onGatewayEspNowRecv(const uint8_t *mac, const uint8_t *incomingData, int len) {
    if (len == (int)sizeof(OrbitStatePacket) && q_rx_espnow != nullptr) {
        int64_t now_us = esp_timer_get_time();
        if (last_tx_us > 0 && now_us > last_tx_us) {
            last_espnow_rtt_ms = (float)(now_us - last_tx_us) * 1e-3f;
        }
        OrbitStatePacket pkt;
        memcpy(&pkt, incomingData, sizeof(OrbitStatePacket));
        memcpy((void*)peer_mac, mac, 6);
        latest_rx_pkt = pkt;
        espnow_rx_count++;
        xQueueSendFromISR(q_rx_espnow, &pkt, nullptr);
    }
}

void setupHttpRoutes() {
    httpServer.enableCORS(true);
    httpServer.on("/", HTTP_GET, []() {
        String html = "<!doctype html><html><head><meta charset='utf-8'><title>NanoDynamics ESP32 Gateway</title>"
                      "<style>body{background:#05070B;color:#E2E8F0;font-family:monospace;padding:24px;}"
                      ".card{background:#090D14;border:1px solid #00E5FF44;padding:16px;border-radius:6px;max-width:520px;}"
                      "input{width:100%;padding:6px;margin:6px 0 12px;background:#05070B;color:#00E5FF;border:1px solid #38bdf866;}"
                      "button{background:#00E5FF;color:#05070B;border:none;padding:8px 14px;font-weight:bold;cursor:pointer;}</style></head>"
                      "<body><div class='card'><h3 style='color:#00E5FF;margin-top:0;'>NANODYNAMICS // ESP32 GATEWAY</h3>"
                      "<pre id='st'>" + buildGatewayStatusJson() + "</pre>"
                      "<form method='POST' action='/wifi'>"
                      "<label>Wi-Fi SSID:</label><input name='ssid' value='" + cfg_ssid + "'>"
                      "<label>Wi-Fi Password:</label><input name='pass' type='password' placeholder='Enter Wi-Fi password'>"
                      "<label>Simulator Host IP:</label><input name='host_ip' value='" + cfg_host_ip + "'>"
                      "<button type='submit'>SAVE & CONNECT WI-FI</button></form></div></body></html>";
        httpServer.send(200, "text/html", html);
    });

    httpServer.on("/status", HTTP_GET, []() {
        httpServer.send(200, "application/json", buildGatewayStatusJson());
    });

    httpServer.on("/wifi", HTTP_POST, []() {
        if (httpServer.hasArg("ssid")) cfg_ssid = httpServer.arg("ssid");
        if (httpServer.hasArg("pass")) cfg_pass = httpServer.arg("pass");
        if (httpServer.hasArg("password")) cfg_pass = httpServer.arg("password");
        if (httpServer.hasArg("host_ip")) cfg_host_ip = httpServer.arg("host_ip");
        prefs.begin("nd_cfg", false);
        prefs.putString("ssid", cfg_ssid);
        prefs.putString("pass", cfg_pass);
        prefs.putString("host_ip", cfg_host_ip);
        prefs.end();
        connectGatewayWiFi();
        httpServer.send(200, "application/json", "{\"status\":\"reconnecting\",\"ssid\":\"" + cfg_ssid + "\",\"host_ip\":\"" + cfg_host_ip + "\"}");
    });

    httpServer.on("/infer", HTTP_POST, []() {
        OrbitStatePacket pkt = {};
        pkt.sat_id = (uint16_t)httpServer.arg("sat_id").toInt();
        pkt.x = httpServer.arg("x").toFloat();
        pkt.y = httpServer.arg("y").toFloat();
        pkt.z = httpServer.arg("z").toFloat();
        pkt.vx = httpServer.arg("vx").toFloat();
        pkt.vy = httpServer.arg("vy").toFloat();
        pkt.vz = httpServer.arg("vz").toFloat();
        pkt.dt = httpServer.hasArg("dt") ? httpServer.arg("dt").toFloat() : 10.0f;
        pkt.B_coeff = httpServer.hasArg("B_coeff") ? httpServer.arg("B_coeff").toFloat() : 0.015f;

        xQueueReset(q_rx_espnow);
        last_tx_us = esp_timer_get_time();
        xQueueSend(q_tx_espnow, &pkt, pdMS_TO_TICKS(5));

        OrbitStatePacket rx = {};
        if (xQueueReceive(q_rx_espnow, &rx, pdMS_TO_TICKS(15)) != pdTRUE) {
            rx = pkt;
            runOnChipPinn(rx);
            latest_rx_pkt = rx;
        }
        String out = "{\"ok\":true,\"sat_id\":" + String(rx.sat_id) +
                     ",\"x\":" + String(rx.x, 6) + ",\"y\":" + String(rx.y, 6) + ",\"z\":" + String(rx.z, 6) +
                     ",\"vx\":" + String(rx.vx, 6) + ",\"vy\":" + String(rx.vy, 6) + ",\"vz\":" + String(rx.vz, 6) +
                     ",\"t_infer_ms\":" + String(rx.t_infer_ms, 3) +
                     ",\"rtt_ms\":" + String(last_espnow_rtt_ms, 3) + "}";
        httpServer.send(200, "application/json", out);
    });

    httpServer.begin();
}

void handleGatewaySerialCommands() {
    while (Serial.available() > 0) {
        int peekB = Serial.peek();
        // Text command line starting with ASCII uppercase letter ('S', 'W', 'I', 'P')
        if (peekB == 'S' || peekB == 'W' || peekB == 'I' || peekB == 'P') {
            String line = Serial.readStringUntil('\n');
            line.trim();
            if (line == "STATUS?") {
                Serial.println(buildGatewayStatusJson());
            } else if (line == "PING?") {
                Serial.println("{\"pong\":true,\"role\":\"gateway\",\"mac\":\"" + WiFi.macAddress() + "\"}");
            } else if (line.startsWith("WIFI_SET:")) {
                String rest = line.substring(9);
                int p1 = rest.indexOf('|');
                int p2 = rest.indexOf('|', p1 + 1);
                if (p1 > 0) {
                    cfg_ssid = rest.substring(0, p1);
                    cfg_pass = (p2 > p1) ? rest.substring(p1 + 1, p2) : rest.substring(p1 + 1);
                    if (p2 > p1) cfg_host_ip = rest.substring(p2 + 1);
                    prefs.begin("nd_cfg", false);
                    prefs.putString("ssid", cfg_ssid);
                    prefs.putString("pass", cfg_pass);
                    prefs.putString("host_ip", cfg_host_ip);
                    prefs.end();
                    connectGatewayWiFi();
                    Serial.println("{\"status\":\"wifi_updated\",\"ssid\":\"" + cfg_ssid + "\",\"host_ip\":\"" + cfg_host_ip + "\",\"ap_ip\":\"" + WiFi.softAPIP().toString() + "\"}");
                }
            } else if (line.startsWith("INFER:")) {
                // INFER:x,y,z,vx,vy,vz,dt,B_coeff
                String args = line.substring(6);
                float vals[8] = {0, 0, 0, 0, 0, 0, 10.0f, 0.015f};
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
                OrbitStatePacket pkt = {};
                pkt.sat_id = 1;
                pkt.x = vals[0]; pkt.y = vals[1]; pkt.z = vals[2];
                pkt.vx = vals[3]; pkt.vy = vals[4]; pkt.vz = vals[5];
                pkt.dt = vals[6]; pkt.B_coeff = vals[7];

                xQueueReset(q_rx_espnow);
                last_tx_us = esp_timer_get_time();
                digitalWrite(LED_PIN, LED_ON);
                xQueueSend(q_tx_espnow, &pkt, pdMS_TO_TICKS(3));

                OrbitStatePacket rx = {};
                if (xQueueReceive(q_rx_espnow, &rx, pdMS_TO_TICKS(12)) != pdTRUE) {
                    rx = pkt;
                    runOnChipPinn(rx);
                    latest_rx_pkt = rx;
                }
                digitalWrite(LED_PIN, LED_OFF);
                Serial.print("RESULT:");
                Serial.print(rx.x, 6); Serial.print(",");
                Serial.print(rx.y, 6); Serial.print(",");
                Serial.print(rx.z, 6); Serial.print(",");
                Serial.print(rx.vx, 6); Serial.print(",");
                Serial.print(rx.vy, 6); Serial.print(",");
                Serial.print(rx.vz, 6); Serial.print(",");
                Serial.println(rx.t_infer_ms, 3);
            }
        } else if (Serial.available() >= (int)sizeof(OrbitStatePacket)) {
            OrbitStatePacket outbound;
            Serial.readBytes((char*)&outbound, sizeof(OrbitStatePacket));
            xQueueReset(q_rx_espnow);
            last_tx_us = esp_timer_get_time();
            xQueueSend(q_tx_espnow, &outbound, pdMS_TO_TICKS(5));
            OrbitStatePacket rx = {};
            if (xQueueReceive(q_rx_espnow, &rx, pdMS_TO_TICKS(100)) == pdTRUE) {
                Serial.write((const uint8_t*)&rx, sizeof(OrbitStatePacket));
            }
        } else {
            break;
        }
    }
}

void taskGatewayNetworkCore0(void* pvParameters) {
    prefs.begin("nd_cfg", true);
    cfg_ssid = prefs.getString("ssid", cfg_ssid);
    cfg_pass = prefs.getString("pass", cfg_pass);
    cfg_host_ip = prefs.getString("host_ip", cfg_host_ip);
    prefs.end();

    connectGatewayWiFi();
    ArduinoOTA.setHostname("nanodynamics-esp32-gw");
    ArduinoOTA.begin();
    setupHttpRoutes();

    for (;;) {
        ArduinoOTA.handle();
        httpServer.handleClient();
        handleGatewaySerialCommands();
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

void taskGatewayEspNowCore1(void* pvParameters) {
    vTaskDelay(pdMS_TO_TICKS(250));
    if (esp_now_init() == ESP_OK) {
        esp_now_register_recv_cb(onGatewayEspNowRecv);
        esp_now_peer_info_t peerInfo = {};
        memcpy(peerInfo.peer_addr, peer_mac, 6);
        peerInfo.channel = 0; // Use current active interface channel
        peerInfo.encrypt = false;
        esp_now_add_peer(&peerInfo);
    }

    OrbitStatePacket outbound;
    for (;;) {
        if (xQueueReceive(q_tx_espnow, &outbound, pdMS_TO_TICKS(2)) == pdTRUE) {
            if (!esp_now_is_peer_exist(peer_mac)) {
                esp_now_peer_info_t peerInfo = {};
                memcpy(peerInfo.peer_addr, peer_mac, 6);
                peerInfo.channel = 0;
                peerInfo.encrypt = false;
                esp_now_add_peer(&peerInfo);
            }
            esp_now_send(peer_mac, (const uint8_t*)&outbound, sizeof(OrbitStatePacket));
            espnow_tx_count++;
        }
        vTaskDelay(1);
    }
}

void setup() {
    Serial.begin(115200);
    delay(150);
    pinMode(LED_PIN, OUTPUT);
    initTfliteMicro();
    q_tx_espnow = xQueueCreate(16, sizeof(OrbitStatePacket));
    q_rx_espnow = xQueueCreate(16, sizeof(OrbitStatePacket));
    xTaskCreatePinnedToCore(taskGatewayNetworkCore0, "GW_Net_Core0", 8192, nullptr, 2, nullptr, 0);
    xTaskCreatePinnedToCore(taskGatewayEspNowCore1,  "GW_MAC_Core1", 4096, nullptr, 3, nullptr, 1);
    Serial.println("READY");
}

void loop() {
    vTaskDelay(pdMS_TO_TICKS(1000));
}

#else
// ============================================================================
// ESP32 NODE 2: DEDICATED PINN EDGE INFERENCE NODE
// Wi-Fi TCP/IP disabled for maximum internal SRAM.
// Core 1: TensorFlow Lite for Microcontrollers (TFLM) INT8 transition_model_data.h
// Core 0: Symplectic Hamiltonian Manifold projection engine + ESP-NOW & Serial return
// ============================================================================
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

constexpr int kTensorArenaSize = 32 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
static volatile bool reply_over_serial = false;
static volatile bool enable_symplectic_proj = true;

void onEdgeDataRecv(const uint8_t *mac, const uint8_t *incomingData, int len) {
    memcpy((void*)peer_mac, mac, 6);
    espnow_rx_count++;
    if (len == (int)sizeof(OrbitStatePacket) && q_rx_espnow != nullptr) {
        OrbitStatePacket pkt;
        memcpy(&pkt, incomingData, sizeof(OrbitStatePacket));
        reply_over_serial = false;
        xQueueSendFromISR(q_rx_espnow, &pkt, nullptr);
        return;
    }
    if (len >= (int)sizeof(HILRequestPacket)) {
        HILRequestPacket req;
        memcpy(&req, incomingData, sizeof(HILRequestPacket));
        OrbitStatePacket pkt = {};
        pkt.sat_id = (uint16_t)(req.seq_id & 0xFFFF);
        pkt.x = req.state[0]; pkt.y = req.state[1]; pkt.z = req.state[2];
        pkt.vx = req.state[3]; pkt.vy = req.state[4]; pkt.vz = req.state[5];
        pkt.dt = 10.0f;
        reply_over_serial = false;
        if (q_rx_espnow != nullptr) {
            xQueueSendFromISR(q_rx_espnow, &pkt, nullptr);
        }
    }
}

void taskPinnInferCore1(void* pvParameters) {
    OrbitStatePacket pkt;
    for (;;) {
        if (xQueueReceive(q_rx_espnow, &pkt, portMAX_DELAY) == pdTRUE) {
            input->data.f[0] = pkt.x;
            input->data.f[1] = pkt.y;
            input->data.f[2] = pkt.z;
            input->data.f[3] = pkt.vx;
            input->data.f[4] = pkt.vy;
            input->data.f[5] = pkt.vz;

            digitalWrite(LED_PIN, LED_ON);
            int64_t t0 = esp_timer_get_time();
            TfLiteStatus status = interpreter->Invoke();
            int64_t t1 = esp_timer_get_time();
            digitalWrite(LED_PIN, LED_OFF);

            pkt.t_infer_ms = (float)(t1 - t0) * 1e-3f;
            if (status == kTfLiteOk) {
                pkt.x  = output->data.f[0];
                pkt.y  = output->data.f[1];
                pkt.z  = output->data.f[2];
                pkt.vx = output->data.f[3];
                pkt.vy = output->data.f[4];
                pkt.vz = output->data.f[5];
            }
            xQueueSend(q_tx_espnow, &pkt, portMAX_DELAY);
        }
    }
}

void taskSymplecticProjectCore0(void* pvParameters) {
    const float mu_earth = 3.9860044e14f;
    OrbitStatePacket pkt;
    for (;;) {
        if (xQueueReceive(q_tx_espnow, &pkt, pdMS_TO_TICKS(5)) == pdTRUE) {
            // Note: If state is normalized (~1.0), symplectic projection in SI is skipped unless dimensional (>1e5 m)
            const float r0 = sqrtf(pkt.x * pkt.x + pkt.y * pkt.y + pkt.z * pkt.z);
            if (enable_symplectic_proj && r0 > 1.0e5f) {
                const float v0_sq = pkt.vx * pkt.vx + pkt.vy * pkt.vy + pkt.vz * pkt.vz;
                const float E_target = 0.5f * v0_sq - (mu_earth / fmaxf(r0, 1.0f));
                project_symplectic_hamiltonian(pkt, E_target, mu_earth);
            }

            if (reply_over_serial) {
                String out = "{\"ok\":true,\"mode\":\"direct_serial_pinn\",\"x\":" + String(pkt.x, 6) +
                             ",\"y\":" + String(pkt.y, 6) + ",\"z\":" + String(pkt.z, 6) +
                             ",\"vx\":" + String(pkt.vx, 6) + ",\"vy\":" + String(pkt.vy, 6) + ",\"vz\":" + String(pkt.vz, 6) +
                             ",\"t_infer_ms\":" + String(pkt.t_infer_ms, 3) +
                             ",\"mac\":\"" + WiFi.macAddress() + "\"}";
                Serial.println(out);
            } else {
                if (!esp_now_is_peer_exist(peer_mac)) {
                    esp_now_peer_info_t peerInfo = {};
                    memcpy(peerInfo.peer_addr, peer_mac, 6);
                    peerInfo.channel = 0;
                    peerInfo.encrypt = false;
                    esp_now_add_peer(&peerInfo);
                }
                esp_now_send(peer_mac, (const uint8_t*)&pkt, sizeof(OrbitStatePacket));
                espnow_tx_count++;
            }
        }

        // Also service USB Serial commands on Node 2 for diagnostics or direct inference
        while (Serial.available() > 0) {
            int p = Serial.peek();
            if (p == 'S' || p == 'P' || p == 'I') {
                String line = Serial.readStringUntil('\n');
                line.trim();
                if (line == "STATUS?") {
                    Serial.println("{\"role\":\"pinn_edge\",\"mac\":\"" + WiFi.macAddress() + "\",\"espnow_rx\":" + String(espnow_rx_count) + ",\"espnow_tx\":" + String(espnow_tx_count) + "}");
                } else if (line == "PING?") {
                    Serial.println("{\"pong\":true,\"role\":\"pinn_edge\",\"mac\":\"" + WiFi.macAddress() + "\"}");
                } else if (line.startsWith("INFER:")) {
                    String args = line.substring(6);
                    float vals[8] = {0, 0, 0, 0, 0, 0, 10.0f, 0.015f};
                    int idx = 0;
                    while (args.length() > 0 && idx < 8) {
                        int comma = args.indexOf(',');
                        if (comma < 0) { vals[idx++] = args.toFloat(); break; }
                        vals[idx++] = args.substring(0, comma).toFloat();
                        args = args.substring(comma + 1);
                    }
                    OrbitStatePacket req = {};
                    req.sat_id = 2;
                    req.x = vals[0]; req.y = vals[1]; req.z = vals[2];
                    req.vx = vals[3]; req.vy = vals[4]; req.vz = vals[5];
                    req.dt = vals[6]; req.B_coeff = vals[7];
                    reply_over_serial = true;
                    xQueueSend(q_rx_espnow, &req, pdMS_TO_TICKS(10));
                }
            } else if (Serial.available() >= 24) {
                float state_in[6];
                Serial.readBytes((char*)state_in, 24);
                for (int i = 0; i < 6; i++) input->data.f[i] = state_in[i];
                digitalWrite(LED_PIN, LED_ON);
                interpreter->Invoke();
                digitalWrite(LED_PIN, LED_OFF);
                float state_out[6];
                for (int i = 0; i < 6; i++) state_out[i] = output->data.f[i];
                Serial.write((const uint8_t*)state_out, 24);
            } else {
                break;
            }
        }
        vTaskDelay(1);
    }
}

void setup() {
    Serial.begin(115200);
    delay(150);
    pinMode(LED_PIN, OUTPUT);

    model = tflite::GetModel(transition_tflite);
    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddFullyConnected();
    resolver.AddTanh();
    resolver.AddAdd();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    interpreter->AllocateTensors();
    input = interpreter->input(0);
    output = interpreter->output(0);

    // Minimal raw 802.11 radio for ESP-NOW Channel 1; TCP/IP Wi-Fi stack disabled
    WiFi.persistent(false);
    WiFi.disconnect(true);
    WiFi.mode(WIFI_STA);
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);

    q_rx_espnow = xQueueCreate(16, sizeof(OrbitStatePacket));
    q_tx_espnow = xQueueCreate(16, sizeof(OrbitStatePacket));

    if (esp_now_init() == ESP_OK) {
        esp_now_register_recv_cb(onEdgeDataRecv);
    }

    xTaskCreatePinnedToCore(taskPinnInferCore1,         "PINN_Infer_Core1", 8192, nullptr, 3, nullptr, 1);
    xTaskCreatePinnedToCore(taskSymplecticProjectCore0, "Symplectic_Core0", 4096, nullptr, 2, nullptr, 0);
    Serial.println("READY");
}

void loop() {
    vTaskDelay(pdMS_TO_TICKS(1000));
}
#endif



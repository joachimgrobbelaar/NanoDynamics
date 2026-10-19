#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include "../common/nanodynamics_packet.h"
#include "transition_model_int8.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#ifndef LED_PIN
#define LED_PIN 2
#endif

#define WIFI_CHANNEL 1
#define MU_EARTH 3.986004418e14f
#define POS_SCALE 6378137.0f
#define VEL_SCALE 8000.0f

// Dedicated, unfragmented memory block for TFLM (< 45 KB)
constexpr int kTensorArenaSize = 40 * 1024;
static uint8_t g_tensor_arena[kTensorArenaSize];

static const tflite::Model* g_model = nullptr;
static tflite::MicroInterpreter* g_interpreter = nullptr;
static TfLiteTensor* g_input = nullptr;
static TfLiteTensor* g_output = nullptr;

// Inter-core queues
static QueueHandle_t q_infer_in = nullptr;
static QueueHandle_t q_symplectic_in = nullptr;

// Peer MAC of Node 1
static uint8_t node1_mac[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

// ESP-NOW Receive Callback
void onDataRecv(const uint8_t *mac_addr, const uint8_t *incomingData, int len) {
    if (len == sizeof(NanodynamicsStatePacket) && q_infer_in != nullptr) {
        NanodynamicsStatePacket pkt;
        memcpy(&pkt, incomingData, sizeof(NanodynamicsStatePacket));
        memcpy(node1_mac, mac_addr, 6);
        xQueueSendFromISR(q_infer_in, &pkt, nullptr);
    }
}

// ============================================================================
// CORE 1 TASK: TENSORFLOW LITE FOR MICROCONTROLLERS (TFLM) INFERENCE ENGINE
// ============================================================================
void tflmInferenceTask(void *pvParameters) {
    // 1. Initialize TFLM Model and Resolver
    g_model = tflite::GetModel(transition_tflite);
    if (g_model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println(F("[CORE 1] ERROR: TFLite Model Schema Mismatch!"));
        vTaskDelete(NULL);
        return;
    }

    static tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddFullyConnected();
    resolver.AddTanh();
    resolver.AddAdd();
    resolver.AddQuantize();
    resolver.AddDequantize();

    static tflite::MicroInterpreter static_interpreter(
        g_model, resolver, g_tensor_arena, kTensorArenaSize);
    g_interpreter = &static_interpreter;

    if (g_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println(F("[CORE 1] ERROR: AllocateTensors() failed!"));
        vTaskDelete(NULL);
        return;
    }

    g_input = g_interpreter->input(0);
    g_output = g_interpreter->output(0);
    Serial.printf("[CORE 1] TFLM Engine Ready. Tensor Arena Used: %u / %u bytes.\n",
        g_interpreter->arena_used_bytes(), kTensorArenaSize);

    NanodynamicsStatePacket pkt;
    while (true) {
        if (xQueueReceive(q_infer_in, &pkt, portMAX_DELAY) == pdTRUE) {
            digitalWrite(LED_PIN, HIGH);
            int64_t t0 = esp_timer_get_time();

            // Populate normalized inputs: [x, y, z, vx, vy, vz]
            if (g_input->type == kTfLiteFloat32) {
                g_input->data.f[0] = pkt.x / POS_SCALE;
                g_input->data.f[1] = pkt.y / POS_SCALE;
                g_input->data.f[2] = pkt.z / POS_SCALE;
                g_input->data.f[3] = pkt.vx / VEL_SCALE;
                g_input->data.f[4] = pkt.vy / VEL_SCALE;
                g_input->data.f[5] = pkt.vz / VEL_SCALE;
            } else if (g_input->type == kTfLiteInt8) {
                const float scale = g_input->params.scale;
                const int32_t zero_pt = g_input->params.zero_point;
                g_input->data.int8[0] = (int8_t)constrain(roundf((pkt.x / POS_SCALE) / scale) + zero_pt, -128, 127);
                g_input->data.int8[1] = (int8_t)constrain(roundf((pkt.y / POS_SCALE) / scale) + zero_pt, -128, 127);
                g_input->data.int8[2] = (int8_t)constrain(roundf((pkt.z / POS_SCALE) / scale) + zero_pt, -128, 127);
                g_input->data.int8[3] = (int8_t)constrain(roundf((pkt.vx / VEL_SCALE) / scale) + zero_pt, -128, 127);
                g_input->data.int8[4] = (int8_t)constrain(roundf((pkt.vy / VEL_SCALE) / scale) + zero_pt, -128, 127);
                g_input->data.int8[5] = (int8_t)constrain(roundf((pkt.vz / VEL_SCALE) / scale) + zero_pt, -128, 127);
            }

            // Invoke inference
            TfLiteStatus status = g_interpreter->Invoke();
            int64_t t1 = esp_timer_get_time();
            pkt.t_infer_ms = (float)(t1 - t0) * 1e-3f;

            if (status == kTfLiteOk) {
                if (g_output->type == kTfLiteFloat32) {
                    pkt.x  = g_output->data.f[0] * POS_SCALE;
                    pkt.y  = g_output->data.f[1] * POS_SCALE;
                    pkt.z  = g_output->data.f[2] * POS_SCALE;
                    pkt.vx = g_output->data.f[3] * VEL_SCALE;
                    pkt.vy = g_output->data.f[4] * VEL_SCALE;
                    pkt.vz = g_output->data.f[5] * VEL_SCALE;
                } else if (g_output->type == kTfLiteInt8) {
                    const float scale = g_output->params.scale;
                    const int32_t zero_pt = g_output->params.zero_point;
                    pkt.x  = ((float)(g_output->data.int8[0] - zero_pt) * scale) * POS_SCALE;
                    pkt.y  = ((float)(g_output->data.int8[1] - zero_pt) * scale) * POS_SCALE;
                    pkt.z  = ((float)(g_output->data.int8[2] - zero_pt) * scale) * POS_SCALE;
                    pkt.vx = ((float)(g_output->data.int8[3] - zero_pt) * scale) * VEL_SCALE;
                    pkt.vy = ((float)(g_output->data.int8[4] - zero_pt) * scale) * VEL_SCALE;
                    pkt.vz = ((float)(g_output->data.int8[5] - zero_pt) * scale) * VEL_SCALE;
                }
                pkt.flags |= FLAG_INFER_OK;
            }

            // Forward to Core 0 for Symplectic Hamiltonian Manifold Projection
            xQueueSend(q_symplectic_in, &pkt, pdMS_TO_TICKS(5));
            digitalWrite(LED_PIN, LOW);
        }
    }
}

// ============================================================================
// CORE 0 TASK: SYMPLECTIC HAMILTONIAN MANIFOLD PROJECTION & ESP-NOW TRANSMIT
// ============================================================================
void symplecticCore0Task(void *pvParameters) {
    Serial.println(F("[CORE 0] Symplectic Hamiltonian Projection Task Online."));

    NanodynamicsStatePacket pkt;
    while (true) {
        if (xQueueReceive(q_symplectic_in, &pkt, portMAX_DELAY) == pdTRUE) {
            // Compute Symplectic Hamiltonian Manifold Projection:
            // v_{k+1} = v_NN * sqrt(2 * (E_target + mu / ||r_{k+1}||)) / ||v_NN||
            apply_symplectic_projection(&pkt, MU_EARTH);

            // Transmit corrected 64-byte state packet back to Node 1 via ESP-NOW
            esp_now_send(node1_mac, (const uint8_t*)&pkt, sizeof(NanodynamicsStatePacket));
        }
    }
}

void setup() {
    Serial.begin(115200);
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW);

    // 1. Immediately disable Bluetooth and Wi-Fi to free and de-fragment SRAM
    WiFi.mode(WIFI_OFF);
    btStop();
    Serial.println(F("[INIT] Radios halted. Maximizing contiguous SRAM block for TFLM..."));

    // 2. Initialize queues
    q_infer_in = xQueueCreate(16, sizeof(NanodynamicsStatePacket));
    q_symplectic_in = xQueueCreate(16, sizeof(NanodynamicsStatePacket));

    // 3. Configure ESP-NOW MAC radio on Wi-Fi Channel 1 without DHCP/TCP stack
    WiFi.mode(WIFI_STA);
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);

    if (esp_now_init() != ESP_OK) {
        Serial.println(F("[INIT] ERROR: ESP-NOW Init Failed!"));
        return;
    }
    esp_now_register_recv_cb(onDataRecv);

    esp_now_peer_info_t peerInfo = {};
    memcpy(peerInfo.peer_addr, node1_mac, 6);
    peerInfo.channel = WIFI_CHANNEL;
    peerInfo.encrypt = false;
    esp_now_add_peer(&peerInfo);

    // 4. Pin Tasks to Cores
    // Core 1: TFLM INT8 Inference Task
    xTaskCreatePinnedToCore(
        tflmInferenceTask,
        "TFLMInferenceCore1",
        8192,
        NULL,
        2,
        NULL,
        1
    );

    // Core 0: Symplectic Hamiltonian Projection & ESP-NOW Task
    xTaskCreatePinnedToCore(
        symplecticCore0Task,
        "SymplecticCore0",
        4096,
        NULL,
        1,
        NULL,
        0
    );
}

void loop() {
    vTaskDelay(pdMS_TO_TICKS(1000));
}

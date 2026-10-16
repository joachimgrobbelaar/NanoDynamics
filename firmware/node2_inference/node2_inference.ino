// Nanodynamics-v3 — ESP32 Node 2: Dedicated TFLite Micro INT8 PINN & Symplectic Engine
// Core 1: TensorFlow Lite for Microcontrollers (TFLM) INT8 inference (transition_model_data.h)
// Core 0: Symplectic Hamiltonian Manifold projection (hamiltonian.h) + FDIR & ESP-NOW return

#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "transition_model_data.h"
#include "hamiltonian.h"

constexpr int kTensorArenaSize = 32 * 1024;
static uint8_t tensor_arena[kTensorArenaSize];
static const tflite::Model* model = nullptr;
static tflite::MicroInterpreter* interpreter = nullptr;
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

static QueueHandle_t q_infer_in = nullptr;
static QueueHandle_t q_project_out = nullptr;
static uint8_t gateway_mac[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

void onEdgeEspNowRecv(const uint8_t* mac, const uint8_t* incomingData, int len) {
    if (len == (int)sizeof(OrbitStatePacket) && q_infer_in != nullptr) {
        memcpy(gateway_mac, mac, 6);
        OrbitStatePacket pkt;
        memcpy(&pkt, incomingData, sizeof(OrbitStatePacket));
        xQueueSendFromISR(q_infer_in, &pkt, nullptr);
    }
}

void taskPinnInferCore1(void* pvParameters) {
    OrbitStatePacket pkt;
    for (;;) {
        if (xQueueReceive(q_infer_in, &pkt, portMAX_DELAY) == pdTRUE) {
            input->data.f[0] = pkt.x;
            input->data.f[1] = pkt.y;
            input->data.f[2] = pkt.z;
            input->data.f[3] = pkt.vx;
            input->data.f[4] = pkt.vy;
            input->data.f[5] = pkt.vz;

            int64_t t0 = esp_timer_get_time();
            TfLiteStatus st = interpreter->Invoke();
            int64_t t1 = esp_timer_get_time();
            pkt.t_infer_ms = (float)(t1 - t0) * 1e-3f;

            if (st == kTfLiteOk) {
                pkt.x  = output->data.f[0];
                pkt.y  = output->data.f[1];
                pkt.z  = output->data.f[2];
                pkt.vx = output->data.f[3];
                pkt.vy = output->data.f[4];
                pkt.vz = output->data.f[5];
            }
            xQueueSend(q_project_out, &pkt, portMAX_DELAY);
        }
    }
}

void taskSymplecticProjectCore0(void* pvParameters) {
    OrbitStatePacket pkt;
    for (;;) {
        if (xQueueReceive(q_project_out, &pkt, portMAX_DELAY) == pdTRUE) {
            const float r0 = sqrtf(pkt.x * pkt.x + pkt.y * pkt.y + pkt.z * pkt.z);
            const float v0_sq = pkt.vx * pkt.vx + pkt.vy * pkt.vy + pkt.vz * pkt.vz;
            // Canonical dimensionless gravitational parameter mu* = 1.0
            const float E_target = 0.5f * v0_sq - (1.0f / fmaxf(r0, 0.5f));
            project_symplectic_hamiltonian(pkt, E_target, 1.0f);
            esp_now_send(gateway_mac, (const uint8_t*)&pkt, sizeof(OrbitStatePacket));
        }
    }
}

void setup() {
    Serial.begin(115200);
    model = tflite::GetModel(transition_tflite);
    static tflite::MicroMutableOpResolver<6> resolver;
    resolver.AddFullyConnected();
    resolver.AddTanh();
    resolver.AddAdd();

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    interpreter->AllocateTensors();
    input = interpreter->input(0);
    output = interpreter->output(0);

    WiFi.persistent(false);
    WiFi.disconnect(true);
    WiFi.mode(WIFI_STA);
    esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);

    q_infer_in = xQueueCreate(16, sizeof(OrbitStatePacket));
    q_project_out = xQueueCreate(16, sizeof(OrbitStatePacket));
    if (esp_now_init() == ESP_OK) {
        esp_now_register_recv_cb(onEdgeEspNowRecv);
    }

    xTaskCreatePinnedToCore(taskPinnInferCore1,         "TFLM_PINN_Core1",  8192, nullptr, 3, nullptr, 1);
    xTaskCreatePinnedToCore(taskSymplecticProjectCore0, "Symplectic_Core0", 4096, nullptr, 2, nullptr, 0);
}

void loop() {
    vTaskDelay(pdMS_TO_TICKS(1000));
}

#ifndef NANODYNAMICS_V3_NETWORK_CONFIG_H_
#define NANODYNAMICS_V3_NETWORK_CONFIG_H_

#include <stdint.h>

#define ND_SOFTAP_SSID      "NANODYNAMICS-ESP32-GW"
#define ND_SOFTAP_PASS      "nanodynamics"
#define ND_DEFAULT_WIFI_SSID "The Collective Genius Tutors"
#define ND_DEFAULT_HOST_IP  "192.168.1.38"
#define ND_WS_PORT          8000
#define ND_ESPNOW_CHANNEL   1

// 42-byte packed binary telemetry & PINN inference frame (CCSDS/ESP-NOW compatible)
typedef struct __attribute__((packed)) {
    uint16_t sat_id;
    float t;
    float x;
    float y;
    float z;
    float vx;
    float vy;
    float vz;
    float B_coeff;
    float dt;
    float t_infer_ms;
} OrbitStatePacket;

#endif  // NANODYNAMICS_V3_NETWORK_CONFIG_H_

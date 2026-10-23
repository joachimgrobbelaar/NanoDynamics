#pragma once
#include <stdint.h>
#include <math.h>

/**
 * Packed binary/JSON packet contract for Dual-ESP32 EdgeLink & OTA WebSocket Bridge.
 * Total packed size: 2 + 10 * 4 = 42 bytes (< 250 byte ESP-NOW frame limit).
 */
struct __attribute__((packed)) OrbitStatePacket {
    uint16_t sat_id;
    float t;
    float x, y, z;
    float vx, vy, vz;
    float B_coeff;
    float dt;
    float t_infer_ms;
};

/**
 * Symplectic Hamiltonian Manifold projection engine (Core 0 routine on Node 2):
 *   v_target = sqrt(2 * (E_target + mu / ||r_{k+1}||))
 *   v_{k+1}  = v_NN * (v_target / ||v_NN||)
 */
static inline void project_symplectic_hamiltonian(
    OrbitStatePacket& pkt,
    float E_target,
    float mu = 3.9860044e14f
) {
    const float r_norm = sqrtf(pkt.x * pkt.x + pkt.y * pkt.y + pkt.z * pkt.z);
    const float v_norm = sqrtf(pkt.vx * pkt.vx + pkt.vy * pkt.vy + pkt.vz * pkt.vz);
    if (r_norm <= 1.0f || v_norm <= 1e-6f) {
        return;
    }
    const float two_term = 2.0f * (E_target + (mu / r_norm));
    if (two_term <= 0.0f) {
        return;
    }
    const float v_target = sqrtf(two_term);
    const float scale = v_target / v_norm;
    pkt.vx *= scale;
    pkt.vy *= scale;
    pkt.vz *= scale;
}

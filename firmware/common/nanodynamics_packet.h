#pragma once
#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Packed 64-byte binary C-struct state structure for Nanodynamics v3 dual-ESP32 pipeline.
 * Transmitted over ESP-NOW locked to Wi-Fi Channel 1 with sub-2 ms latency.
 */
typedef struct __attribute__((packed)) {
    uint32_t seq_id;          // [0-3]   Monotonic sequence identifier
    uint16_t sat_id;          // [4-5]   Satellite numeric identifier
    uint16_t flags;           // [6-7]   Bitflags: 0x01=Inference OK, 0x02=Symplectic Applied
    float t;                  // [8-11]  Simulation epoch timestamp (s)
    float dt;                 // [12-15] Step time delta (s)
    float x;                  // [16-19] ECI Position X (m)
    float y;                  // [20-23] ECI Position Y (m)
    float z;                  // [24-27] ECI Position Z (m)
    float vx;                 // [28-31] ECI Velocity Vx (m/s)
    float vy;                 // [32-35] ECI Velocity Vy (m/s)
    float vz;                 // [36-39] ECI Velocity Vz (m/s)
    float e_target;           // [40-43] Specific mechanical orbital energy target (J/kg)
    float b_coeff;            // [44-47] Ballistic drag coefficient
    float t_infer_ms;         // [48-51] Hardware inference execution duration (ms)
    float battery_v;          // [52-55] Bus telemetry voltage (V)
    uint8_t reserved[8];      // [56-63] Expansion padding / spare bytes
} NanodynamicsStatePacket;

#ifdef __cplusplus
static_assert(sizeof(NanodynamicsStatePacket) == 64, "NanodynamicsStatePacket must be exactly 64 bytes");
#endif

#define FLAG_INFER_OK         (1U << 0)
#define FLAG_SYMPLECTIC_OK    (1U << 1)
#define FLAG_DEORBIT_RISK     (1U << 2)
#define FLAG_COLLISION_AVOID  (1U << 3)

/**
 * Symplectic Hamiltonian Manifold Projection routine:
 * Rescales predicted velocity vector to strictly conserve target orbital mechanical energy:
 *   v_{k+1} = v_NN * sqrt(2 * (E_target + mu / ||r_{k+1}||)) / ||v_NN||
 *
 * @param pkt       Pointer to state packet containing predicted r and v
 * @param mu        Central body gravitational parameter (default Earth: 3.986004418e14 m^3/s^2)
 */
static inline void apply_symplectic_projection(
    NanodynamicsStatePacket* pkt,
    float mu
) {
    if (!pkt) return;
    const float r_sq = pkt->x * pkt->x + pkt->y * pkt->y + pkt->z * pkt->z;
    const float v_sq = pkt->vx * pkt->vx + pkt->vy * pkt->vy + pkt->vz * pkt->vz;
    if (r_sq <= 1.0f || v_sq <= 1e-12f) return;

    const float r_norm = sqrtf(r_sq);
    const float v_norm = sqrtf(v_sq);

    const float two_term = 2.0f * (pkt->e_target + (mu / r_norm));
    if (two_term <= 0.0f) return;

    const float v_target = sqrtf(two_term);
    const float scale = v_target / v_norm;

    pkt->vx *= scale;
    pkt->vy *= scale;
    pkt->vz *= scale;
    pkt->flags |= FLAG_SYMPLECTIC_OK;
}

#ifdef __cplusplus
}
#endif

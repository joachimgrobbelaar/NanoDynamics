#ifndef NANODYNAMICS_V3_HAMILTONIAN_H_
#define NANODYNAMICS_V3_HAMILTONIAN_H_

#include <math.h>
#include <stdint.h>

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

// Fast single-precision Symplectic Hamiltonian Manifold projection on Core 0
static inline void project_symplectic_hamiltonian(
    OrbitStatePacket& pkt,
    float E_target,
    float mu
) {
    const float r_norm = sqrtf(pkt.x * pkt.x + pkt.y * pkt.y + pkt.z * pkt.z);
    const float v_nn_norm = sqrtf(pkt.vx * pkt.vx + pkt.vy * pkt.vy + pkt.vz * pkt.vz);
    if (r_norm <= 1e-4f || v_nn_norm <= 1e-9f) {
        return;
    }
    const float term = 2.0f * (E_target + (mu / r_norm));
    if (term <= 0.0f) {
        return;
    }
    const float v_target = sqrtf(term);
    const float scale = v_target / v_nn_norm;
    pkt.vx *= scale;
    pkt.vy *= scale;
    pkt.vz *= scale;
}

#endif  // NANODYNAMICS_V3_HAMILTONIAN_H_

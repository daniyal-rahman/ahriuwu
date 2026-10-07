// JAX's threefry2x32 PRNG with jax_threefry_partitionable=True (the default since JAX 0.5): keys are (k0, k1)
// uint32 pairs. split/fold_in/uniform reproduce jax.random bit for bit.
#pragma once
#include <cstdint>
#include <cstring>

namespace lanesim::rng {

struct Key { uint32_t k0, k1; };

inline uint32_t rotl(uint32_t v, int r) { return (v << r) | (v >> (32 - r)); }

// threefry2x32_p on one counter pair.
inline void threefry2x32(Key key, uint32_t x0, uint32_t x1, uint32_t* o0, uint32_t* o1) {
    static const int R[2][4] = {{13, 15, 26, 6}, {17, 29, 16, 24}};
    uint32_t ks[3] = {key.k0, key.k1, key.k0 ^ key.k1 ^ 0x1BD11BDAu};
    x0 += ks[0];
    x1 += ks[1];
    for (int i = 0; i < 5; ++i) {
        const int* rot = R[i % 2];
        for (int j = 0; j < 4; ++j) {
            x0 += x1;
            x1 = rotl(x1, rot[j]);
            x1 ^= x0;
        }
        x0 += ks[(i + 1) % 3];
        x1 += ks[(i + 2) % 3] + (uint32_t)(i + 1);
    }
    *o0 = x0, *o1 = x1;
}

// jax.random.PRNGKey(seed) for a 32-bit seed.
inline Key from_seed(uint32_t seed) { return {0u, seed}; }

// jax.random.split(key, n)[i]: threefry over the counter pair (0, i).
inline Key split(Key key, uint32_t i) {
    Key out;
    threefry2x32(key, 0u, i, &out.k0, &out.k1);
    return out;
}

// jax.random.fold_in(key, data) for 32-bit data.
inline Key fold_in(Key key, uint32_t data) {
    Key out;
    threefry2x32(key, 0u, data, &out.k0, &out.k1);
    return out;
}

// jax.random.uniform(key, shape)[i] in [0, 1), float32.
inline float uniform(Key key, uint32_t i) {
    uint32_t a, b;
    threefry2x32(key, 0u, i, &a, &b);
    uint32_t bits = ((a ^ b) >> 9) | 0x3F800000u;
    float f;
    std::memcpy(&f, &bits, 4);
    return f - 1.f;
}

}  // namespace lanesim::rng

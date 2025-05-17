// gpu_kernel.cu
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// BIP39 wordlist in constant memory with new GpuWord format
struct GpuWord {
    char bytes[10];  // 9 chars + null terminator
    uint8_t len;
};

__device__ __constant__ GpuWord wordlist[2048];

// SHA-512 Constants
__device__ __constant__ uint64_t K[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

// SHA-256 Constants
__device__ __constant__ uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// secp256k1 Field Prime
__device__ __constant__ uint64_t SECP256K1_P[4] = {
    0xFFFFFFFFFFFFFC2FULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL
};

// secp256k1 p-2 for modular inversion
__device__ __constant__ uint64_t EXP_P_MINUS_2[4] = {
    0xFFFFFFFFFFFFFC2DULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL
};

// Macros
__device__ __forceinline__ uint64_t ROTR(uint64_t x, uint64_t n) {
    return (x >> n) | (x << (64 - n));
}

__device__ __forceinline__ uint32_t ROTR32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

// KECCAK-256 (defined early to resolve undefined identifier)
__device__ void keccak_f(uint64_t state[25]) {
    const uint64_t keccakf_rndc[24] = {
        0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
        0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
        0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
        0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
        0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
        0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
        0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
        0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
    };
    const int keccakf_rotc[24] = {
         1,  3,  6, 10, 15, 21,
         28, 36, 45, 55,  2, 14,
         27, 41, 56, 8, 25, 43,
         62, 18, 39, 61, 20, 44
    };
    const int keccakf_piln[24] = {
         10, 7, 11, 17, 18, 3, 5, 16,
         8, 21, 24, 4, 15, 23, 19, 13,
         12, 2, 20, 14, 22, 9, 6, 1
    };
    for (int round = 0; round < 24; round++) {
        uint64_t bc[5];
        for (int i = 0; i < 5; i++) {
            bc[i] = state[i] ^ state[i+5] ^ state[i+10] ^ state[i+15] ^ state[i+20];
        }
        for (int i = 0; i < 5; i++) {
            uint64_t t = bc[(i+4)%5] ^ ROTR(bc[(i+1)%5], 1);
            for (int j = 0; j < 25; j += 5) {
                state[j+i] ^= t;
            }
        }
        uint64_t t = state[1];
        for (int i = 0; i < 24; i++) {
            int j = keccakf_piln[i];
            bc[0] = state[j];
            state[j] = ROTR(t, keccakf_rotc[i]);
            t = bc[0];
        }
        for (int j = 0; j < 5; j++) {
            bc[j] = state[j] ^ state[j+5] ^ state[j+10] ^ state[j+15] ^ state[j+20];
        }
        for (int j = 0; j < 5; j++) {
            for (int i = 0; i < 25; i += 5) {
                state[i+j] ^= (~bc[(j+1)%5]) & bc[(j+2)%5];
            }
        }
        state[0] ^= keccakf_rndc[round];
    }
}

__device__ void keccak256(const uint8_t* input, int inputlen, uint8_t* output) {
    uint64_t state[25] = {0};
    for (int i = 0; i < inputlen; i++) {
        int idx = i / 8;
        int shift = (i % 8) * 8;
        state[idx] ^= (uint64_t)input[i] << shift;
    }
    state[inputlen / 8] ^= (uint64_t)0x01ULL << ((inputlen % 8) * 8);
    state[16] ^= 0x8000000000000000ULL;
    keccak_f(state);
    for (int i = 0; i < 32; i++) {
        output[i] = (state[i/8] >> ((i%8)*8)) & 0xFF;
    }
}

// SHA-256 Implementation
__device__ void sha256_transform(uint32_t* state, const uint8_t* block) {
    uint32_t W[64];
    uint32_t a, b, c, d, e, f, g, h;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = ((uint32_t)block[i * 4 + 0] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               ((uint32_t)block[i * 4 + 3]);
    }
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ROTR32(W[i-15], 7) ^ ROTR32(W[i-15], 18) ^ (W[i-15] >> 3);
        uint32_t s1 = ROTR32(W[i-2], 17) ^ ROTR32(W[i-2], 19) ^ (W[i-2] >> 10);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + K256[i] + W[i];
        uint32_t S0 = ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

__device__ void sha256(const uint8_t* data, int datalen, uint8_t* out) {
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    uint8_t block[64] = {0};
    for (int i = 0; i < datalen; i++) {
        block[i] = data[i];
    }
    block[datalen] = 0x80;
    block[62] = (datalen * 8) >> 8;
    block[63] = (datalen * 8);
    sha256_transform(state, block);
    for (int i = 0; i < 8; i++) {
        out[i*4 + 0] = (state[i] >> 24) & 0xff;
        out[i*4 + 1] = (state[i] >> 16) & 0xff;
        out[i*4 + 2] = (state[i] >> 8) & 0xff;
        out[i*4 + 3] = (state[i] >> 0) & 0xff;
    }
}

// Field Arithmetic Functions
__device__ void fe_set(uint64_t r[4], const uint64_t a[4]) {
    for (int i = 0; i < 4; i++) {
        r[i] = a[i];
    }
}

__device__ int fe_cmp(const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 3; i >= 0; i--) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

__device__ void fe_sub_inv(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t diff = a[i] - b[i] - borrow;
        borrow = (diff > a[i]) || (diff == a[i] && borrow);
        r[i] = diff;
    }
    if (borrow) {
        uint64_t carry = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t sum = r[i] + SECP256K1_P[i] + carry;
            carry = sum < r[i];
            r[i] = sum;
        }
    }
}

__device__ void fe_reduce(uint64_t r[4], const uint64_t a[8]) {
    uint64_t tmp[4];
    for (int i = 0; i < 4; i++) {
        tmp[i] = a[i];
    }
    while (fe_cmp(tmp, SECP256K1_P) >= 0) {
        fe_sub_inv(tmp, tmp, SECP256K1_P);
    }
    fe_set(r, tmp);
}

__device__ void fe_mul(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t tmp[8] = {0};
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint64_t hi, lo;
            asm("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a[i]), "l"(b[j]));
            asm("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a[i]), "l"(b[j]));
            tmp[i + j] += lo;
            tmp[i + j + 1] += hi;
        }
    }
    while (true) {
        bool greater = false;
        for (int i = 7; i >= 0; i--) {
            if (tmp[i] > (i < 4 ? SECP256K1_P[i] : 0)) { greater = true; break; }
            if (tmp[i] < (i < 4 ? SECP256K1_P[i] : 0)) break;
        }
        if (!greater) break;
        uint64_t borrow = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t sub = (i < 4 ? SECP256K1_P[i] : 0) + borrow;
            borrow = tmp[i] < sub;
            tmp[i] -= sub;
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) r[i] = tmp[i];
}

__device__ void fe_mul_sq(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    uint64_t tmp[8] = {0};
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t hi, lo;
            asm("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a[i]), "l"(b[j]));
            asm("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a[i]), "l"(b[j]));
            int k = i + j;
            uint64_t sum = tmp[k] + lo + carry;
            carry = (sum < tmp[k]) + hi;
            tmp[k] = sum;
            if (k + 1 < 8) {
                tmp[k + 1] += carry;
                carry = tmp[k + 1] < carry;
            }
        }
    }
    fe_reduce(r, tmp);
}

__device__ void fe_square(uint64_t r[4], const uint64_t a[4]) {
    fe_mul_sq(r, a, a);
}

__device__ void fe_pow(uint64_t r[4], const uint64_t base[4], const uint64_t exp[4]) {
    uint64_t result[4] = {1, 0, 0, 0};
    uint64_t temp[4];
    fe_set(temp, base);
    for (int i = 3; i >= 0; i--) {
        uint64_t word = exp[i];
        for (int j = 63; j >= 0; j--) {
            fe_square(temp, temp);
            if (word & (1ULL << j)) {
                fe_mul(result, result, temp);
            }
        }
    }
    fe_set(r, result);
}

__device__ void fe_inv(uint64_t r[4], const uint64_t a[4]) {
    fe_pow(r, a, EXP_P_MINUS_2);
}

// Helper: Add two field elements modulo P
__device__ void fe_add(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        r[i] = a[i] + b[i];
    }
    bool carry = (r[0] < a[0]);
    for (int i = 1; i < 4; i++) {
        if (carry) {
            r[i]++;
            carry = (r[i] == 0);
        }
    }
    bool greater_or_equal = true;
    for (int i = 3; i >= 0; i--) {
        if (r[i] < SECP256K1_P[i]) break;
        if (r[i] > SECP256K1_P[i]) {
            greater_or_equal = false;
            break;
        }
    }
    if (greater_or_equal) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            r[i] -= SECP256K1_P[i];
        }
    }
}

// Helper: Subtract two field elements modulo P
__device__ void fe_sub(uint64_t r[4], const uint64_t a[4], const uint64_t b[4]) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        r[i] = a[i] - b[i];
    }
    bool borrow = (a[0] < b[0]);
    for (int i = 1; i < 4; i++) {
        if (borrow) {
            if (a[i] == 0) {
                r[i] = (uint64_t)-1 - (b[i] - 1);
            } else {
                r[i] = a[i] - b[i] - 1;
            }
            borrow = (a[i] <= b[i]);
        }
    }
    if (borrow) {
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            r[i] += SECP256K1_P[i];
        }
    }
}

// Elliptic Curve: Point Doubling (P = 2P)
__device__ void ec_double(uint64_t rx[4], uint64_t ry[4], const uint64_t px[4], const uint64_t py[4]) {
    uint64_t t0[4], t1[4], t2[4], t3[4];
    fe_mul(t0, px, px);     // t0 = px^2
    fe_add(t1, t0, t0);     // t1 = 2 * px^2
    fe_add(t1, t1, t0);     // t1 = 3 * px^2
    fe_add(t2, py, py);     // t2 = 2 * py
    fe_inv(t3, t2);         // t3 = 1 / (2 * py)
    fe_mul(t1, t1, t3);     // t1 = λ = (3 * px^2) / (2 * py)
    fe_mul(t3, t1, t1);     // t3 = λ^2
    fe_add(t0, px, px);     // t0 = 2 * px
    fe_sub(rx, t3, t0);     // rx = λ^2 - 2 * px
    fe_sub(t0, px, rx);     // t0 = px - x3
    fe_mul(t0, t1, t0);     // t0 = λ * (px - x3)
    fe_sub(ry, t0, py);     // ry = λ * (px - x3) - py
}

// Elliptic Curve: Point Addition (R = P + Q)
__device__ void ec_add(uint64_t rx[4], uint64_t ry[4], const uint64_t px[4], const uint64_t py[4], const uint64_t qx[4], const uint64_t qy[4]) {
    uint64_t t0[4], t1[4], t2[4], t3[4];
    fe_sub(t0, qy, py);    // t0 = qy - py
    fe_sub(t1, qx, px);    // t1 = qx - px
    fe_inv(t2, t1);        // t2 = 1 / (qx - px)
    fe_mul(t0, t0, t2);    // t0 = λ = (qy - py) / (qx - px)
    fe_mul(t2, t0, t0);    // t2 = λ^2
    fe_sub(t3, t2, px);    // t3 = λ^2 - px
    fe_sub(rx, t3, qx);    // rx = λ^2 - px - qx
    fe_sub(t2, px, rx);    // t2 = px - x3
    fe_mul(t2, t0, t2);    // t2 = λ(px - x3)
    fe_sub(ry, t2, py);    // ry = λ(px - x3) - py
}

// Scalar Multiplication: privkey * G
__device__ void ec_scalar_mul(uint64_t rx[4], uint64_t ry[4], const uint64_t privkey[4]) {
    uint64_t gx[4] = { 
        0x79BE667EF9DCBBACULL, 
        0x55A06295CE870B07ULL, 
        0x029BFCDB2DCE28D9ULL, 
        0x59F2815B16F81798ULL 
    };
    uint64_t gy[4] = {
        0x483ADA7726A3C465ULL,
        0x5DA4FBFC0E1108A8ULL,
        0xFD17B448A6855419ULL,
        0x9C47D08FFB10D4B8ULL
    };
    uint64_t rx0[4] = {0};
    uint64_t ry0[4] = {0};
    for (int bit = 255; bit >= 0; bit--) {
        ec_double(rx0, ry0, rx0, ry0);
        int word = bit / 64;
        int bit_in_word = bit % 64;
        if ((privkey[word] >> bit_in_word) & 1) {
            ec_add(rx0, ry0, rx0, ry0, gx, gy);
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        rx[i] = rx0[i];
        ry[i] = ry0[i];
    }
}

// Serialize compressed public key (33 bytes: 0x02 or 0x03 + X coordinate)
__device__ void serialize_pubkey_compressed(uint8_t out[33], const uint64_t x[4], const uint64_t y[4]) {
    out[0] = (y[0] & 1) ? 0x03 : 0x02;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        out[1 + i * 8 + 0] = (x[i] >> 56) & 0xff;
        out[1 + i * 8 + 1] = (x[i] >> 48) & 0xff;
        out[1 + i * 8 + 2] = (x[i] >> 40) & 0xff;
        out[1 + i * 8 + 3] = (x[i] >> 32) & 0xff;
        out[1 + i * 8 + 4] = (x[i] >> 24) & 0xff;
        out[1 + i * 8 + 5] = (x[i] >> 16) & 0xff;
        out[1 + i * 8 + 6] = (x[i] >> 8) & 0xff;
        out[1 + i * 8 + 7] = (x[i] >> 0) & 0xff;
    }
}

// Generate Ethereum address from pubkey
__device__ void pubkey_to_eth_address(uint8_t address_out[20], const uint64_t pubkey_x[4], const uint64_t pubkey_y[4]) {
    uint8_t serialized[33];
    uint8_t hash[32];
    serialize_pubkey_compressed(serialized, pubkey_x, pubkey_y);
    keccak256(serialized, 33, hash);
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        address_out[i] = hash[12 + i];
    }
}

// SHA512 Transform
__device__ void sha512_transform(uint64_t* state, const uint8_t* block) {
    uint64_t W[80];
    uint64_t a, b, c, d, e, f, g, h;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        W[i] = ((uint64_t)block[i * 8 + 0] << 56) |
               ((uint64_t)block[i * 8 + 1] << 48) |
               ((uint64_t)block[i * 8 + 2] << 40) |
               ((uint64_t)block[i * 8 + 3] << 32) |
               ((uint64_t)block[i * 8 + 4] << 24) |
               ((uint64_t)block[i * 8 + 5] << 16) |
               ((uint64_t)block[i * 8 + 6] << 8) |
               ((uint64_t)block[i * 8 + 7]);
    }
    #pragma unroll
    for (int i = 16; i < 80; i++) {
        uint64_t s0 = ROTR(W[i-15],1) ^ ROTR(W[i-15],8) ^ (W[i-15] >> 7);
        uint64_t s1 = ROTR(W[i-2],19) ^ ROTR(W[i-2],61) ^ (W[i-2] >> 6);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];
    #pragma unroll
    for (int i = 0; i < 80; i++) {
        uint64_t S1 = ROTR(e,14) ^ ROTR(e,18) ^ ROTR(e,41);
        uint64_t ch = (e & f) ^ ((~e) & g);
        uint64_t temp1 = h + S1 + ch + K[i] + W[i];
        uint64_t S0 = ROTR(a,28) ^ ROTR(a,34) ^ ROTR(a,39);
        uint64_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint64_t temp2 = S0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

// SHA512 wrapper
__device__ void sha512(const uint8_t* data, int datalen, uint8_t* out) {
    uint64_t state[8] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
        0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
        0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
    };
    sha512_transform(state, data);
    for (int i = 0; i < 8; i++) {
        out[i*8 + 0] = (state[i] >> 56) & 0xff;
        out[i*8 + 1] = (state[i] >> 48) & 0xff;
        out[i*8 + 2] = (state[i] >> 40) & 0xff;
        out[i*8 + 3] = (state[i] >> 32) & 0xff;
        out[i*8 + 4] = (state[i] >> 24) & 0xff;
        out[i*8 + 5] = (state[i] >> 16) & 0xff;
        out[i*8 + 6] = (state[i] >> 8) & 0xff;
        out[i*8 + 7] = (state[i] >> 0) & 0xff;
    }
}

// HMAC-SHA512
__device__ void hmac_sha512(const uint8_t* key, int key_len, const uint8_t* message, int message_len, uint8_t* output) {
    uint8_t k_ipad[128] = {0};
    uint8_t k_opad[128] = {0};
    uint8_t inner_hash[64];
    if (key_len > 128) {
        sha512(key, key_len, k_ipad);
        key = k_ipad;
        key_len = 64;
    }
    for (int i = 0; i < key_len; i++) {
        k_ipad[i] = key[i] ^ 0x36;
        k_opad[i] = key[i] ^ 0x5c;
    }
    for (int i = key_len; i < 128; i++) {
        k_ipad[i] = 0x36;
        k_opad[i] = 0x5c;
    }
    uint8_t inner_data[256];
    for (int i = 0; i < 128; i++) inner_data[i] = k_ipad[i];
    for (int i = 0; i < message_len; i++) inner_data[128+i] = message[i];
    sha512(inner_data, 128+message_len, inner_hash);
    uint8_t outer_data[192];
    for (int i = 0; i < 128; i++) outer_data[i] = k_opad[i];
    for (int i = 0; i < 64; i++) outer_data[128+i] = inner_hash[i];
    sha512(outer_data, 192, output);
}

// PBKDF2-HMAC-SHA512
__device__ void pbkdf2_hmac_sha512(const uint8_t* password, int password_len, const uint8_t* salt, int salt_len, int iterations, uint8_t* output, int dklen) {
    uint8_t U[64];
    uint8_t T[64];
    for (int block_index = 1; dklen > 0; block_index++) {
        uint8_t salt_block[256];
        for (int i = 0; i < salt_len; i++) {
            salt_block[i] = salt[i];
        }
        salt_block[salt_len + 0] = (block_index >> 24) & 0xFF;
        salt_block[salt_len + 1] = (block_index >> 16) & 0xFF;
        salt_block[salt_len + 2] = (block_index >> 8) & 0xFF;
        salt_block[salt_len + 3] = (block_index) & 0xFF;
        hmac_sha512(password, password_len, salt_block, salt_len+4, U);
        for (int i = 0; i < 64; i++) {
            T[i] = U[i];
        }
        for (int j = 1; j < iterations; j++) {
            hmac_sha512(password, password_len, U, 64, U);
            for (int i = 0; i < 64; i++) {
                T[i] ^= U[i];
            }
        }
        int copy_len = dklen < 64 ? dklen : 64;
        for (int i = 0; i < copy_len; i++) {
            output[i] = T[i];
        }
        output += copy_len;
        dklen -= copy_len;
    }
}

// Helper: Convert 4 bytes to 32 bits
__device__ uint32_t to_u32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) |
           ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8)  |
           ((uint32_t)p[3]);
}

// Update the word indices structure to ensure proper alignment
struct WordIndices {
    uint16_t indices[12];
};

// Update the validate_bip39_checksum function signature
__device__ bool validate_bip39_checksum(const WordIndices* word_indices) {
    uint8_t entropy[16] = {0};
    for (int i = 0; i < 128; i++) {
        int word_idx = i / 11;
        int bit_idx = 10 - (i % 11);
        if (word_indices->indices[word_idx] & (1 << bit_idx)) {
            entropy[i / 8] |= (1 << (7 - (i % 8)));
        }
    }
    uint8_t hash[32];
    sha256(entropy, 16, hash);
    uint8_t checksum_expected = hash[0] >> 4;
    uint8_t checksum_actual = word_indices->indices[11] & 0xF;
    return checksum_expected == checksum_actual;
}

__device__ void fe_mod_exp(uint64_t r[4], const uint64_t base[4], const uint64_t exp[4]) {
    uint64_t result[4] = {0, 0, 0, 0};
    result[0] = 1; // Initialize result to 1

    uint64_t base_copy[4];
    fe_set(base_copy, base);

    for (int i = 3; i >= 0; i--) {
        for (int bit = 63; bit >= 0; bit--) {
            // Square result
            fe_square(result, result);

            if ((exp[i] >> bit) & 1) {
                fe_mul(result, result, base_copy);
            }
        }
    }
    fe_set(r, result);
}
__global__ void kernel_sha256(const uint8_t* inputs, int input_len, uint8_t* outputs, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        sha256(&inputs[idx * input_len], input_len, &outputs[idx * 32]);
    }
}

__global__ void kernel_keccak256(const uint8_t* inputs, int input_len, uint8_t* outputs, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        keccak256(&inputs[idx * input_len], input_len, &outputs[idx * 32]);
    }
}

__device__ void fe_mod_inverse(uint64_t r[4], const uint64_t a[4]) {
    // Modular inverse: a^(p-2) mod p
    fe_mod_exp(r, a, EXP_P_MINUS_2);
}


__global__ void kernel_wordlist_lookup(char* candidates, int candidate_count, int* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < candidate_count) {
        char* candidate = &candidates[idx * 10]; // max 10 chars per word
        results[idx] = -1;
        for (int i = 0; i < 2048; i++) {
            bool match = true;
            for (int j = 0; j < 10; j++) {
                if (candidate[j] != wordlist[i].bytes[j]) {
                    match = false;
                    break;
                }
                if (candidate[j] == '\0') break;
            }
            if (match) {
                results[idx] = i; // index found
                break;
            }
        }
    }
}

__device__ int get_bit(const int* word_indices, int phrase_idx, int bit_pos, int phrase_len) {
    // Each word is 11 bits
    int word_bit_index = bit_pos / 11;
    int bit_in_word = 10 - (bit_pos % 11); // MSB first in each 11-bit word
    int word = word_indices[phrase_idx * phrase_len + word_bit_index];
    return (word >> bit_in_word) & 1;
}

// Main kernel
__global__ void kernel_validate_mnemonic(int* word_indices, int phrase_len, int phrase_count, int* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= phrase_count) return;

    // Calculate total bits = phrase_len * 11
    int total_bits = phrase_len * 11;
    int checksum_bits = total_bits / 33;
    int entropy_bits = total_bits - checksum_bits;
    int entropy_bytes = entropy_bits / 8;

    // Extract entropy bytes from word indices bits
    uint8_t entropy[32] = {0}; // max 32 bytes entropy (for 24 words)
    #pragma unroll
    for (int i = 0; i < entropy_bits; i++) {
        int bit = get_bit(word_indices, idx, i, phrase_len);
        entropy[i / 8] |= bit << (7 - (i % 8));
    }

    // Calculate SHA-256 of entropy
    uint8_t hash[32];
    sha256(entropy, entropy_bytes, hash);

    // Extract checksum bits from phrase
    int valid = 1;
    #pragma unroll
    for (int i = 0; i < checksum_bits; i++) {
        int checksum_bit = get_bit(word_indices, idx, entropy_bits + i, phrase_len);
        int hash_bit = (hash[0] >> (7 - i)) & 1;
        if (checksum_bit != hash_bit) {
            valid = 0;
            break;
        }
    }

    results[idx] = valid;
}


// Main GPU Kernel
extern "C" __global__ void search_seeds(
    uint64_t* d_seeds_tested,
    uint64_t* d_seeds_found,
    uint64_t start_offset,
    uint64_t batch_size,
    int wordlist_len,
    int known_count,
    const int* d_known,
    const uint8_t* d_address,
    int match_mode,
    int match_prefix_len,
    int* d_found_mnemonic
) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;
    uint64_t idx = start_offset + tid;
    
    // Use properly aligned structure
    WordIndices word_indices;
    #pragma unroll
    for (int i = 0; i < known_count; i++) {
        word_indices.indices[i] = (uint16_t)d_known[i];
    }
    uint64_t rem = idx;
    for (int pos = known_count; pos < 12; pos++) {
        word_indices.indices[pos] = (uint16_t)(rem % 2048);
        rem /= 2048;
    }

    // Validate checksum with proper alignment
    if (!validate_bip39_checksum(&word_indices)) {
        atomicAdd((unsigned long long*)d_seeds_tested, 1ULL);
        return;
    }

    uint8_t mnemonic[256];
    int mnemonic_len = 0;
    #pragma unroll
    for (int i = 0; i < 12; i++) {
        const char* w = wordlist[word_indices.indices[i]].bytes;
        for (int j = 0; w[j] != '\0'; j++) {
            mnemonic[mnemonic_len++] = w[j];
        }
        if (i < 11) mnemonic[mnemonic_len++] = ' ';
    }
    mnemonic[mnemonic_len] = '\0';
    uint8_t seed[64];
    pbkdf2_hmac_sha512(
        mnemonic,
        mnemonic_len,
        (const uint8_t*)"mnemonic",
        8,
        2048,
        seed,
        64
    );
    uint8_t private_key[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        private_key[i] = seed[i];
    }
    uint64_t priv_scalar[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        priv_scalar[i] = 
            ((uint64_t)private_key[i*8 + 0] << 56) |
            ((uint64_t)private_key[i*8 + 1] << 48) |
            ((uint64_t)private_key[i*8 + 2] << 40) |
            ((uint64_t)private_key[i*8 + 3] << 32) |
            ((uint64_t)private_key[i*8 + 4] << 24) |
            ((uint64_t)private_key[i*8 + 5] << 16) |
            ((uint64_t)private_key[i*8 + 6] << 8) |
            ((uint64_t)private_key[i*8 + 7] << 0);
    }
    uint64_t pubkey_x[4], pubkey_y[4];
    ec_scalar_mul(pubkey_x, pubkey_y, priv_scalar);
    uint8_t eth_addr[20];
    pubkey_to_eth_address(eth_addr, pubkey_x, pubkey_y);

    // Add bounds checking for match_prefix_len
    int actual_prefix_len = (match_prefix_len < 0) ? 0 : 
                           (match_prefix_len > 20) ? 20 : 
                           match_prefix_len;

    bool match = true;
    if (match_mode == 0) {
        for (int i = 0; i < 20; i++)
            if (eth_addr[i] != d_address[i]) { match = false; break; }
    } else if (match_mode == 1) {
        for (int i = 0; i < actual_prefix_len; i++)
            if (eth_addr[i] != d_address[i]) { match = false; break; }
    } else {
        for (int i = 0; i < 20; i++)
            if (eth_addr[i] != 0x00) { match = false; break; }
    }
    if (match) {
        printf("[FOUND] Mnemonic: %s\n", mnemonic);
        atomicAdd((unsigned long long*)d_seeds_found, 1ULL);
        d_found_mnemonic[tid] = 1;
    }
    atomicAdd((unsigned long long*)d_seeds_tested, 1ULL);
}
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define TRUNC_BITS 32  // Use 32 bits for truncation

// --------------------
// Utility functions for SHA-256

__device__ __forceinline__ unsigned int rotr(unsigned int x, unsigned int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ unsigned int Ch(unsigned int x, unsigned int y, unsigned int z) {
    return (x & y) ^ ((~x) & z);
}

__device__ __forceinline__ unsigned int Maj(unsigned int x, unsigned int y, unsigned int z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

__device__ __forceinline__ unsigned int Sigma0(unsigned int x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

__device__ __forceinline__ unsigned int Sigma1(unsigned int x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

__device__ __forceinline__ unsigned int sigma0(unsigned int x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ unsigned int sigma1(unsigned int x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

// --------------------
// SHA-256 constants in constant memory

__constant__ unsigned int K[64] = {
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
  0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
  0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
  0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
  0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
  0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
  0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// --------------------
// SHA-256 computation on a single 512-bit block (with padding)

__device__ void sha256(const unsigned char* msg, int msg_len, unsigned int* hash_out) {
    // Initial hash values as defined in SHA-256
    unsigned int H[8] = {
        0x6a09e667,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19
    };

    // Prepare a 64-byte block and pad the message.
    unsigned char block[64] = {0};
    for (int i = 0; i < msg_len && i < 64; i++) {
        block[i] = msg[i];
    }
    if (msg_len < 64) {
        block[msg_len] = 0x80;
    }
    if (msg_len <= 55) {
        unsigned long long bit_len = (unsigned long long)msg_len * 8;
        for (int i = 0; i < 8; i++) {
            block[63 - i] = (unsigned char)(bit_len >> (8 * i));
        }
    }
    
    // Prepare message schedule array W[0..63]
    unsigned int W[64];
    for (int i = 0; i < 16; i++) {
        W[i] = ((unsigned int)block[i*4] << 24) |
               ((unsigned int)block[i*4+1] << 16) |
               ((unsigned int)block[i*4+2] << 8) |
               ((unsigned int)block[i*4+3]);
    }
    for (int t = 16; t < 64; t++) {
        W[t] = sigma1(W[t-2]) + W[t-7] + sigma0(W[t-15]) + W[t-16];
    }
    
    unsigned int a = H[0], b = H[1], c = H[2], d = H[3];
    unsigned int e = H[4], f = H[5], g = H[6], h = H[7];
    
    for (int t = 0; t < 64; t++) {
        unsigned int T1 = h + Sigma1(e) + Ch(e, f, g) + K[t] + W[t];
        unsigned int T2 = Sigma0(a) + Maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }
    
    H[0] += a; H[1] += b; H[2] += c; H[3] += d;
    H[4] += e; H[5] += f; H[6] += g; H[7] += h;
    
    for (int i = 0; i < 8; i++) {
        hash_out[i] = H[i];
    }
}

// --------------------
// Kernel: Compute SHA-256 hash for each key and truncate to 32 bits

extern "C" __global__ void sha256_hash(const unsigned char* keys, uint64_t* hashes, int num_keys, int key_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_keys) return;
    
    int bytes_per_key = (key_length + 7) / 8;
    const unsigned char* key_ptr = keys + idx * bytes_per_key;
    
    unsigned int hash[8];
    sha256(key_ptr, bytes_per_key, hash);
    
    // Truncate to 32 bits: take the first 32 bits (word 0)
#if TRUNC_BITS == 32
    uint32_t truncated = hash[0];
    hashes[idx] = (uint64_t) truncated;
#elif TRUNC_BITS == 64
    uint64_t truncated = ((uint64_t)hash[0] << 32) | hash[1];
    hashes[idx] = truncated;
#else
    hashes[idx] = 0;
#endif

    if (idx == 0) {
        printf("Kernel launched successfully! Key Length: %d, Truncated to %d bits\n", key_length, TRUNC_BITS);
    }
}


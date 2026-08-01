// androidllm neon kernel: fp16 matrix multiply for aarch64 (Android/Termux).
// Build: see scripts/build_neon.sh
// Signature: void matmul_f16_f16(const uint16_t* a, const uint16_t* b,
//                                uint16_t* out, int m, int k, int n)
// Computes out[m,n] = a[m,k] * b[n,k]^T  (row-major, fp16 bit layout).
// Each 16-bit lane holds a raw IEEE fp16; the file is compiled with
// -ffast-math so signed zeros / denormals are flushed, keeping lane math
// equivalent to a plain float multiply.

#include <arm_neon.h>
#include <stdint.h>
#include <string.h>

typedef uint16_t half;

static inline float half_to_f32(half h) {
    return vget_lane_f32(vcvt_f32_f16(vdup_n_f16(*(const __fp16 *)&h)), 0);
}

static inline half f32_to_half(float f) {
    __fp16 h = (__fp16)f;
    half out;
    memcpy(&out, &h, sizeof(out));
    return out;
}

static void kern(const half *a, const half *b, half *out, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        const half *arow = a + (size_t)i * k;
        for (int j = 0; j < n; j++) {
            const half *brow = b + (size_t)j * k;
            float acc = 0.0f;
            int t = 0;
            for (; t + 8 <= k; t += 8) {
                float16x8_t av = vld1q_f16((const __fp16 *)(arow + t));
                float16x8_t bv = vld1q_f16((const __fp16 *)(brow + t));
                float32x4_t lo = vmulq_f32(vcvt_f32_f16(vget_low_f16(av)),
                                           vcvt_f32_f16(vget_low_f16(bv)));
                float32x4_t hi = vmulq_f32(vcvt_f32_f16(vget_high_f16(av)),
                                           vcvt_f32_f16(vget_high_f16(bv)));
                acc += vaddvq_f32(lo) + vaddvq_f32(hi);
            }
            for (; t < k; t++) {
                acc += half_to_f32(arow[t]) * half_to_f32(brow[t]);
            }
            out[(size_t)i * n + j] = f32_to_half(acc);
        }
    }
}

void matmul_f16_f16(const half *a, const half *b, half *out, int m, int k, int n) {
    kern(a, b, out, m, k, n);
}

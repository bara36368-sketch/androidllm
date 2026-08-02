// androidllm neon kernel: multithreaded fp16 matrix multiply for aarch64.
// Build: scripts/build_neon.sh  (needs clang + pthreads from Termux)
//
// void matmul_f16_f16(const uint16_t* a, const uint16_t* b,
//                     uint16_t* out, int m, int k, int n, int n_threads)
// Computes out[m,n] = a[m,k] * b[n,k]^T (row-major, raw IEEE fp16 lanes).
// Compiled with -ffast-math so denormals/signed zeros flush; every lane
// math is a plain float multiply, so results match the numpy fallback.
//
// Tiling: the output is partitioned into contiguous column blocks, one
// per thread (the m=1 GEMV case makes columns fully independent). The
// main thread does its block inline; workers are created per call with
// pthread_create. With n_threads <= 1 the pthread path is skipped.

#include <arm_neon.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>

typedef uint16_t half;

typedef struct {
    const half *a, *b;
    half *out;
    int m, k, n;
    int j0, j1;
} job_t;

static inline float half_to_f32(half h) {
    return vget_lane_f32(vcvt_f32_f16(vdup_n_f16(*(const __fp16 *)&h)), 0);
}

static inline half f32_to_half(float f) {
    __fp16 h = (__fp16)f;
    half out;
    memcpy(&out, &h, sizeof(out));
    return out;
}

static void kern(const half *a, const half *b, half *out, int m, int k, int n,
                 int j0, int j1) {
    for (int i = 0; i < m; i++) {
        const half *arow = a + (size_t)i * k;
        for (int j = j0; j < j1; j++) {
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

static void *worker(void *arg) {
    job_t *jb = (job_t *)arg;
    kern(jb->a, jb->b, jb->out, jb->m, jb->k, jb->n, jb->j0, jb->j1);
    return NULL;
}

void matmul_f16_f16(const half *a, const half *b, half *out, int m, int k,
                    int n, int n_threads) {
    if (n_threads > n) {
        n_threads = n;
    }
    if (n_threads <= 1 || m <= 0 || n <= 0) {
        kern(a, b, out, m, k, n, 0, n);
        return;
    }
    pthread_t threads[8];
    job_t jobs[8];
    if (n_threads > 8) {
        n_threads = 8;
    }
    int base = n / n_threads;
    int rem = n % n_threads;
    int cur = 0;
    for (int t = 0; t < n_threads; t++) {
        int span = base + (t < rem ? 1 : 0);
        jobs[t] = (job_t){a, b, out, m, k, n, cur, cur + span};
        cur += span;
    }
    for (int t = 1; t < n_threads; t++) {
        pthread_create(&threads[t], NULL, worker, &jobs[t]);
    }
    worker(&jobs[0]);
    for (int t = 1; t < n_threads; t++) {
        pthread_join(threads[t], NULL);
    }
}

#include <immintrin.h>
#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {

    const int p = K / 2;
    const int in_stride = W + 2 * p;

    const int tile_H = 16;
    const int tile_W = 128;
    for (int oy1 = 0; oy1 < H; oy1 += tile_H) {
        for (int ox1 = 0; ox1 < W; ox1 += tile_W) {
            const int oy_mx = (oy1 + tile_H < H) ? oy1 + tile_H : H;
            const int ox_mx = (ox1 + tile_W < W) ? ox1 + tile_W : W;
            for (int oy = oy1; oy < oy_mx; ++oy) {
                int ox = ox1;
                for (; ox + 31 < ox_mx; ox += 32) {
                    __m256 acc0 = _mm256_setzero_ps();
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();
                    for (int ky = 0; ky < K; ++ky) {
                        const float* in_r = &in[(oy + ky) * in_stride];
                        const float* ker_r = &ker[ky * K];
                        for (int kx = 0; kx < K; ++kx) {
                            const __m256 vk = _mm256_set1_ps(ker_r[kx]);
                            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(&in_r[ox + kx]),vk,acc0);
                            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(&in_r[ox + kx + 8]),vk,acc1);
                            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(&in_r[ox + kx + 16]),vk,acc2);
                            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(&in_r[ox + kx + 24]),vk,acc3);
                        }
                    }
                    _mm256_storeu_ps(&out[oy * W + ox],acc0);
                    _mm256_storeu_ps(&out[oy * W + ox + 8],acc1);
                    _mm256_storeu_ps(&out[oy * W + ox + 16],acc2);
                    _mm256_storeu_ps(&out[oy * W + ox + 24],acc3);
                }
                for (; ox + 7 < ox_mx; ox += 8) {
                    __m256 acc = _mm256_setzero_ps();
                    for (int ky = 0; ky < K; ++ky) {
                        const float* in_r =&in[(oy + ky) * in_stride];
                        const float* ker_r =&ker[ky * K];
                        for (int kx = 0; kx < K; ++kx) {
                            const __m256 vk =_mm256_set1_ps(ker_r[kx]);
                            acc = _mm256_fmadd_ps(_mm256_loadu_ps(&in_r[ox + kx]),vk,acc);
                        }
                    }
                    _mm256_storeu_ps(&out[oy * W + ox], acc);
                }
                for (; ox < ox_mx; ++ox) {
                    float acc = 0.0f;
                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {
                            acc += in[(oy + ky) * in_stride + ox + kx] * ker[ky * K + kx];
                        }
                    }
                    out[oy * W + ox] = acc;
                }
            }
        }
    }
}
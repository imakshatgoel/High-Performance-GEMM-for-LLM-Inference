// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

static inline float hsum256(__m256 v){
    __m128 lo= _mm256_castps256_ps128(v);
    __m128 hi= _mm256_extractf128_ps(v,1);
    __m128 sum128= _mm_add_ps(lo,hi);
    __m128 shuf= _mm_movehdup_ps(sum128);
    __m128 sums= _mm_add_ps(sum128,shuf);
    shuf= _mm_movehl_ps(shuf,sums);
    sums= _mm_add_ss(sums,shuf);
    return _mm_cvtss_f32(sums);
}

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    int ksimd = K & ~7;
    int i=0;
    for(; i+3<M; i+=4){
        const float* a0= A+static_cast<long> (i+0)*lda;
        const float* a1= A+static_cast<long> (i+1)*lda;
        const float* a2= A+static_cast<long> (i+2)*lda;
        const float* a3= A+static_cast<long> (i+3)*lda;

        int j=0;
        for(; j+1<N; j+=2){
            const float* b0= B+static_cast<long> (j+0)*ldb;
            const float* b1= B+static_cast<long> (j+1)*ldb;

            __m256 acc00= _mm256_setzero_ps();
            __m256 acc01= _mm256_setzero_ps();
            __m256 acc10= _mm256_setzero_ps();
            __m256 acc11= _mm256_setzero_ps();
            __m256 acc20= _mm256_setzero_ps();
            __m256 acc21= _mm256_setzero_ps();
            __m256 acc30= _mm256_setzero_ps();
            __m256 acc31= _mm256_setzero_ps();

            for(int p=0; p<ksimd; p+=8){
                __m256 vb0= _mm256_loadu_ps(b0+p);
                __m256 vb1= _mm256_loadu_ps(b1+p);
                __m256 va0= _mm256_loadu_ps(a0+p);
                acc00= _mm256_fmadd_ps(va0,vb0,acc00);
                acc01= _mm256_fmadd_ps(va0,vb1,acc01);
                __m256 va1= _mm256_loadu_ps(a1+p);
                acc10= _mm256_fmadd_ps(va1,vb0,acc10);
                acc11= _mm256_fmadd_ps(va1,vb1,acc11);
                __m256 va2= _mm256_loadu_ps(a2+p);
                acc20= _mm256_fmadd_ps(va2,vb0,acc20);
                acc21= _mm256_fmadd_ps(va2,vb1,acc21);
                __m256 va3= _mm256_loadu_ps(a3 + p);
                acc30= _mm256_fmadd_ps(va3,vb0,acc30);
                acc31= _mm256_fmadd_ps(va3,vb1,acc31);
            }

            float s00= hsum256(acc00);
            float s01= hsum256(acc01);
            float s10= hsum256(acc10);
            float s11= hsum256(acc11);
            float s20= hsum256(acc20);
            float s21= hsum256(acc21);
            float s30= hsum256(acc30);
            float s31= hsum256(acc31);

            for(int p=ksimd; p<K; p++){
                float bp0 =b0[p];
                float bp1 =b1[p];
                s00+= a0[p]*bp0;
                s01+= a0[p]*bp1;
                s10+= a1[p]*bp0;
                s11+= a1[p]*bp1;
                s20+= a2[p]*bp0;
                s21+= a2[p]*bp1;
                s30+= a3[p]*bp0;
                s31+= a3[p]*bp1;
            }

            C[static_cast<long> (i+0)*ldc + (j+0)] =s00;
            C[static_cast<long> (i+0)*ldc + (j+1)] =s01;
            C[static_cast<long> (i+1)*ldc + (j+0)] =s10;
            C[static_cast<long> (i+1)*ldc + (j+1)] =s11;
            C[static_cast<long> (i+2)*ldc + (j+0)] =s20;
            C[static_cast<long> (i+2)*ldc + (j+1)] =s21;
            C[static_cast<long> (i+3)*ldc + (j+0)] =s30;
            C[static_cast<long> (i+3)*ldc + (j+1)] =s31;
        }

        for(; j<N; j++){
            const float* b0 = B + static_cast<long>(j) * ldb;
            __m256 acc0= _mm256_setzero_ps();
            __m256 acc1= _mm256_setzero_ps();
            __m256 acc2= _mm256_setzero_ps();
            __m256 acc3= _mm256_setzero_ps();

            for(int p=0; p<ksimd; p+=8){
                __m256 vb= _mm256_loadu_ps(b0+p);
                acc0= _mm256_fmadd_ps(_mm256_loadu_ps(a0+p),vb,acc0);
                acc1= _mm256_fmadd_ps(_mm256_loadu_ps(a1+p),vb,acc1);
                acc2= _mm256_fmadd_ps(_mm256_loadu_ps(a2+p),vb,acc2);
                acc3= _mm256_fmadd_ps(_mm256_loadu_ps(a3+p),vb,acc3);
            }

            float s0= hsum256(acc0);
            float s1= hsum256(acc1);
            float s2= hsum256(acc2);
            float s3= hsum256(acc3);

            for(int p=ksimd; p<K; p++){
                float bp=b0[p];
                s0+= a0[p]*bp;
                s1+= a1[p]*bp;
                s2+= a2[p]*bp;
                s3+= a3[p]*bp;
            }

            C[static_cast<long> (i+0)*ldc +j] =s0;
            C[static_cast<long> (i+1)*ldc +j] =s1;
            C[static_cast<long> (i+2)*ldc +j] =s2;
            C[static_cast<long> (i+3)*ldc +j] =s3;
        }
    }

    for(; i<M;i++){
        const float* a = A +static_cast<long>(i)*lda;
        for(int j=0; j<N;j++){
            const float* b=B + static_cast<long>(j)*ldb;
            __m256 acc= _mm256_setzero_ps();
            for(int p=0; p<ksimd; p+=8) acc=_mm256_fmadd_ps(_mm256_loadu_ps(a+p), _mm256_loadu_ps(b+p),acc);
            float s=hsum256(acc);
            for(int p=ksimd; p<K; p++) s+= a[p]*b[p];
            C[static_cast<long>(i)*ldc +j] =s;
        }
    }
}
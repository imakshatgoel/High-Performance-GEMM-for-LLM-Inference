// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

#include <immintrin.h>

#include "matmul.h"

static inline float hsum256opt(__m256 v){
    __m128 lo= _mm256_castps256_ps128(v);
    __m128 hi= _mm256_extractf128_ps(v,1);
    __m128 sum128= _mm_add_ps(lo, hi);
    __m128 shuf= _mm_movehdup_ps(sum128);
    __m128 sums= _mm_add_ps(sum128,shuf);
    shuf= _mm_movehl_ps(shuf,sums);
    sums= _mm_add_ss(sums,shuf);
    return _mm_cvtss_f32(sums);
}

static void mmgemv(const float* A, const float* B, float* C,
                        int N, int K, int lda, int ldb, int ldc){
    (void)lda;
    (void)ldc;
    const float* a0=A;
    const int ksimd= K& ~7;
    int j=0;

    for(; j+3<N; j+=4){
        const float* b0= B+ static_cast<long> (j+0)*ldb;
        const float* b1= B+ static_cast<long> (j+1)*ldb;
        const float* b2= B+ static_cast<long> (j+2)*ldb;
        const float* b3= B+ static_cast<long> (j+3)*ldb;

        __m256 acc0 =_mm256_setzero_ps();
        __m256 acc1 =_mm256_setzero_ps();
        __m256 acc2 =_mm256_setzero_ps();
        __m256 acc3 =_mm256_setzero_ps();

        for(int p=0; p<ksimd; p+=8){
            _mm_prefetch((const char*)(b0+p+64), _MM_HINT_T0);
            _mm_prefetch((const char*)(b1+p+64), _MM_HINT_T0);
            _mm_prefetch((const char*)(b2+p+64), _MM_HINT_T0);
            _mm_prefetch((const char*)(b3+p+64), _MM_HINT_T0);

            __m256 va =_mm256_loadu_ps(a0+p);
            acc0= _mm256_fmadd_ps(va,_mm256_loadu_ps(b0+p),acc0);
            acc1= _mm256_fmadd_ps(va,_mm256_loadu_ps(b1+p),acc1);
            acc2= _mm256_fmadd_ps(va,_mm256_loadu_ps(b2+p),acc2);
            acc3= _mm256_fmadd_ps(va,_mm256_loadu_ps(b3+p),acc3);
        }

        float s0= hsum256opt(acc0);
        float s1= hsum256opt(acc1);
        float s2= hsum256opt(acc2);
        float s3= hsum256opt(acc3);

        for(int p=ksimd; p<K; p++){
            float ap=a0[p];
            s0+= ap*b0[p];
            s1+= ap*b1[p];
            s2+= ap*b2[p];
            s3+= ap*b3[p];
        }

        C[j+0] =s0;
        C[j+1] =s1;
        C[j+2] =s2;
        C[j+3] =s3;
    }

    for(; j<N; j++){
        const float* b0 = B + static_cast<long>(j)*ldb;
        __m256 acc= _mm256_setzero_ps();
        for(int p=0; p<ksimd; p+=8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + p), _mm256_loadu_ps(b0 + p), acc);
        float s=hsum256opt(acc);
        for(int p=ksimd; p<K; ++p) s+= a0[p]*b0[p];
        C[j] =s;
    }
}

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    if(M==1){
        mmgemv(A,B,C,N,K,lda,ldb,ldc);
        return;
    }
    constexpr int NC=144;
    constexpr int MC=128;
    constexpr int PREFETCH_DIST=48;

    const int ksimd16= K& ~15;
    const int ksimd8= K& ~7;

    for(int j0=0; j0<N; j0+=NC){
        const int jmax=(j0+NC <N) ?(j0+NC) :N;
        for(int i0=0; i0<M; i0+=MC){
            int imax =(i0+MC <M) ?(i0+MC): M;
            int i=i0;
            for(; i+3<imax; i+=4){
                const float* a0= A+ static_cast<long> (i+0) *lda;
                const float* a1= A+ static_cast<long> (i+1) *lda;
                const float* a2= A+ static_cast<long> (i+2) *lda;
                const float* a3= A+ static_cast<long> (i+3) *lda;

                int j=j0;
                for(; j+2< jmax; j+=3){
                    const float* b0= B+ static_cast<long>(j+0)* ldb;
                    const float* b1= B+ static_cast<long>(j+1)* ldb;
                    const float* b2= B+ static_cast<long>(j+2)* ldb;

                    __m256 acc00= _mm256_setzero_ps();
                    __m256 acc01= _mm256_setzero_ps();
                    __m256 acc02= _mm256_setzero_ps();
                    __m256 acc10= _mm256_setzero_ps();
                    __m256 acc11= _mm256_setzero_ps();
                    __m256 acc12= _mm256_setzero_ps();
                    __m256 acc20= _mm256_setzero_ps();
                    __m256 acc21= _mm256_setzero_ps();
                    __m256 acc22= _mm256_setzero_ps();
                    __m256 acc30= _mm256_setzero_ps();
                    __m256 acc31= _mm256_setzero_ps();
                    __m256 acc32= _mm256_setzero_ps();

                    int p=0;
                    for(; p<ksimd16; p+=16){
                        _mm_prefetch((const char*)(b0+p+PREFETCH_DIST), _MM_HINT_T0);
                        _mm_prefetch((const char*)(b1+p+PREFETCH_DIST), _MM_HINT_T0);
                        _mm_prefetch((const char*)(b2+p+PREFETCH_DIST), _MM_HINT_T0);
                        __m256 vb0_0= _mm256_loadu_ps(b0+p);
                        __m256 vb1_0= _mm256_loadu_ps(b1+p);
                        __m256 vb2_0= _mm256_loadu_ps(b2+p);
                        __m256 va0_0= _mm256_loadu_ps(a0+p);

                        acc00= _mm256_fmadd_ps(va0_0,vb0_0,acc00);
                        acc01= _mm256_fmadd_ps(va0_0,vb1_0,acc01);
                        acc02= _mm256_fmadd_ps(va0_0,vb2_0,acc02);
                        __m256 va1_0= _mm256_loadu_ps(a1+p);

                        acc10= _mm256_fmadd_ps(va1_0,vb0_0,acc10);
                        acc11= _mm256_fmadd_ps(va1_0,vb1_0,acc11);
                        acc12= _mm256_fmadd_ps(va1_0,vb2_0,acc12);
                        __m256 va2_0 =_mm256_loadu_ps(a2+p);

                        acc20 =_mm256_fmadd_ps(va2_0,vb0_0,acc20);
                        acc21 =_mm256_fmadd_ps(va2_0,vb1_0,acc21);
                        acc22 =_mm256_fmadd_ps(va2_0,vb2_0,acc22);
                        __m256 va3_0= _mm256_loadu_ps(a3+p);

                        acc30= _mm256_fmadd_ps(va3_0,vb0_0,acc30);
                        acc31= _mm256_fmadd_ps(va3_0,vb1_0,acc31);
                        acc32= _mm256_fmadd_ps(va3_0,vb2_0,acc32);

                        __m256 vb0_1= _mm256_loadu_ps(b0+p+8);
                        __m256 vb1_1= _mm256_loadu_ps(b1+p+8);
                        __m256 vb2_1= _mm256_loadu_ps(b2+p+8);
                        __m256 va0_1= _mm256_loadu_ps(a0+p+8);

                        acc00= _mm256_fmadd_ps(va0_1,vb0_1,acc00);
                        acc01= _mm256_fmadd_ps(va0_1,vb1_1,acc01);
                        acc02= _mm256_fmadd_ps(va0_1,vb2_1,acc02);
                        __m256 va1_1= _mm256_loadu_ps(a1+p+8);

                        acc10= _mm256_fmadd_ps(va1_1,vb0_1,acc10);
                        acc11= _mm256_fmadd_ps(va1_1,vb1_1,acc11);
                        acc12= _mm256_fmadd_ps(va1_1,vb2_1,acc12);
                        __m256 va2_1 =_mm256_loadu_ps(a2+p +8);

                        acc20 =_mm256_fmadd_ps(va2_1,vb0_1,acc20);
                        acc21 =_mm256_fmadd_ps(va2_1,vb1_1,acc21);
                        acc22 =_mm256_fmadd_ps(va2_1,vb2_1,acc22);
                        __m256 va3_1= _mm256_loadu_ps(a3+p+8);

                        acc30= _mm256_fmadd_ps(va3_1,vb0_1,acc30);
                        acc31= _mm256_fmadd_ps(va3_1,vb1_1,acc31);
                        acc32= _mm256_fmadd_ps(va3_1,vb2_1,acc32);
                    }

                    if(p<ksimd8){
                        __m256 vb0= _mm256_loadu_ps(b0+p);
                        __m256 vb1= _mm256_loadu_ps(b1+p);
                        __m256 vb2= _mm256_loadu_ps(b2+p);
                        __m256 va0= _mm256_loadu_ps(a0+p);

                        acc00= _mm256_fmadd_ps(va0,vb0,acc00);
                        acc01= _mm256_fmadd_ps(va0,vb1,acc01);
                        acc02= _mm256_fmadd_ps(va0,vb2,acc02);
                        __m256 va1= _mm256_loadu_ps(a1+p);

                        acc10= _mm256_fmadd_ps(va1,vb0,acc10);
                        acc11= _mm256_fmadd_ps(va1,vb1,acc11);
                        acc12= _mm256_fmadd_ps(va1,vb2,acc12);
                        __m256 va2 =_mm256_loadu_ps(a2+p);

                        acc20 =_mm256_fmadd_ps(va2,vb0,acc20);
                        acc21 =_mm256_fmadd_ps(va2,vb1,acc21);
                        acc22 =_mm256_fmadd_ps(va2,vb2,acc22);
                        __m256 va3= _mm256_loadu_ps(a3+p);

                        acc30= _mm256_fmadd_ps(va3,vb0,acc30);
                        acc31= _mm256_fmadd_ps(va3,vb1,acc31);
                        acc32= _mm256_fmadd_ps(va3,vb2,acc32);
                        p+=8;
                    }

                    float s00=hsum256opt(acc00);
                    float s01=hsum256opt(acc01);
                    float s02=hsum256opt(acc02);
                    float s10=hsum256opt(acc10);
                    float s11=hsum256opt(acc11);
                    float s12=hsum256opt(acc12);
                    float s20=hsum256opt(acc20);
                    float s21=hsum256opt(acc21);
                    float s22=hsum256opt(acc22);
                    float s30=hsum256opt(acc30);
                    float s31=hsum256opt(acc31);
                    float s32=hsum256opt(acc32);

                    for(; p<K; p++){
                        float bp0=b0[p], bp1=b1[p], bp2=b2[p];
                        s00 += a0[p]*bp0; s01+=a0[p]*bp1; s02+= a0[p]*bp2;
                        s10 += a1[p]*bp0; s11+=a1[p]*bp1; s12+= a1[p]*bp2;
                        s20 += a2[p]*bp0; s21+=a2[p]*bp1; s22+= a2[p]*bp2;
                        s30 += a3[p]*bp0; s31+=a3[p]*bp1; s32+= a3[p]*bp2;
                    }

                    C[static_cast<long>(i+0) *ldc +(j+0)]= s00;
                    C[static_cast<long>(i+0) *ldc +(j+1)]= s01;
                    C[static_cast<long>(i+0) *ldc +(j+2)]= s02;
                    C[static_cast<long>(i+1) *ldc +(j+0)]= s10;
                    C[static_cast<long>(i+1) *ldc +(j+1)]= s11;
                    C[static_cast<long>(i+1) *ldc +(j+2)]= s12;
                    C[static_cast<long>(i+2) *ldc +(j+0)]= s20;
                    C[static_cast<long>(i+2) *ldc +(j+1)]= s21;
                    C[static_cast<long>(i+2) *ldc +(j+2)]= s22;
                    C[static_cast<long>(i+3) *ldc +(j+0)]= s30;
                    C[static_cast<long>(i+3) *ldc +(j+1)]= s31;
                    C[static_cast<long>(i+3) *ldc +(j+2)]= s32;
                }

                for(; j<jmax; j++){
                    const float* b0= B + static_cast<long>(j)*ldb;
                    __m256 acc0= _mm256_setzero_ps();
                    __m256 acc1= _mm256_setzero_ps();
                    __m256 acc2= _mm256_setzero_ps();
                    __m256 acc3= _mm256_setzero_ps();

                    for(int p=0; p<ksimd8; p+=8){
                        __m256 vb=_mm256_loadu_ps(b0+p);
                        acc0= _mm256_fmadd_ps(_mm256_loadu_ps(a0+p),vb,acc0);
                        acc1= _mm256_fmadd_ps(_mm256_loadu_ps(a1+p),vb,acc1);
                        acc2= _mm256_fmadd_ps(_mm256_loadu_ps(a2+p),vb,acc2);
                        acc3= _mm256_fmadd_ps(_mm256_loadu_ps(a3+p),vb,acc3);
                    }

                    float s0=hsum256opt(acc0);
                    float s1=hsum256opt(acc1);
                    float s2=hsum256opt(acc2);
                    float s3=hsum256opt(acc3);

                    for(int p=ksimd8; p<K; p++){
                        float bp=b0[p];
                        s0+= a0[p]* bp;
                        s1+= a1[p]* bp;
                        s2+= a2[p]* bp;
                        s3+= a3[p]* bp;
                    }

                    C[static_cast<long>(i+0)* ldc+j]= s0;
                    C[static_cast<long>(i+1)* ldc+j]= s1;
                    C[static_cast<long>(i+2)* ldc+j]= s2;
                    C[static_cast<long>(i+3)* ldc+j]= s3;
                }
            }

            for(; i<imax; i++){
                const float* a = A + static_cast<long>(i)*lda;
                for(int j=j0; j<jmax; j++){
                    const float* b = B+static_cast<long>(j)* ldb;
                    __m256 acc=_mm256_setzero_ps();
                    for(int p=0; p<ksimd8; p+=8) acc= _mm256_fmadd_ps(_mm256_loadu_ps(a+p), _mm256_loadu_ps(b+p),acc);
                    float s=hsum256opt(acc);
                    for(int p=ksimd8; p<K; p++) s+= a[p]*b[p];
                    C[static_cast<long>(i)*ldc +j]= s;
                }
            }
        }
    }
}
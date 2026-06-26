#pragma once
#include <cblas.h>
#include "tensor.h"
#include <iostream>
#include <immintrin.h>
#include <experimental/simd>

namespace Forge {
    template <typename DTYPE>
    void forge_contract(const DTYPE* A_ptr, const DTYPE* B_ptr, DTYPE* C, int M, int N, int K, bool transA,
        bool transB, bool rowMajor, DTYPE alpha=1.0,  DTYPE beta=0.0);

}

template<typename ReturnType, typename... Args>
auto Forge::select_optimal_kernel(ReturnType (*AVX2_fn)(Args...), ReturnType (*AVX512_fn)(Args...),
    ReturnType (*FALLBACK_fn)(Args...)) -> ReturnType(*)(Args...) {
#if defined (__x86_64__) || defined(_M_X64)
    #if defined(__GNUC__) || defined(__clang__)
        if (__builtin_cpu_supports("avx512f")) return AVX512_fn;
        if (__builtin_cpu_supports("avx2")) return AVX2_fn;
        return FALLBACK_fn;
    #elif defined(_MSC_VER)
        int cpuInfo[4] = {0};
        __cpuid(cpuInfo, 1);
        bool osxsave = (cpuInfo[2] & (1 << 27)) != 0;
        if (osxsave) {
            __cpuid(cpuInfo, 7);
            bool has_avx512 = (cpuInfo[1] & (1 << 16)) != 0;
            bool has_avx2   = (cpuInfo[1] & (1 << 5)) != 0;
            if (has_avx512) return AVX512_fn;
            if (has_avx2) return AVX2_fn;
        }
        return FALLBACK_fn;
    #endif
#endif
    return FALLBACK_fn;
}

template<typename DTYPE>
void Forge::forge_contract(const DTYPE* A_ptr, const DTYPE* B_ptr, DTYPE* C, int M, int N, int K, bool transA,
    bool transB, bool rowMajor, const DTYPE alpha, const DTYPE beta) {

    int lda {transA?M:K}, ldb {transB?K:N}, ldc {N};
    if (!rowMajor) {
        lda = transA?K:M;
        ldb = transB?N:K;
        ldc = M;
    }

    if constexpr (std::is_same_v<std::decay_t<DTYPE>, float>) {
        cblas_sgemm(
            rowMajor?CblasRowMajor : CblasColMajor,
            transA?CblasTrans : CblasNoTrans,
            transB?CblasTrans : CblasNoTrans,
            M, N, K, alpha, A_ptr, lda , B_ptr, ldb, beta, C, ldc);
    } else if constexpr (std::is_same_v<std::decay_t<DTYPE>, double>) {
        cblas_dgemm(
            rowMajor?CblasRowMajor : CblasColMajor,
            transA?CblasTrans : CblasNoTrans,
            transB?CblasTrans : CblasNoTrans,
            M, N, K, alpha, A_ptr, lda , B_ptr, ldb, beta, C, ldc);
    } else {
        throw std::runtime_error("Unsupported DTYPE For Contraction");
    }
}

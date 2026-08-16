#pragma once
#include <cblas.h>
#include <type_traits>
#include <stdexcept>

namespace Forge {
    template <typename DTYPE>
    void forge_contract(const DTYPE* A_ptr, const DTYPE* B_ptr, DTYPE* C, int M, int N, int K, bool transA,
    bool transB, bool rowMajor, DTYPE alpha=1.0,  DTYPE beta=0.0);
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
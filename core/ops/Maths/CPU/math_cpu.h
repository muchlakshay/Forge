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

    struct ExeCTX {
        int m_threads{static_cast<int>(std::thread::hardware_concurrency()/2)};
        explicit ExeCTX(int threads) : m_threads(threads) {
            openblas_set_num_threads(threads);
            omp_set_num_threads(threads);
        }
        ExeCTX() = default;
        ExeCTX(const ExeCTX&) = delete;
        ExeCTX(ExeCTX&&) = delete;
        ExeCTX& operator=(const ExeCTX&) = delete;
        ExeCTX& operator=(ExeCTX&&) = delete;
        void set_num_threads(int threads) {
            m_threads = threads;
            openblas_set_num_threads(threads);
            omp_set_num_threads(threads);
        }
        [[nodiscard]] auto get_num_threads() const {return m_threads;}
    };

    inline ExeCTX forge_exeCTX;
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


